import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import logging
import math
from pathlib import Path
from threading import RLock
import cv2
import numpy as np

MODEL_NAME = "SFace"
DETECTOR_BACKEND = "yunet"
EMBEDDING_SCHEMA = "opencv-sface-yunet-alignCrop-v1"
ANTI_SPOOFING = False
MODEL_DIR = Path(os.getenv("FACE_MODEL_DIR", str(Path(__file__).resolve().parent / "models")))
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.363"))
if not math.isfinite(FACE_MATCH_THRESHOLD) or not -1 <= FACE_MATCH_THRESHOLD <= 1:
    raise ValueError("FACE_MATCH_THRESHOLD must be a finite cosine similarity between -1 and 1")
DISTANCE_THRESHOLD = 1.0 - FACE_MATCH_THRESHOLD
INFERENCE_LOCK = RLock()
logger = logging.getLogger("uvicorn.error")
recognizer = None
detector = None


def preload_model():
    global recognizer, detector
    try:
        with INFERENCE_LOCK:
            if model_ready():
                return
            cv2.setNumThreads(1)
            cv2.ocl.setUseOpenCL(False)
            new_recognizer = cv2.FaceRecognizerSF.create(
                str(MODEL_DIR / "face_recognition_sface_2021dec.onnx"), "",
                cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
            new_detector = cv2.FaceDetectorYN.create(
                str(MODEL_DIR / "face_detection_yunet_2023mar.onnx"), "", (320, 320),
                0.9, 0.3, 500, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
            # Warm lazy native allocations before reporting readiness.
            new_detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
            new_recognizer.feature(np.zeros((112, 112, 3), dtype=np.uint8))
            recognizer, detector = new_recognizer, new_detector
            logger.info("Loaded OpenCV SFace + YuNet; similarity threshold=%s; liveness=False", FACE_MATCH_THRESHOLD)
    except Exception:
        logger.exception("Native model preload failed; run python preload.py at build time and restart")


def model_ready():
    return recognizer is not None and detector is not None


def embedding(image):
    with INFERENCE_LOCK:
        if not model_ready():
            raise RuntimeError("Face models are not loaded")
        # Fixed detector input bounds DNN allocations; align on the higher resolution image.
        height, width = image.shape[:2]
        scale = min(320 / width, 320 / height)
        resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
        canvas = np.zeros((320, 320, 3), dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        _, faces = detector.detect(canvas)
        if faces is None or len(faces) != 1:
            raise ValueError("Exactly one detectable face is required")
        face = faces[0].copy()
        face[:14:2] *= width / resized.shape[1]
        face[1:14:2] *= height / resized.shape[0]
        if not np.isfinite(face).all() or min(face[2], face[3]) < 40:
            raise ValueError("Face is too small or invalid")
        aligned = recognizer.alignCrop(image, face)
        feature = recognizer.feature(aligned).reshape(-1)
        if feature.shape != (128,) or not np.isfinite(feature).all() or np.linalg.norm(feature) == 0:
            raise RuntimeError("Invalid recognition output")
        return feature.tolist()


def cosine_distance(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    for vector in (a, b):
        if vector.shape != (128,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise RuntimeError("Invalid native SFace embedding")
    # OpenCV match normalizes its inputs in place; arrays here are private copies.
    with INFERENCE_LOCK:
        if recognizer is None:
            raise RuntimeError("Face recognizer is not loaded")
        similarity = recognizer.match(a.reshape(1, -1).copy(), b.reshape(1, -1).copy(),
                                      cv2.FaceRecognizerSF_FR_COSINE)
    return 1.0 - float(np.clip(similarity, -1.0, 1.0))


def verify_images(image1, image2):
    distance = cosine_distance(embedding(image1), embedding(image2))
    return {"verified": distance <= DISTANCE_THRESHOLD, "distance": distance,
            "threshold": DISTANCE_THRESHOLD, "model": MODEL_NAME,
            "detector_backend": DETECTOR_BACKEND, "similarity_metric": "cosine"}

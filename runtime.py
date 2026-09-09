import os

# Must precede DeepFace/TensorFlow/numpy imports.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("DEEPFACE_HOME", os.path.dirname(os.path.abspath(__file__)))

import logging
from threading import RLock

MODEL_NAME = "SFace"
DETECTOR_BACKEND = "opencv"
ANTI_SPOOFING = False  # Existing behavior; no extra spoofing model is loaded.
INFERENCE_LOCK = RLock()
logger = logging.getLogger("uvicorn.error")
_model_ready = False


def preload_model():
    global _model_ready
    try:
        with INFERENCE_LOCK:
            if _model_ready:
                return
            import cv2
            from deepface import DeepFace
            cv2.setNumThreads(1)
            DeepFace.build_model(MODEL_NAME)
            DeepFace.build_model(DETECTOR_BACKEND, task="face_detector")
            _model_ready = True
            logger.info("Loaded model=%s detector=%s anti_spoofing=%s", MODEL_NAME, DETECTOR_BACKEND, ANTI_SPOOFING)
    except Exception:
        logger.exception("Face model preload failed; inference unavailable until restart")


def model_ready():
    return _model_ready

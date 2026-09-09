import json
from datetime import datetime, timezone
from pathlib import Path

from runtime import MODEL_NAME, DETECTOR_BACKEND, ANTI_SPOOFING

import numpy as np
from deepface import DeepFace


FACE_DATA_DIR = Path("face-data")
FACE_DATABASE_PATH = FACE_DATA_DIR / "embeddings.json"


# DeepFace 0.0.100 SFace cosine threshold; validate on your enrollment data.
DUPLICATE_THRESHOLD = 0.593


def ensure_database():
    FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not FACE_DATABASE_PATH.exists():
        FACE_DATABASE_PATH.write_text(
            json.dumps({"model": MODEL_NAME, "subjects": []}, indent=2),
            encoding="utf-8",
        )


def load_database():
    ensure_database()

    with FACE_DATABASE_PATH.open("r", encoding="utf-8") as file:
        database = json.load(file)
    if database.get("subjects") and database.get("model") != MODEL_NAME:
        raise RuntimeError("Legacy face database requires re-enrollment with SFace; back up embeddings.json first")
    database["model"] = MODEL_NAME
    return database


def save_database(database):
    ensure_database()

    with FACE_DATABASE_PATH.open("w", encoding="utf-8") as file:
        json.dump(database, file, indent=2)


def generate_embedding(image_path):
    representations = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        align=True,
        anti_spoofing=ANTI_SPOOFING,
    )

    if not representations:
        raise ValueError("No face representation could be generated.")

    embedding = representations[0].get("embedding")

    if not embedding:
        raise ValueError("Face embedding was empty.")

    return embedding


def cosine_distance(embedding_a, embedding_b):
    a = np.asarray(embedding_a, dtype=np.float32)
    b = np.asarray(embedding_b, dtype=np.float32)

    if a.shape != (128,) or b.shape != (128,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("Invalid SFace embedding in face database")

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        raise ValueError("Invalid face embedding.")

    similarity = np.dot(a, b) / denominator

    return float(1.0 - similarity)


def search_duplicate(candidate_embedding):
    database = load_database()

    best_match = None

    for subject in database["subjects"]:
        distance = cosine_distance(
            candidate_embedding,
            subject["embedding"],
        )

        if best_match is None or distance < best_match["distance"]:
            best_match = {
                "subject_id": subject["subject_id"],
                "distance": distance,
            }

    if best_match is None:
        return {
            "matched": False,
            "distance": None,
            "threshold": DUPLICATE_THRESHOLD,
        }

    return {
        "matched": best_match["distance"] <= DUPLICATE_THRESHOLD,
        "distance": best_match["distance"],
        "threshold": DUPLICATE_THRESHOLD,
    }


def enroll_subject(subject_id, embedding):
    database = load_database()

    # Replace an existing enrollment for the same subject ID.
    database["subjects"] = [
        subject
        for subject in database["subjects"]
        if subject["subject_id"] != subject_id
    ]

    database["subjects"].append(
        {
            "subject_id": subject_id,
            "embedding": embedding,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    save_database(database)
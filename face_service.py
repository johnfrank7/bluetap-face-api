import json
from datetime import datetime, timezone
from pathlib import Path

from runtime import MODEL_NAME, EMBEDDING_SCHEMA, DISTANCE_THRESHOLD, embedding, cosine_distance


FACE_DATA_DIR = Path("face-data")
FACE_DATABASE_PATH = FACE_DATA_DIR / "embeddings.json"


# Same cosine distance threshold as pair verification.
DUPLICATE_THRESHOLD = DISTANCE_THRESHOLD


def ensure_database():
    FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not FACE_DATABASE_PATH.exists():
        FACE_DATABASE_PATH.write_text(
            json.dumps({"model": MODEL_NAME, "embedding_schema": EMBEDDING_SCHEMA, "subjects": []}, indent=2),
            encoding="utf-8",
        )


def load_database():
    ensure_database()

    with FACE_DATABASE_PATH.open("r", encoding="utf-8") as file:
        database = json.load(file)
    if database.get("subjects") and (database.get("model") != MODEL_NAME or database.get("embedding_schema") != EMBEDDING_SCHEMA):
        raise RuntimeError("Legacy face database requires re-enrollment with SFace; back up embeddings.json first")
    database["model"] = MODEL_NAME
    database["embedding_schema"] = EMBEDDING_SCHEMA
    return database


def save_database(database):
    ensure_database()

    with FACE_DATABASE_PATH.open("w", encoding="utf-8") as file:
        json.dump(database, file, indent=2)


def generate_embedding(image):
    return embedding(image)


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
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from runtime import MODEL_NAME, EMBEDDING_SCHEMA, DISTANCE_THRESHOLD, embedding, cosine_distance


FACE_DATA_DIR = Path("face-data")
FACE_DATABASE_PATH = FACE_DATA_DIR / "embeddings.json"
REGISTRATION_FACE_TTL_SECONDS = int(os.getenv("REGISTRATION_FACE_TTL_SECONDS", "3600"))


# Same cosine distance threshold as pair verification.
DUPLICATE_THRESHOLD = DISTANCE_THRESHOLD


def ensure_database():
    FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not FACE_DATABASE_PATH.exists():
        FACE_DATABASE_PATH.write_text(
            json.dumps({
                "model": MODEL_NAME,
                "embedding_schema": EMBEDDING_SCHEMA,
                "subjects": [],
                "temporary_registrations": [],
            }, indent=2),
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
    database.setdefault("subjects", [])
    database.setdefault("temporary_registrations", [])
    return database


def save_database(database):
    ensure_database()

    with FACE_DATABASE_PATH.open("w", encoding="utf-8") as file:
        json.dump(database, file, indent=2)


def _now():
    return datetime.now(timezone.utc)


def _parse_timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (AttributeError, TypeError, ValueError):
        return None


def purge_expired_registrations(database):
    """Remove only expired temporary registration embeddings.

    Finalized enrollments are deliberately never considered by this routine.
    """
    now = _now()
    original = database["temporary_registrations"]
    database["temporary_registrations"] = [
        record for record in original
        if (expires_at := _parse_timestamp(record.get("expires_at"))) is not None and expires_at > now
    ]
    return len(database["temporary_registrations"]) != len(original)


def generate_embedding(image):
    return embedding(image)


def search_duplicate(candidate_embedding):
    database = load_database()
    if purge_expired_registrations(database):
        save_database(database)
    if any(subject.get("finalized") is not True for subject in database["subjects"]):
        raise RuntimeError("Legacy face records must be reconciled as finalized before duplicate checks can continue")
    return search_duplicate_in_database(database, candidate_embedding)


def store_registration_face(registration_session_id, embedding):
    """Store an expiring, non-searchable registration embedding by session ID."""
    database = load_database()
    purge_expired_registrations(database)
    expires_at = _now() + timedelta(seconds=REGISTRATION_FACE_TTL_SECONDS)

    database["temporary_registrations"] = [
        record for record in database["temporary_registrations"]
        if record.get("registration_session_id") != registration_session_id
    ]
    database["temporary_registrations"].append(
        {
            "registration_session_id": registration_session_id,
            "embedding": embedding,
            "created_at": _now().isoformat(),
            "expires_at": expires_at.isoformat(),
        }
    )
    save_database(database)
    return {"registrationSessionId": registration_session_id, "expiresAt": expires_at.isoformat()}


def discard_registration_face(registration_session_id):
    """Safely delete one temporary record; finalized subjects are never touched."""
    database = load_database()
    purge_expired_registrations(database)
    original_count = len(database["temporary_registrations"])
    database["temporary_registrations"] = [
        record for record in database["temporary_registrations"]
        if record.get("registration_session_id") != registration_session_id
    ]
    deleted = len(database["temporary_registrations"]) != original_count
    save_database(database)
    return deleted


def finalize_registration_face(uid, registration_session_id):
    """Atomically promote an unexpired temporary registration face to a UID."""
    database = load_database()
    purge_expired_registrations(database)
    if any(subject.get("finalized") is not True for subject in database["subjects"]):
        raise RuntimeError("Legacy face records must be reconciled as finalized before enrollment can continue")
    temporary = next(
        (record for record in database["temporary_registrations"]
         if record.get("registration_session_id") == registration_session_id),
        None,
    )
    if temporary is None:
        save_database(database)
        raise ValueError("The registration face reference is missing or expired.")

    duplicate = search_duplicate_in_database(database, temporary["embedding"], exclude_uid=uid)
    if duplicate["matched"]:
        save_database(database)
        return {"enrolled": False, "duplicate": duplicate}

    # A retry for the same finalized UID replaces only that UID's enrollment.
    database["subjects"] = [
        subject
        for subject in database["subjects"]
        if subject.get("subject_id") != uid
    ]
    database["subjects"].append(
        {
            "subject_id": uid,
            "embedding": temporary["embedding"],
            "finalized": True,
            "registration_session_id": registration_session_id,
            "created_at": _now().isoformat(),
        }
    )

    database["temporary_registrations"] = [
        record for record in database["temporary_registrations"]
        if record.get("registration_session_id") != registration_session_id
    ]

    save_database(database)
    return {"enrolled": True}


def search_duplicate_in_database(database, candidate_embedding, exclude_uid=None):
    best_match = None
    for subject in database["subjects"]:
        if subject.get("finalized") is not True or subject.get("subject_id") == exclude_uid:
            continue
        distance = cosine_distance(candidate_embedding, subject["embedding"])
        if best_match is None or distance < best_match["distance"]:
            best_match = {"subject_id": subject["subject_id"], "distance": distance}
    if best_match is None:
        return {"matched": False, "distance": None, "threshold": DUPLICATE_THRESHOLD}
    return {
        "matched": best_match["distance"] <= DUPLICATE_THRESHOLD,
        "distance": best_match["distance"],
        "threshold": DUPLICATE_THRESHOLD,
    }

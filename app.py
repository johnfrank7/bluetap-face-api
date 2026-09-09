import os
import tempfile
from pathlib import Path

from deepface import DeepFace
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from face_service import (
    enroll_subject,
    generate_embedding,
    search_duplicate,
)

app = FastAPI(
    title="BlueTap Face Verification API",
    version="1.0.0",
)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

DEEPFACE_API_KEY = os.getenv(
    "DEEPFACE_API_KEY",
    "bluetap-local-dev-key",
)

security = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key.",
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication scheme.",
        )

    if credentials.credentials != DEEPFACE_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    return True


@app.get("/")
def root():
    return {
        "service": "BlueTap Face Verification API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post("/verify-face")
async def verify_face(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    _: bool = Depends(require_api_key),
):
    if image1.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="image1 must be a supported image file.",
        )

    if image2.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="image2 must be a supported image file.",
        )

    temp_paths = []

    try:
        for upload in (image1, image2):
            suffix = Path(upload.filename or "image.jpg").suffix or ".jpg"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:
                temp_file.write(await upload.read())
                temp_paths.append(temp_file.name)

        result = DeepFace.verify(
            img1_path=temp_paths[0],
            img2_path=temp_paths[1],
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True,
        )

        return {
            "verified": bool(result.get("verified", False)),
            "distance": result.get("distance"),
            "threshold": result.get("threshold"),
            "model": result.get("model"),
            "detector_backend": result.get("detector_backend"),
            "similarity_metric": result.get("similarity_metric"),
        }

    except ValueError as error:
        underlying_error = error.__cause__

        detail = (
            str(underlying_error)
            if underlying_error is not None
            else str(error)
        )

        print("DeepFace ValueError:", repr(error))
        print("Underlying error:", repr(underlying_error))

        raise HTTPException(
            status_code=400,
            detail=detail,
        )

    except Exception as error:
        print("Face verification error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Face verification failed.",
        )

    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass


@app.post("/check-duplicate")
async def check_duplicate(
    image: UploadFile = File(...),
    _: bool = Depends(require_api_key),
):
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="A supported face image is required.",
        )

    temp_path = None

    try:
        suffix = Path(image.filename or "face.jpg").suffix or ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(await image.read())
            temp_path = temp_file.name

        embedding = generate_embedding(temp_path)

        result = search_duplicate(embedding)

        return {
            "duplicateDetected": result["matched"],
            "distance": result["distance"],
            "threshold": result["threshold"],
            "reviewRequired": result["matched"],
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        print("Duplicate search error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Duplicate face check failed.",
        )

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.post("/enroll-face")
async def enroll_face(
    subject_id: str = Form(...),
    image: UploadFile = File(...),
    _: bool = Depends(require_api_key),
):
    subject_id = subject_id.strip()

    if not subject_id:
        raise HTTPException(
            status_code=400,
            detail="subject_id is required.",
        )

    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="A supported face image is required.",
        )

    temp_path = None

    try:
        suffix = Path(image.filename or "face.jpg").suffix or ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(await image.read())
            temp_path = temp_file.name

        embedding = generate_embedding(temp_path)

        duplicate = search_duplicate(embedding)

        if duplicate["matched"]:
            return {
                "enrolled": False,
                "duplicateDetected": True,
                "reviewRequired": True,
                "distance": duplicate["distance"],
                "threshold": duplicate["threshold"],
            }

        enroll_subject(
            subject_id=subject_id,
            embedding=embedding,
        )

        return {
            "enrolled": True,
            "duplicateDetected": False,
            "reviewRequired": False,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        print("Enrollment error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Face enrollment failed.",
        )

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
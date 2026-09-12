import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager

from runtime import (MODEL_NAME, DETECTOR_BACKEND, ANTI_SPOOFING,
                     INFERENCE_LOCK, model_ready, preload_model, verify_images)
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from image_utils import prepare_image
from config import development_face_reset_enabled, development_face_store_allowed

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    logger.info("Development face reset enabled: %s",
                str(development_face_reset_enabled()).lower())
    logger.info("Face data environment: %s",
                "development" if development_face_store_allowed() else "non-development")
    task = asyncio.create_task(run_in_threadpool(preload_model))
    yield
    await task


app = FastAPI(title="BlueTap Face Verification API", version="1.0.0", lifespan=lifespan)
DEEPFACE_API_KEY = os.getenv("DEEPFACE_API_KEY", "bluetap-local-dev-key")
security = HTTPBearer(auto_error=False)
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


class UploadLimitMiddleware:
    """Bound the entire multipart body before parsing, including chunked uploads."""
    def __init__(self, app):
        self.app = app
        self.busy = False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        if self.busy:
            return await JSONResponse(
                {"detail": "Face service is busy. Please retry later."}, status_code=503
            )(scope, receive, send)
        self.busy = True
        total = 0
        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > 11 * 1024 * 1024:
                # Drain without buffering so clients receive the JSON 413 reliably.
                while message.get("more_body", False):
                    message = await receive()
                raise HTTPException(413, "Request exceeds 11 MiB.")
            return message
        try:
            await self.app(scope, limited_receive, send)
        finally:
            self.busy = False


app.add_middleware(UploadLimitMiddleware)


def require_api_key(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    if credentials is None:
        raise HTTPException(401, "Missing API key.")
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Invalid authentication scheme.")
    if credentials.credentials != DEEPFACE_API_KEY:
        raise HTTPException(401, "Invalid API key.")
    return True


@app.get("/")
def root():
    return {"service": "BlueTap Face Verification API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/ready")
async def ready():
    loaded = model_ready()
    return JSONResponse({"status": "ready" if loaded else "not_ready", "model": MODEL_NAME,
                         "recognizer": "OpenCV SFace", "detector": "YuNet", "modelLoaded": loaded,
                         "developmentFaceResetEnabled": development_face_reset_enabled(),
                         "faceDataEnvironment": "development" if development_face_store_allowed() else "non-development"},
                        status_code=200 if loaded else 503)


UID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
REGISTRATION_SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


class DevelopmentResetRequest(BaseModel):
    dryRun: bool
    confirm: str | None = None


def process(operation, uploads=(), uid=None, registration_session_id=None,
            dry_run=True, confirmation=None):
    if operation in {"verify", "duplicate", "store"} and not model_ready():
        raise HTTPException(503, "Face model is not ready. Please retry later.")
    if not INFERENCE_LOCK.acquire(blocking=False):
        raise HTTPException(503, "Face service is busy. Please retry later.")
    try:
        for upload in uploads:
            if upload.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(400, "A supported face image is required.")
        images = [prepare_image(upload) for upload in uploads]
        from face_service import (discard_registration_face, finalize_registration_face,
                                  generate_embedding, search_duplicate,
                                  store_registration_face, reset_development_face_storage)
        if operation == "development-reset":
            return reset_development_face_storage(dry_run, confirmation)
        if operation == "discard":
            return {"discarded": discard_registration_face(registration_session_id)}
        if operation == "finalize":
            result = finalize_registration_face(uid, registration_session_id)
            if not result["enrolled"]:
                duplicate = result["duplicate"]
                return {"enrolled": False, "duplicateDetected": True,
                        "distance": duplicate["distance"], "threshold": duplicate["threshold"],
                        "reviewRequired": True}
            return {"enrolled": True, "duplicateDetected": False, "reviewRequired": False}
        if operation == "verify":
            return verify_images(images[0], images[1])
        embedding = generate_embedding(images[0])
        result = search_duplicate(embedding)
        response = {"duplicateDetected": result["matched"], "distance": result["distance"],
                    "threshold": result["threshold"], "reviewRequired": result["matched"]}
        if operation == "duplicate":
            return response
        if result["matched"]:
            return {"stored": False, **response}
        temporary = store_registration_face(registration_session_id, embedding)
        return {"stored": True, "duplicateDetected": False, "reviewRequired": False, **temporary}
    except HTTPException:
        raise
    except (PermissionError, ValueError) as error:
        if operation == "development-reset":
            logger.warning("Development face reset rejected: %s", error)
            raise HTTPException(403, str(error)) from None
        if operation == "finalize":
            logger.info("Face finalization rejected: %s", error)
            raise HTTPException(409, str(error)) from None
        logger.exception("Face %s rejected an image", operation)
        raise HTTPException(400, "Could not process the face image. Use a clear photo with a visible face.") from None
    except Exception:
        logger.exception("Face %s failed", operation)
        raise HTTPException(500, {"verify": "Face verification failed.",
                                  "duplicate": "Duplicate face check failed.",
                                  "store": "Temporary registration face storage failed.",
                                  "finalize": "Face enrollment failed.",
                                  "discard": "Temporary registration cleanup failed.",
                                  "development-reset": "Development face reset failed."}[operation]) from None
    finally:
        INFERENCE_LOCK.release()


async def handle(operation, uploads=(), uid=None, registration_session_id=None,
                 dry_run=True, confirmation=None):
    try:
        return await run_in_threadpool(
            process, operation, uploads, uid, registration_session_id, dry_run, confirmation
        )
    finally:
        # Closes and deletes Starlette's spooled temporary upload files.
        for upload in uploads:
            await upload.close()


@app.post("/verify-face")
async def verify_face(image1: UploadFile = File(...), image2: UploadFile = File(...),
                      _: bool = Depends(require_api_key)):
    return await handle("verify", [image1, image2])


@app.post("/check-duplicate")
async def check_duplicate(image: UploadFile = File(...), _: bool = Depends(require_api_key)):
    return await handle("duplicate", [image])


def require_registration_session_id(registration_session_id: str):
    registration_session_id = registration_session_id.strip()
    if not REGISTRATION_SESSION_PATTERN.fullmatch(registration_session_id):
        raise HTTPException(400, "A valid registrationSessionId is required.")
    return registration_session_id


def require_finalization_identifiers(uid: str, registration_session_id: str):
    uid = uid.strip()
    if not UID_PATTERN.fullmatch(uid):
        raise HTTPException(400, "A valid finalized uid is required.")
    return uid, require_registration_session_id(registration_session_id)


@app.post("/store-registration-face")
async def store_registration_face(registration_session_id: str = Form(...), image: UploadFile = File(...),
                                  _: bool = Depends(require_api_key)):
    registration_session_id = require_registration_session_id(registration_session_id)
    return await handle("store", [image], registration_session_id=registration_session_id)


@app.post("/enroll-face")
async def enroll_face(uid: str = Form(...), registration_session_id: str = Form(...),
                      _: bool = Depends(require_api_key)):
    uid, registration_session_id = require_finalization_identifiers(uid, registration_session_id)
    return await handle("finalize", uid=uid, registration_session_id=registration_session_id)


@app.post("/discard-registration-face")
async def discard_registration_face(registration_session_id: str = Form(...),
                                    _: bool = Depends(require_api_key)):
    registration_session_id = require_registration_session_id(registration_session_id)
    return await handle("discard", registration_session_id=registration_session_id)


@app.post("/admin/reset-development-enrollments")
async def reset_development_enrollments(
    request: DevelopmentResetRequest,
    _: bool = Depends(require_api_key),
):
    """Reset a dedicated development dataset; never enabled implicitly."""
    if not development_face_reset_enabled():
        raise HTTPException(403, "Development face reset is disabled.")
    if not development_face_store_allowed():
        raise HTTPException(403, "Face storage is not declared as development-only.")
    if not request.dryRun and request.confirm != "RESET_BLUETAP_FACE_DEV":
        raise HTTPException(403, "Exact development reset confirmation is required.")
    return await handle(
        "development-reset",
        dry_run=request.dryRun,
        confirmation=request.confirm,
    )

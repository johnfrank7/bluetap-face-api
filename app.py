import asyncio
import logging
import os
from contextlib import asynccontextmanager

from runtime import (MODEL_NAME, DETECTOR_BACKEND, ANTI_SPOOFING,
                     INFERENCE_LOCK, model_ready, preload_model, verify_images)
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool
from image_utils import prepare_image

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
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
                         "recognizer": "OpenCV SFace", "detector": "YuNet", "modelLoaded": loaded}, status_code=200 if loaded else 503)


def process(operation, uploads, subject_id=None):
    if not model_ready():
        raise HTTPException(503, "Face model is not ready. Please retry later.")
    if not INFERENCE_LOCK.acquire(blocking=False):
        raise HTTPException(503, "Face service is busy. Please retry later.")
    try:
        for upload in uploads:
            if upload.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(400, "A supported face image is required.")
        images = [prepare_image(upload) for upload in uploads]
        from face_service import generate_embedding, search_duplicate, enroll_subject
        if operation == "verify":
            return verify_images(images[0], images[1])
        embedding = generate_embedding(images[0])
        result = search_duplicate(embedding)
        response = {"duplicateDetected": result["matched"], "distance": result["distance"],
                    "threshold": result["threshold"], "reviewRequired": result["matched"]}
        if operation == "duplicate":
            return response
        if result["matched"]:
            return {"enrolled": False, **response}
        enroll_subject(subject_id, embedding)
        return {"enrolled": True, "duplicateDetected": False, "reviewRequired": False}
    except HTTPException:
        raise
    except ValueError:
        logger.exception("Face %s rejected an image", operation)
        raise HTTPException(400, "Could not process the face image. Use a clear photo with a visible face.") from None
    except Exception:
        logger.exception("Face %s failed", operation)
        raise HTTPException(500, {"verify": "Face verification failed.",
                                  "duplicate": "Duplicate face check failed.",
                                  "enroll": "Face enrollment failed."}[operation]) from None
    finally:
        INFERENCE_LOCK.release()


async def handle(operation, uploads, subject_id=None):
    try:
        return await run_in_threadpool(process, operation, uploads, subject_id)
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


@app.post("/enroll-face")
async def enroll_face(subject_id: str = Form(...), image: UploadFile = File(...),
                      _: bool = Depends(require_api_key)):
    subject_id = subject_id.strip()
    if not subject_id:
        await image.close()
        raise HTTPException(400, "subject_id is required.")
    return await handle("enroll", [image], subject_id)

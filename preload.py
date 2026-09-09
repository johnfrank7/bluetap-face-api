"""Download official OpenCV model assets at build time, never from a request."""
import os
import shutil
import urllib.request
from runtime import MODEL_DIR, preload_model, model_ready

MODELS = {
    "face_recognition_sface_2021dec.onnx": "face_recognition_sface",
    "face_detection_yunet_2023mar.onnx": "face_detection_yunet",
}


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, folder in MODELS.items():
        destination = MODEL_DIR / filename
        if destination.exists():
            continue
        temporary = destination.with_suffix(".download")
        try:
            url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{folder}/{filename}"
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    preload_model()
    if not model_ready():
        raise SystemExit("Model build failed; verify or replace the model files before deploying")


if __name__ == "__main__":
    main()

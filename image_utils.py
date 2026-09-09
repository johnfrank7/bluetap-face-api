import io
import warnings
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MAX_DIMENSION = 800
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def prepare_image(upload):
    payload = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image must be at most 5 MiB.")
    try:
        # Pillow only parses metadata to enforce limits before native decoding.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as header:
                if header.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Unsupported image format")
                width, height = header.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "Image must be at most 12 megapixels.")
                flag = cv2.IMREAD_COLOR
                if header.format == "JPEG":
                    for factor, candidate in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2)):
                        if min(width, height) / factor >= MAX_DIMENSION:
                            flag = candidate
                            break
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flag)
        if image is None:
            raise ValueError("Image cannot be decoded")
        height, width = image.shape[:2]
        if width * height > MAX_IMAGE_PIXELS:
            raise HTTPException(413, "Image must be at most 12 megapixels.")
        scale = min(1.0, MAX_DIMENSION / max(width, height))
        if scale < 1:
            image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
        return image
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(413, "Image dimensions are too large.") from None
    except (UnidentifiedImageError, OSError, ValueError, cv2.error):
        raise HTTPException(400, "Invalid or unsupported image.") from None

import warnings
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import HTTPException

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MAX_DIMENSION = 800
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def prepare_image(upload):
    # Read in the worker thread, under the shared inference lock.
    payload = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image must be at most 5 MiB.")
    import io
    import numpy as np
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Unsupported image format")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "Image must be at most 12 megapixels.")
                image.draft("RGB", (MAX_DIMENSION, MAX_DIMENSION))
                image.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
                image = ImageOps.exif_transpose(image).convert("RGB")
                return np.array(image)[:, :, ::-1].copy()
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(413, "Image dimensions are too large.") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "Invalid or unsupported image.") from None

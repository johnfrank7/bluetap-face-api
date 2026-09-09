import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from PIL import Image
from image_utils import prepare_image


class ImageTests(unittest.TestCase):
    def test_resize_and_bgr(self):
        stream = io.BytesIO()
        Image.new("RGB", (2000, 1000), (255, 0, 0)).save(stream, format="PNG")
        stream.seek(0)
        image = prepare_image(UploadFile(stream))
        self.assertEqual(image.shape, (400, 800, 3))
        self.assertEqual(image[0, 0].tolist(), [0, 0, 255])

    def test_pixel_limit(self):
        stream = io.BytesIO()
        Image.new("1", (4000, 4000)).save(stream, format="PNG")
        stream.seek(0)
        with self.assertRaises(HTTPException) as context:
            prepare_image(UploadFile(stream))
        self.assertEqual(context.exception.status_code, 413)


class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from runtime import preload_model, model_ready
        preload_model()
        assert model_ready()

    def test_old_sface_schema_rejected(self):
        import face_service as service
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.json"
            path.write_text(json.dumps({"model": "SFace", "subjects": [{"embedding": [0.1] * 128}]}))
            with patch.object(service, "FACE_DATA_DIR", Path(directory)), patch.object(service, "FACE_DATABASE_PATH", path):
                with self.assertRaises(RuntimeError):
                    service.load_database()

    def test_legacy_database_is_preserved_and_rejected(self):
        import face_service as service
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.json"
            original = json.dumps({"subjects": [{"subject_id": "test", "embedding": [0.1] * 512}]})
            path.write_text(original)
            with patch.object(service, "FACE_DATA_DIR", Path(directory)), patch.object(service, "FACE_DATABASE_PATH", path):
                with self.assertRaises(RuntimeError):
                    service.search_duplicate([0.1] * 128)
            self.assertEqual(path.read_text(), original)

    def test_sface_enrollment_and_duplicate(self):
        import face_service as service
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(service, "FACE_DATA_DIR", Path(directory)), patch.object(service, "FACE_DATABASE_PATH", Path(directory) / "embeddings.json"):
                embedding = [0.1] * 128
                self.assertFalse(service.search_duplicate(embedding)["matched"])
                service.enroll_subject("test", embedding)
                self.assertTrue(service.search_duplicate(embedding)["matched"])
                self.assertEqual(service.load_database()["model"], "SFace")


if __name__ == "__main__":
    unittest.main()

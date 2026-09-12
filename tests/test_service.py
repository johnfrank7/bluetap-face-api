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

    def test_unclassified_sface_subject_fails_closed(self):
        import face_service as service
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.json"
            path.write_text(json.dumps({"model": "SFace", "embedding_schema": service.EMBEDDING_SCHEMA,
                                        "subjects": [{"subject_id": "legacy", "embedding": [0.1] * 128}]}))
            with patch.object(service, "FACE_DATA_DIR", Path(directory)), patch.object(service, "FACE_DATABASE_PATH", path):
                with self.assertRaises(RuntimeError):
                    service.search_duplicate([0.1] * 128)

    def test_temporary_registration_is_not_a_duplicate_until_finalized(self):
        import face_service as service
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(service, "FACE_DATA_DIR", Path(directory)), patch.object(service, "FACE_DATABASE_PATH", Path(directory) / "embeddings.json"):
                embedding = [0.1] * 128
                self.assertFalse(service.search_duplicate(embedding)["matched"])
                session = "12345678-1234-4234-8234-123456789abc"
                service.store_registration_face(session, embedding)
                self.assertFalse(service.search_duplicate(embedding)["matched"])
                self.assertTrue(service.finalize_registration_face("firebase-uid", session)["enrolled"])
                self.assertTrue(service.search_duplicate(embedding)["matched"])
                database = service.load_database()
                self.assertEqual(database["model"], "SFace")
                self.assertEqual(database["subjects"][0]["finalized"], True)
                self.assertEqual(database["temporary_registrations"], [])

    def test_expired_temporary_entry_is_purged_without_touching_finalized_faces(self):
        import face_service as service
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(service, "FACE_DATA_DIR", Path(directory)), patch.object(service, "FACE_DATABASE_PATH", Path(directory) / "embeddings.json"):
                database = {
                    "model": "SFace", "embedding_schema": service.EMBEDDING_SCHEMA,
                    "subjects": [{"subject_id": "permanent", "embedding": [0.1] * 128, "finalized": True}],
                    "temporary_registrations": [{"registration_session_id": "expired", "embedding": [0.2] * 128, "expires_at": "2000-01-01T00:00:00+00:00"}],
                }
                service.save_database(database)
                self.assertTrue(service.purge_expired_registrations(database))
                service.save_database(database)
                cleaned = service.load_database()
                self.assertEqual(cleaned["temporary_registrations"], [])
                self.assertEqual(cleaned["subjects"][0]["subject_id"], "permanent")

    def test_discard_only_removes_temporary_registration(self):
        import face_service as service
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(service, "FACE_DATA_DIR", Path(directory)), patch.object(service, "FACE_DATABASE_PATH", Path(directory) / "embeddings.json"):
                session = "12345678-1234-4234-8234-123456789abc"
                service.store_registration_face(session, [0.1] * 128)
                service.finalize_registration_face("permanent", session)
                self.assertFalse(service.discard_registration_face(session))
                self.assertEqual(service.load_database()["subjects"][0]["subject_id"], "permanent")


if __name__ == "__main__":
    unittest.main()

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

import app as application
import face_service


class DevelopmentResetEndpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.directory.name) / "embeddings.json"
        self.database_path.write_text(json.dumps({
            "model": face_service.MODEL_NAME,
            "embedding_schema": face_service.EMBEDDING_SCHEMA,
            "subjects": [
                {"subject_id": "final-user", "embedding": [0.1], "finalized": True},
                {"subject_id": "orphan", "embedding": [0.2]},
            ],
            "temporary_registrations": [
                {"registration_session_id": "temporary", "embedding": [0.3]},
            ],
        }))
        self.path_patches = [
            patch.object(face_service, "FACE_DATA_DIR", Path(self.directory.name)),
            patch.object(face_service, "FACE_DATABASE_PATH", self.database_path),
            patch.dict(os.environ, {
                "ENABLE_DEVELOPMENT_FACE_RESET": "true",
                "FACE_DATA_ENVIRONMENT": "development",
            }),
        ]
        for active_patch in self.path_patches:
            active_patch.start()

    def tearDown(self):
        for active_patch in reversed(self.path_patches):
            active_patch.stop()
        self.directory.cleanup()

    def test_authentication_and_explicit_request_are_required(self):
        with self.assertRaises(HTTPException) as missing:
            application.require_api_key(None)
        self.assertEqual(missing.exception.status_code, 401)
        with self.assertRaises(HTTPException) as wrong:
            application.require_api_key(HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="wrong"
            ))
        self.assertEqual(wrong.exception.status_code, 401)
        self.assertTrue(application.require_api_key(HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=application.DEEPFACE_API_KEY
        )))
        with self.assertRaises(ValidationError):
            application.DevelopmentResetRequest()

    def test_openapi_registers_exact_unprefixed_post_route(self):
        schema = application.app.openapi()
        route = schema["paths"].get("/admin/reset-development-enrollments")
        self.assertIsNotNone(route)
        self.assertIn("post", route)
        self.assertNotIn("/api/admin/reset-development-enrollments", schema["paths"])

    async def test_apply_requires_exact_confirmation(self):
        for confirmation in (None, "WRONG"):
            request = application.DevelopmentResetRequest(
                dryRun=False, confirm=confirmation
            )
            with self.assertRaises(HTTPException) as rejected:
                await application.reset_development_enrollments(request, True)
            self.assertEqual(rejected.exception.status_code, 403)

    async def test_dry_run_reports_counts_without_face_data_or_mutation(self):
        before = self.database_path.read_text()
        result = await application.reset_development_enrollments(
            application.DevelopmentResetRequest(dryRun=True), True
        )
        self.assertEqual(result["finalizedEnrollments"], 1)
        self.assertEqual(result["orphanedEnrollments"], 1)
        self.assertEqual(result["temporaryRegistrations"], 1)
        self.assertEqual(result["wouldDelete"], 3)
        self.assertNotIn("subjects", result)
        self.assertNotIn("embedding", json.dumps(result))
        self.assertEqual(self.database_path.read_text(), before)

    async def test_apply_preserves_schema_and_is_idempotent(self):
        request = application.DevelopmentResetRequest(
            dryRun=False, confirm="RESET_BLUETAP_FACE_DEV"
        )
        result = await application.reset_development_enrollments(request, True)
        self.assertEqual(result["deletedEntries"], 3)
        database = json.loads(self.database_path.read_text())
        self.assertEqual(database["model"], face_service.MODEL_NAME)
        self.assertEqual(database["embedding_schema"], face_service.EMBEDDING_SCHEMA)
        self.assertEqual(database["subjects"], [])
        self.assertEqual(database["temporary_registrations"], [])

        retry = await application.reset_development_enrollments(request, True)
        self.assertEqual(retry["deletedEntries"], 0)

    async def test_reset_is_disabled_unless_store_is_explicitly_development(self):
        with patch.dict(os.environ, {"FACE_DATA_ENVIRONMENT": "production"}):
            with self.assertRaises(HTTPException) as rejected:
                await application.reset_development_enrollments(
                    application.DevelopmentResetRequest(dryRun=True), True
                )
        self.assertEqual(rejected.exception.status_code, 403)

    async def test_health_and_ready_behavior_are_unchanged(self):
        result = await application.health()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["model"], face_service.MODEL_NAME)
        with patch.object(application, "model_ready", return_value=True):
            response = await application.ready()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)["status"], "ready")

    async def test_normal_verification_dispatch_still_works_after_reset(self):
        request = application.DevelopmentResetRequest(
            dryRun=False, confirm="RESET_BLUETAP_FACE_DEV"
        )
        await application.reset_development_enrollments(request, True)
        uploads = [type("Upload", (), {"content_type": "image/png"})() for _ in range(2)]
        expected = {"verified": True, "distance": 0.1}
        with patch.object(application, "model_ready", return_value=True), \
                patch.object(application, "prepare_image", return_value=object()), \
                patch.object(application, "verify_images", return_value=expected):
            self.assertEqual(application.process("verify", uploads), expected)


if __name__ == "__main__":
    unittest.main()

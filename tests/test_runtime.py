import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import runtime


class RuntimeTests(unittest.TestCase):
    def test_no_heavy_imports(self):
        import app
        import face_service
        runtime.preload_model()
        self.assertTrue(runtime.model_ready())
        self.assertFalse(any(name.split('.')[0] in {'deepface', 'tensorflow', 'keras', 'tf_keras'} for name in sys.modules))

    def test_cached_models(self):
        runtime.preload_model()
        before = runtime.recognizer, runtime.detector
        runtime.preload_model()
        self.assertIs(before[0], runtime.recognizer)
        self.assertIs(before[1], runtime.detector)

    def test_preload_failure_stays_unready(self):
        with TemporaryDirectory() as directory:
            with patch.object(runtime, 'recognizer', None), patch.object(runtime, 'detector', None), patch.object(runtime, 'MODEL_DIR', Path(directory)):
                with self.assertLogs('uvicorn.error', level='ERROR'):
                    runtime.preload_model()
                self.assertFalse(runtime.model_ready())

    def test_threshold_semantics(self):
        runtime.preload_model()
        self.assertAlmostEqual(runtime.DISTANCE_THRESHOLD, 1 - runtime.FACE_MATCH_THRESHOLD)
        self.assertAlmostEqual(runtime.cosine_distance([1.0] * 128, [1.0] * 128), 0, places=6)
        self.assertAlmostEqual(runtime.cosine_distance([1.0] * 128, [-1.0] * 128), 2, places=6)

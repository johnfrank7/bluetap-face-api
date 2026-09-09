import io
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from PIL import Image

BASE = "http://127.0.0.1:8011"


def post(parts, key="bluetap-local-dev-key"):
    boundary = "bluetap-smoke-boundary"
    body = b""
    for name, payload in parts:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"face.jpg\"\r\n"
                 "Content-Type: image/jpeg\r\n\r\n").encode() + payload + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    request = urllib.request.Request(BASE + "/verify-face", data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}", "Authorization": f"Bearer {key}"})
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


class SmokeTests(unittest.TestCase):
    def test_health(self):
        self.assertEqual(json.load(urllib.request.urlopen(BASE + "/health")), {"status": "ok", "model": "SFace"})

    def test_ready(self):
        self.assertTrue(json.load(urllib.request.urlopen(BASE + "/ready"))["modelLoaded"])

    def test_real_face(self):
        face = Path("tests/face.jpg").read_bytes()
        status, result = post([("image1", face), ("image2", face)])
        self.assertEqual(status, 200, result)
        self.assertTrue(result["verified"])
        self.assertEqual(result["model"], "SFace")
        self.assertEqual(result["detector_backend"], "opencv")
        self.assertEqual(set(result), {"verified", "distance", "threshold", "model", "detector_backend", "similarity_metric"})

    def test_bad_image(self):
        status, result = post([("image1", b"invalid"), ("image2", b"invalid")])
        self.assertEqual(status, 400)
        self.assertNotIn("Traceback", json.dumps(result))

    def test_no_face(self):
        image = io.BytesIO()
        Image.new("RGB", (800, 800)).save(image, format="JPEG")
        status, result = post([("image1", image.getvalue()), ("image2", image.getvalue())])
        self.assertEqual(status, 400)
        self.assertNotIn("Traceback", json.dumps(result))

    def test_upload_limit(self):
        self.assertEqual(post([("image1", b"x" * (5 * 1024 * 1024 + 1)), ("image2", b"x")])[0], 413)

    def test_body_limit(self):
        self.assertEqual(post([("image1", b"x" * (12 * 1024 * 1024)), ("image2", b"x")])[0], 413)

    def test_auth(self):
        self.assertEqual(post([("image1", b"x"), ("image2", b"x")], key="invalid")[0], 401)


if __name__ == "__main__":
    unittest.main()

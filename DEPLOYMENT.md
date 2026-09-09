# Native OpenCV face service on Render Free

Production recognition uses OpenCV FaceRecognizerSF (SFace ONNX) and
FaceDetectorYN (YuNet ONNX), with the OpenCV DNN CPU backend. DeepFace,
TensorFlow, tf-keras, and their transitive ML dependencies are absent from the
production requirements and have no active imports. The existing .venv may still
contain old packages; use a fresh environment rather than overlaying installs.

Dependencies: FastAPI, Uvicorn, opencv-python-headless 4.11.0.86, numpy,
python-multipart, Pillow. Pillow is retained only for image-header inspection
before OpenCV decoding and for test fixture generation. Headless OpenCV was
verified to support both native face APIs in a clean Windows environment.

## Models and initialization

Run `python preload.py` at build time. It streams the official OpenCV Zoo models
into the project `models/` directory (ignored by Git):

- face_recognition_sface_2021dec.onnx (about 38.7 MB)
- face_detection_yunet_2023mar.onnx (about 232 KB)

Sources:
https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

Existing files are reused and validated by model loading. Failed downloads leave
no partial destination file. If an existing file is corrupt, replace that file
and rerun preload.py. Runtime startup never downloads anything. It creates and
warms both models once, then publishes them together as cached module objects.
Requests reuse them under a lock. /ready returns 503 until both have loaded and
warmed successfully; initialization failures log tracebacks and keep /health
available. Restart after correcting model files/configuration.

FACE_MODEL_DIR optionally overrides the model directory; its location must be
identical at build and runtime and included in the deployment artifact.
DEEPFACE_HOME is no longer used.

## Matching and compatibility

FACE_MATCH_THRESHOLD is a cosine SIMILARITY threshold: higher means stricter.
The development default is 0.363, drawn from OpenCV's published SFace LFW
example. It is not validated for BlueTap's population or capture conditions.
Tune false acceptance/rejection rates using representative authorized data.
Source: https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html

The public API keeps cosine DISTANCE semantics:

    distance = 1 - cosine_similarity
    threshold = 1 - FACE_MATCH_THRESHOLD  # default 0.637
    verified = distance <= threshold

Pair verification and duplicate search use the same OpenCV cosine matcher and
threshold. Exactly one face detected with score >=0.9 is required. Detected faces
smaller than 40 px in the resized source image are rejected. YuNet receives a
letterboxed 320x320 image; landmarks are mapped back to the source image for
SFace alignCrop and 128-dimensional embedding extraction. This bounds detector
allocations independently of uploaded image resolution.

URLs, bearer authentication, DEEPFACE_API_KEY (including its existing local
default), multipart image1/image2 fields, subject_id, and verification/duplicate/
enrollment response keys remain unchanged. model remains SFace; detector_backend
now accurately reports yunet. similarity_metric remains cosine. No frontend,
Firebase, Vercel, or DEEPFACE_API_URL changes. registrationSessionId was not a
service field and no new requirement for it was introduced.

/health remains {"status":"ok","model":"SFace"}, without face inference.
/ready retains status/model/modelLoaded and adds recognizer="OpenCV SFace" and
detector="YuNet".

## Enrollment migration and limitations

Back up face-data/embeddings.json before migration. Old Facenet512 embeddings
cannot be converted. Previous DeepFace SFace embeddings also require re-enrollment
because the detector, alignment, and preprocessing differ. Nonempty databases
must now contain embedding_schema="opencv-sface-yunet-alignCrop-v1" and model=SFace;
legacy databases fail closed without modifying records. Do not relabel old data.
Re-enroll from authorized original images into a new native database.

The existing local JSON database remains ephemeral on Render Free. Redeploys and
restarts can lose enrollments; durable duplicate detection still requires external
persistence. This task does not introduce Firebase or other external storage.
Render documentation: https://render.com/docs/free

Liveness is NOT implemented. ANTI_SPOOFING remains False. SFace matches identity;
it does not detect a printed face or screen replay. A separate liveness layer
is still required; no heavy anti-spoofing or demographic models were added.

## Memory and image controls

One worker, one admitted POST at a time; concurrent POSTs return 503. OpenCV uses
one CPU thread. Uploads are limited to 5 MiB each / 11 MiB total multipart body,
including chunked transfer. Headers are checked before decode; images above
12 million pixels are rejected. OpenCV decodes BGR, uses reduced JPEG decoding
where possible, respects EXIF orientation, and resizes to at most 800 px.
Temporary framework upload files are closed/deleted in finally; the engine uses
in-memory arrays. Missing/multiple faces and malformed images return 400; upload
limits return 413, unavailable/busy models 503, unexpected failures 500. Internal
tracebacks are logged server-side and never returned to callers. Application
logic does not generate 502 responses.

Measured in a fresh native-only Windows environment: initial HTTP suite peak
169 MiB; after adding two 12-megapixel PNG inputs peak 230 MiB. Earlier DeepFace
single-verification peak was about 544 MiB. The new large-image peak is roughly
58% lower, though these are different workloads, not a controlled benchmark.
This leaves substantially more headroom for Render Free's 512 MB budget. Linux
container accounting differs: confirm Render memory on startup and real requests.
No Render deployment or Linux cgroup memory measurement was performed here.

## Render deployment

1. Back up existing enrollment records and arrange native SFace re-enrollment.
2. Push the service changes to Render's connected branch. Do not commit models,
   face-data, test photos, or local virtual environments.
3. Set Build Command:

       pip install -r requirements.txt && python preload.py

4. Set Start Command:

       python -m uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1

5. Keep DEEPFACE_API_KEY unchanged. Set FACE_MATCH_THRESHOLD=0.363 initially
   (development default, tune before production decisions). Leave FACE_MODEL_DIR
   unset for the project models directory. Health Check Path: /health.
6. Use Manual Deploy > Clear build cache & deploy. YES, clear the old build cache
   so the previous TensorFlow/DeepFace environment is not reused.
7. Confirm build logs download/load both ONNX models. Runtime logs should say
   OpenCV SFace + YuNet and should contain no TensorFlow/CUDA initialization.
8. Wait for GET /ready to return 200 and modelLoaded=true, then test matching and
   nonmatching multipart POST /verify-face using the existing bearer key. Inspect
   memory metrics and check duplicate/enrollment flows after migration.

## Local validation

Create a fresh environment and install requirements.txt. Run python preload.py,
then start Uvicorn on 127.0.0.1:8011 using the start command without $PORT.

    python tests/smoke.py
    python -m unittest discover -s tests -p "test_*.py"
    python -m pip check

11 HTTP tests passed: health, ready, same person in two real photos, different
people, multiple faces, no face, malformed image, per-file size, whole-body size,
12-megapixel input, and authentication. Nine unit tests passed: resize/BGR, pixel
limit, legacy Facenet database, legacy DeepFace SFace schema, new enrollment and
duplicate matching, missing-model startup failure, model cache reuse, threshold
semantics, and absence of heavy imports. pip check passed in .venv-native.

Smoke test photos are local ignored fixtures from:
https://github.com/ageitgey/face_recognition/tree/master/examples
Download obama.jpg as tests/face-a.jpg, obama2.jpg as tests/face-a2.jpg, and
biden.jpg as tests/face-b.jpg before running the HTTP suite. They are test-only;
no user face database was used or modified by tests.

Remaining deepface/tensorflow/keras text references are this migration document,
the compatibility environment variable, regression assertions against heavy
imports, and the ignored legacy .deepface directory/local environment. None is
an active production framework import.

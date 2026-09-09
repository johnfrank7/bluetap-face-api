# Render deployment

This service uses DeepFace 0.0.100 SFace recognition and the OpenCV detector.
Installed OpenCV 4.11.0 supports FaceRecognizerSF; contrib is not required in the
verified local environment. No age/emotion/race/gender or anti-spoofing models are
built. Anti-spoofing was already disabled and remains disabled: image matching
alone does not establish liveness.

## Resource behavior and limits

- One Uvicorn worker, one admitted POST at a time; excess requests receive 503.
- CPU-only recognition, one TensorFlow/OpenCV/BLAS computation thread.
- 5 MiB per image; 11 MiB multipart request limit, including chunked bodies.
- At most 12 million decoded pixels; JPEG draft decoding when available;
  aspect-preserving resize to 800 px before inference, with EXIF orientation.
- Framework upload files close in finally; inference receives BGR arrays and
  creates no extra upload files.
- Recognition and detector are preloaded in the background at startup. /health
  does not import DeepFace or run inference. /ready returns 503 until loading
  succeeds. Preload failures log full tracebacks, leave health available, and
  keep inference unavailable until restart. Requests never retry downloads.
- Build-time preload bundles weights under the project .deepface directory.
  Do not change DEEPFACE_HOME between build and runtime.

SFace weights are 38.7 MB versus the previous 95.0 MB Facenet512 weights, about
59% smaller. This is NOT a measured RAM reduction. TensorFlow is still imported
by DeepFace 0.0.100, even for OpenCV-based SFace. Removing tensorflow/tf-keras
from requirements breaks that version; CPU-only settings do not remove its
base memory footprint. Local Windows measurements after real verification:
about 544 MiB peak working set; full HTTP stress/validation smoke tests peaked
at about 631 MiB. These are Windows process measurements, not Linux container
memory measurements. This implementation is not certified to fit Render Free's
512 MB budget. Inspect Render memory metrics after deployment; a reliable lower
footprint may require replacing the DeepFace runtime with a direct OpenCV SFace
implementation and validating its preprocessing/threshold equivalence. A native
rewrite has deliberately not been substituted for the requested DeepFace API.
CUDA warnings alone are not fatal. Python cannot catch an OS out-of-memory kill.

## Existing face records

Facenet512 vectors have 512 dimensions; SFace vectors have 128. Back up existing
face-data/embeddings.json before deploying. Re-enroll subjects from authorized
original images into a fresh SFace database. Do not relabel old vectors as SFace.
Nonempty legacy databases are rejected without modifying them; duplicate checks
fail closed rather than reporting no matches. New databases record model=SFace.
Duplicate cosine threshold is now DeepFace's SFace default 0.593 instead of the
old experimental 0.30. Validate false matches/rejections on representative data
before relying on it for registration decisions.

The existing JSON database remains on local disk. Render Free's filesystem is
ephemeral, so enrollment records are lost on redeploy/restart. Durable duplicate
protection requires external persistence; this change does not alter Firebase
or the frontend. https://render.com/docs/free

## Exact redeploy steps

1. Back up existing enrollment records and arrange SFace re-enrollment as above.
2. Commit/push the changed Python files, requirements.txt, .gitignore, and this
   document to the branch connected to the existing Render service. Do not commit
   .deepface weights or face-data.
3. Render dashboard > existing web service > Settings: set Build Command to:

       pip install -r requirements.txt && python preload.py

4. Set Start Command to:

       python -m uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1

   Do not use --reload or multiple workers. Keep .python-version at 3.11.9.
5. Keep the existing DEEPFACE_API_KEY value unchanged. Leave DEEPFACE_API_URL in
   the calling integration unchanged. Remove any previous DEEPFACE_HOME override
   (or ensure it points to the project directory at both build and runtime).
   Set PYTHONIOENCODING=utf-8. CPU/thread defaults are configured in runtime.py.
6. Set Health Check Path to /health. Choose Manual Deploy > Clear build cache &
   deploy. The build must successfully fetch/build SFace before it can succeed.
7. Check GET /health returns {"status":"ok","model":"SFace"}; wait for GET
   /ready to return 200 and modelLoaded=true. If loading fails, inspect server
   tracebacks and restart after correcting the cause.
8. Test an authenticated multipart POST /verify-face with image1 and image2:

       curl -H "Authorization: Bearer $DEEPFACE_API_KEY" -F image1=@face1.jpg -F image2=@face2.jpg https://YOUR-SERVICE.onrender.com/verify-face

9. Inspect Render memory during startup and several real verifications. If the
   container exceeds 512 MB, this runtime still cannot serve reliably on Free;
   increasing HTTP timeouts cannot solve that. Test duplicate enrollment only
   after migration and resolution of the ephemeral database limitation.

Verification result keys and duplicate/enrollment response keys are preserved.
Bearer authentication is unchanged. registrationSessionId did not appear in the
inspected service; no new requirement or renaming was introduced for that field.
No frontend, Firebase, Vercel, or API URL changes were made.

## Local checks

Start the service on port 8011, wait for /ready, then run:

    python tests/smoke.py
    python -m unittest discover -s tests -p "test_*.py"

The smoke fixture tests/face.jpg is downloaded locally from OpenCV's public test
image https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg
and is ignored by git; supply that fixture before running the HTTP smoke suite.
The smoke suite checks real matching, health/readiness, authentication, malformed
images, no-face rejection, per-file limits, and multipart limits. Unit checks
cover resizing/BGR conversion, pixel limits, and legacy/current database behavior.

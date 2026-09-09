"""Run at Render build time so weights ship with the deployment artifact."""
from runtime import preload_model, model_ready

preload_model()
if not model_ready():
    raise SystemExit("Model build failed; deployment must not proceed without weights")

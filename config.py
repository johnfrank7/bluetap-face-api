import os


TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def env_flag(name, default=False):
    """Read an opt-in environment flag; missing/unknown values stay disabled."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY_ENV_VALUES


def development_face_reset_enabled():
    return env_flag("ENABLE_DEVELOPMENT_FACE_RESET", default=False)


def face_data_environment():
    return os.getenv("FACE_DATA_ENVIRONMENT", "").strip().lower()


def development_face_store_allowed():
    return face_data_environment() == "development"

import os

def get_env(key: str, default=None):
    """Helper to safely fetch env variables"""
    return os.getenv(key, default)


# ---------------- Google API / Gemini ---------------- #

def get_gemini_api_key() -> str | None:
    """
    Used ONLY for local development.
    Cloud Run uses ADC instead of API keys.
    """
    return get_env("GEMINI_API_KEY")


# ---------------- Service Account Config ---------------- #

def get_service_account_file() -> str | None:
    """
    Local dev: path to service account json
    Cloud Run: NOT used (ADC instead)
    """
    return get_env("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")


# ---------------- Weaviate ---------------- #

def get_weaviate_collection() -> str:
    return get_env("WEAVIATE_COLLECTION", "LegalChunk")


def get_weaviate_settings() -> dict:
    """Weaviate connection configuration extracted from environment."""
    return {
        "url": get_env("WEAVIATE_URL"),
        "api_key": get_env("WEAVIATE_API_KEY"),
        "collection": get_weaviate_collection(),
        "grpc_port": int(get_env("WEAVIATE_GRPC_PORT", "50051")),
        "headers": {
            "X-Vertex-Project-Id": get_env("GOOGLE_CLOUD_PROJECT"),
            "X-Vertex-Location": get_env("GOOGLE_VERTEX_LOCATION", "asia-south1"),
        },
    }


# ---------------- Ingestion State Backend ---------------- #

def get_state_backend() -> str:
    return get_env("INGESTION_STATE_BACKEND", "file").lower()


def get_gcs_state_config() -> dict:
    return {
        "bucket": get_env("INGESTION_STATE_GCS_BUCKET"),
        "blob": get_env("INGESTION_STATE_GCS_BLOB", "ingestion/ingestion_state.json"),
    }


# ---------------- Drive Configuration ---------------- #

def get_drive_root_folder_id() -> str | None:
    return get_env("DRIVE_ROOT_FOLDER_ID")


def get_drive_allowed_exts() -> list[str]:
    raw = get_env("DRIVE_ALLOWED_EXTS", "pdf")
    return [ext.strip().lower() for ext in raw.split(",") if ext.strip()]

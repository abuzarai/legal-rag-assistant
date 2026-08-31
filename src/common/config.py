import os


# ---------------- Google API / Gemini ---------------- #

def get_gemini_api_key() -> str | None:
    """
    Gemini API (AI Studio) key. Preferred auth mode: free tier, no billing.
    When unset, falls back to Vertex AI via ADC.
    """
    return os.environ.get("GEMINI_API_KEY")


def get_embedding_model() -> str:
    return os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")


def get_embedding_output_dims() -> int | None:
    """Output dimensionality for gemini-embedding-* (default 768).
    MUST match between ingestion and query time."""
    raw = os.environ.get("EMBEDDING_OUTPUT_DIMS")
    return int(raw) if raw else 768


# ---------------- Service Account Config ---------------- #

def get_service_account_file() -> str | None:
    """
    Local dev: path to service account json
    Cloud Run: NOT used (ADC instead)
    """
    return os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")


# ---------------- Weaviate ---------------- #

def get_weaviate_collection() -> str:
    return os.environ.get("WEAVIATE_COLLECTION", "LegalChunk")


def get_weaviate_settings() -> dict:
    """Weaviate connection configuration extracted from environment."""
    return {
        "url": os.environ.get("WEAVIATE_URL"),
        "api_key": os.environ.get("WEAVIATE_API_KEY"),
        "collection": get_weaviate_collection(),
        "grpc_port": int(os.environ.get("WEAVIATE_GRPC_PORT", "50051")),
        "headers": {
            "X-Vertex-Project-Id": os.environ.get("GOOGLE_CLOUD_PROJECT"),
            "X-Vertex-Location": os.environ.get("GOOGLE_VERTEX_LOCATION", "asia-south1"),
        },
    }


# ---------------- Ingestion State Backend ---------------- #

def get_state_backend() -> str:
    return os.environ.get("INGESTION_STATE_BACKEND", "file").lower()


def get_gcs_state_config() -> dict:
    return {
        "bucket": os.environ.get("INGESTION_STATE_GCS_BUCKET"),
        "blob": os.environ.get("INGESTION_STATE_GCS_BLOB", "ingestion/ingestion_state.json"),
    }


# ---------------- Drive Configuration ---------------- #

def get_drive_root_folder_id() -> str | None:
    return os.environ.get("DRIVE_ROOT_FOLDER_ID")


def get_drive_allowed_exts() -> list[str]:
    raw = os.environ.get("DRIVE_ALLOWED_EXTS", "pdf")
    return [ext.strip().lower() for ext in raw.split(",") if ext.strip()]

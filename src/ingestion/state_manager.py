import os
import json
import hashlib
from datetime import datetime
from src.common.logger import get_logger
from src.common.config import get_state_backend, get_gcs_state_config

logger = get_logger(__name__)

STATE_FILE = "./artifacts/ingestion_state.json"


def _load_state_file() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state_file(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _gcs_client_blob():
    cfg = get_gcs_state_config()
    bucket = cfg.get("bucket")
    blob_path = cfg.get("blob")
    if not bucket or not blob_path:
        raise RuntimeError("GCS state backend requires INGESTION_STATE_GCS_BUCKET and INGESTION_STATE_GCS_BLOB")
    try:
        from google.cloud import storage  # type: ignore
    except Exception as e:
        raise RuntimeError("google-cloud-storage is required for GCS state backend") from e
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(blob_path)
    return blob

def load_state() -> dict:
    backend = get_state_backend()
    if backend == "gcs":
        try:
            blob = _gcs_client_blob()
            if not blob.exists():
                return {}
            data = blob.download_as_bytes()
            if not data:
                return {}
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            logger.error(f"[STATE] Failed to load state from GCS, falling back to file: {e}")
            return _load_state_file()
    # default: file
    return _load_state_file()

def save_state(state: dict) -> None:
    backend = get_state_backend()
    if backend == "gcs":
        try:
            blob = _gcs_client_blob()
            blob.upload_from_string(json.dumps(state, ensure_ascii=False, indent=2), content_type="application/json")
            return
        except Exception as e:
            logger.error(f"[STATE] Failed to save state to GCS, writing to file as fallback: {e}")
            _save_state_file(state)
            return
    _save_state_file(state)

def compute_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def needs_processing(state, filepath):
    """Check if file needs re-processing based on hash."""
    new_hash = compute_file_hash(filepath)
    old_entry = state.get(filepath)

    if not old_entry:
        logger.info(f"[STATE] {filepath} is new → will process.")
        return True, new_hash

    old_hash = old_entry.get("hash")
    if old_hash != new_hash:
        logger.info(f"[STATE] {filepath} hash changed → will reprocess.")
        return True, new_hash

    logger.info(f"[STATE] {filepath} unchanged → skipping.")
    return False, new_hash

def update_state(state, filepath, file_id, new_hash, embedding_model=None, embedding_done=False):
    state[filepath] = {
        "file_id": file_id,
        "hash": new_hash,
        "last_processed": datetime.utcnow().isoformat(),
        "embeddings": {
            "model": embedding_model or "pending",
            "done": embedding_done,
            "last_embedded": None if not embedding_done else datetime.utcnow().isoformat()
        }
    }
    return state

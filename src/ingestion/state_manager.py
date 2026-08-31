import hashlib
import json
import os
from datetime import datetime, timezone

from src.common.config import get_gcs_state_config, get_state_backend
from src.common.logger import get_logger

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


def _lookup_entry(state: dict, file_id: str, local_path: str):
    """State is keyed by Drive file id (stable across renames/paths).

    Legacy state files keyed by local path (sometimes Windows-style with
    backslashes, or old folder names) never match the current layout, which
    forced a full re-ingest on every run. Migrate those entries here: adopt
    the entry under the file id and drop the stale path key.
    """
    if file_id in state:
        return state[file_id]
    if isinstance(state.get(local_path), dict):
        entry = state.pop(local_path, None)
        state[file_id] = entry
        return entry
    # fall back to scanning by stored file_id (path keys may differ entirely)
    for key, entry in list(state.items()):
        if isinstance(entry, dict) and entry.get("file_id") == file_id:
            state.pop(key, None)
            state[file_id] = entry
            return entry
    return None


def decide_processing(state: dict, file_id: str, drive_md5, local_path: str) -> str:
    """Decide what a file needs without downloading anything first.

    Returns 'skip' (unchanged + already embedded), 'download' (new or changed
    remote content) or 'embed' (content already local, embeddings incomplete).
    """
    entry = _lookup_entry(state, file_id, local_path)
    if entry is None:
        return "download"

    entry_md5 = entry.get("drive_md5")
    if entry_md5 and drive_md5:
        if entry_md5 != drive_md5:
            return "download"  # remote content changed
        if entry.get("embeddings", {}).get("done"):
            return "skip"
        return "embed"  # interrupted run; reuse the local file

    # No drive-md5 baseline (legacy entry): fetch to compare content.
    return "download"


def update_state(state: dict, file_id: str, local_path: str, new_hash: str,
                 drive_md5=None, embedding_model=None, embedding_done=False) -> dict:
    state[file_id] = {
        "file_id": file_id,
        "local_path": local_path,
        "hash": new_hash,
        "drive_md5": drive_md5,
        "last_processed": datetime.now(timezone.utc).isoformat(),
        "embeddings": {
            "model": embedding_model or "pending",
            "done": embedding_done,
            "last_embedded": None if not embedding_done else datetime.now(timezone.utc).isoformat(),
        },
    }
    return state
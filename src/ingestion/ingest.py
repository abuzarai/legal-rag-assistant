import os
import threading
from src.ingestion.drive_fetcher import (
    get_drive_service,
    list_files_recursive,
    download_file,
)
from src.ingestion.text_extractor import extract_text_from_pdf, extract_text_from_txt
from src.ingestion.to_json import save_as_json
from src.ingestion.state_manager import (
    load_state,
    save_state,
    decide_processing,
    update_state,
    compute_file_hash,
)
from src.ingestion.embedder import upsert_chunks
from src.common.weaviate_client import get_weaviate_client
from src.common.config import (
    get_drive_allowed_exts,
    get_drive_root_folder_id,
    get_embedding_model,
)
from src.common.logger import get_logger

logger = get_logger(__name__)

# Every ingestion trigger (HTTP /ingest, drive-events, scheduler) funnels
# through run_ingestion, which holds a single non-blocking process-wide lock —
# concurrent runs would otherwise race on state and double-embed.
_ingest_lock = threading.Lock()


class IngestBusyError(Exception):
    """Raised when an ingestion is already running in this process."""


# -------------------------------------------------------------------
# Path helpers
# -------------------------------------------------------------------
def _build_paths(relative_path: str | None, fallback_name: str, category: str):
    cleaned = (relative_path or "").strip("/")
    fallback = fallback_name or "document.pdf"

    if not cleaned:
        cleaned = f"{category}/{fallback}" if category else fallback

    parts = [p for p in cleaned.split("/") if p]
    filename = parts[-1]
    relative_dir = "/".join(parts[:-1])
    safe_path = "/".join(parts)
    return relative_dir, filename, safe_path


def _local_dir(base: str, relative_dir: str) -> str:
    parts = [chunk for chunk in relative_dir.split("/") if chunk]
    return os.path.join(base, *parts) if parts else base


def _normalize_category(raw: str | None) -> str:
    val = (raw or "unclassified").strip().lower()
    aliases = {
        "cpc-sections": "cpc-sections",
        "cpc": "cpc-sections",
        "case-laws": "case-laws",
        "caselaws": "case-laws",
        "case law": "case-laws",
        "case laws": "case-laws",
        "cpc section": "cpc-sections",
        "cpc sections": "cpc-sections",
    }
    return aliases.get(val, val)


# -------------------------------------------------------------------
# Main Ingestion Pipeline
# -------------------------------------------------------------------
def run_ingestion(root_folder_id: str | None = None):
    acquired = _ingest_lock.acquire(blocking=False)
    if not acquired:
        raise IngestBusyError("ingestion already running")
    try:
        _run_ingestion_locked(root_folder_id)
    finally:
        _ingest_lock.release()


def _run_ingestion_locked(root_folder_id: str | None):
    root_id = root_folder_id or get_drive_root_folder_id()
    if not root_id:
        raise RuntimeError("DRIVE_ROOT_FOLDER_ID must be set.")

    logger.info(f"[INGEST] Starting ingestion for root folder: {root_id}")

    service = get_drive_service()
    state = load_state()
    allowed_exts = get_drive_allowed_exts()

    # Scan Drive
    files = list_files_recursive(service, root_id, allowed_exts)
    logger.info(f"[INGEST] {len(files)} eligible files found.")

    for f in files:
        relative_path = f.get("relative_path") or f.get("name")
        category = _normalize_category(f.get("category"))
        rel_dir, filename, safe_relative_path = _build_paths(
            relative_path, f.get("name"), category
        )
        file_id = str(f["id"])
        drive_md5 = f.get("md5_checksum")

        raw_dir = _local_dir("./data/raw_pdfs", rel_dir)
        os.makedirs(raw_dir, exist_ok=True)
        local_path = os.path.join(raw_dir, filename)

        # Decide before any download: unchanged + embedded files are skipped
        # using the Drive md5 from the scan (no bandwidth/quota spent).
        decision = decide_processing(state, file_id, drive_md5, local_path)
        if decision == "skip":
            logger.info(f"[SKIP] {filename} — unchanged, embeddings exist")
            continue

        if decision == "download":
            local_path = download_file(service, file_id, filename, raw_dir)

        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            extracted = extract_text_from_pdf(local_path)
        elif ext == ".txt":
            extracted = extract_text_from_txt(local_path)
        else:
            logger.warning(f"[SKIP] Unsupported extension for {filename}")
            continue

        # Save extracted JSON
        json_dir = _local_dir("./data/processed_json", rel_dir)
        os.makedirs(json_dir, exist_ok=True)
        out_json = os.path.join(json_dir, f"{filename}.json")
        save_as_json(extracted, out_json)

        # Update state (embedding not done yet)
        file_hash = compute_file_hash(local_path)
        state = update_state(
            state,
            file_id,
            local_path,
            file_hash,
            drive_md5=drive_md5,
            embedding_model=get_embedding_model(),
            embedding_done=False,
        )

        # Build document objects for embedding
        from langchain_core.documents import Document
        docs = []
        for page in extracted:
            metadata = dict(page.get("metadata", {}))
            metadata["source"] = safe_relative_path
            metadata["category"] = category
            docs.append(Document(page_content=page["page_content"], metadata=metadata))

        # Upsert embeddings
        state = upsert_chunks(docs, state, local_path, file_id=file_id, state_key=file_id)
        logger.info(f"[OK] {filename} → {out_json}")

        # Persist state after EVERY file, not just at the end, so a
        # killed/restarted run resumes from here instead of re-ingesting
        # everything from scratch.
        save_state(state)

    save_state(state)
    logger.info("[INGEST] Complete.")

    return {"status": "complete", "files": len(files)}


if __name__ == "__main__":
    from src.ingestion.embedder import QuotaExhaustedError

    try:
        run_ingestion()
        print("\n[DONE] Ingestion complete — all eligible files are in the vector store.")
    except QuotaExhaustedError as qe:
        # Graceful: state was saved per-file, so re-running resumes.
        print("\n[QUOTA] " + str(qe))
        print("[QUOTA] Nothing was lost — already-stored chunks are kept, and re-running this script resumes from where it stopped.")
        raise SystemExit(3)
    finally:
        try:
            get_weaviate_client().close()
        except Exception:
            pass
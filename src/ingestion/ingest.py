import os
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
    needs_processing,
    update_state,
)
from src.ingestion.embedder import upsert_chunks
from src.common.weaviate_client import get_weaviate_client
from src.common.config import get_drive_allowed_exts, get_drive_root_folder_id
from src.common.logger import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Ensure state entries always have the required structure
# -------------------------------------------------------------------
def ensure_state_entry(state, local_path):
    if local_path not in state:
        state[local_path] = {}

    state[local_path].setdefault("downloaded", False)
    state[local_path].setdefault("extracted", False)
    state[local_path].setdefault("embeddings", {"done": False})

    return state


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
    }
    return aliases.get(val, val)


# -------------------------------------------------------------------
# Main Ingestion Pipeline
# -------------------------------------------------------------------
def run_ingestion(root_folder_id: str | None = None):
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

        # Download PDF/TXT
        raw_dir = _local_dir("./data/raw_pdfs", rel_dir)
        os.makedirs(raw_dir, exist_ok=True)
        local_path = download_file(service, f["id"], filename, raw_dir)

        # State handling
        state = ensure_state_entry(state, local_path)
        process, file_hash = needs_processing(state, local_path)
        if not process and state[local_path]["embeddings"]["done"]:
            logger.info(f"[SKIP] {filename} — unchanged, embeddings exist")
            continue

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

        # Update state
        state = update_state(
            state,
            local_path,
            f["id"],
            file_hash,
            embedding_model="gemini-embedding-001",
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
        state = upsert_chunks(docs, state, local_path, file_id=f["id"])
        logger.info(f"[OK] {filename} → {out_json}")

    save_state(state)
    logger.info("[INGEST] Complete.")

    return {"status": "complete", "files": len(files)}


if __name__ == "__main__":
    run_ingestion()
    try:
        get_weaviate_client().close()
    except Exception:
        pass

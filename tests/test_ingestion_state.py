"""Ingestion state logic tests: per-file skip/download/embed decisions and
legacy-path-key migration (regression risk from the incremental-ingest work)."""

from src.ingestion.state_manager import decide_processing, update_state


def _embedded_state():
    state = {}
    state = update_state(
        state, "file-1", "./data/raw_pdfs/cpc-sections/x.pdf", "h1",
        drive_md5="md5-a", embedding_model="m", embedding_done=True,
    )
    return state


def test_skip_when_unchanged_and_embedded():
    state = _embedded_state()
    assert decide_processing(state, "file-1", "md5-a", "./data/raw_pdfs/cpc-sections/x.pdf") == "skip"


def test_download_when_drive_md5_changed():
    state = _embedded_state()
    assert decide_processing(state, "file-1", "md5-B", "./data/raw_pdfs/cpc-sections/x.pdf") == "download"


def test_embed_when_interrupted_embedding():
    state = {}
    state = update_state(state, "file-1", "./data/raw_pdfs/cpc-sections/x.pdf", "h1", drive_md5="md5-a", embedding_done=False)
    # unchanged remote, embeddings not done -> reuse the local file
    assert decide_processing(state, "file-1", "md5-a", "./data/raw_pdfs/cpc-sections/x.pdf") == "embed"


def test_new_file_downloads():
    state = _embedded_state()
    assert decide_processing(state, "file-new", "md5-x", "./data/raw_pdfs/cpc-sections/y.pdf") == "download"


def test_legacy_path_key_migrates_to_file_id():
    # Old state files keyed by local path (sometimes Windows-style and with
    # old folder names) never matched the current layout.
    state = {
        "C:\\data\\raw_pdfs\\CPC Section\\z.pdf": {
            "file_id": "file-legacy",
            "hash": "h",
            "drive_md5": "md5-l",
            "embeddings": {"done": True, "model": "old"},
        },
    }
    decision = decide_processing(state, "file-legacy", "md5-l", "./data/raw_pdfs/cpc-sections/z.pdf")
    assert decision == "skip"
    assert "file-legacy" in state
    assert "C:\\data\\raw_pdfs\\CPC Section\\z.pdf" not in state


def test_legacy_entry_without_drive_md5_downloads_once():
    state = {
        "./data/raw_pdfs/old-folder/z.pdf": {
            "file_id": "f-orphan",
            "hash": "h",
            "embeddings": {"done": True},
        },
    }
    # no drive_md5 baseline -> fetch content once to compare
    assert decide_processing(state, "f-orphan", "md5-z", "./data/raw_pdfs/cpc-sections/z.pdf") == "download"
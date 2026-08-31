import os
import glob
import threading
from fastapi import FastAPI, Query, HTTPException, Depends

# --------------------------------------------------------------
# Local development: load .env
# Cloud Run: ENV=prod → .env is NOT loaded
# --------------------------------------------------------------
from dotenv import load_dotenv

ENV = os.getenv("ENV", "local")

if ENV == "local":
    if os.path.exists(".env"):
        load_dotenv(".env")
        print("[ENV] Loaded local .env")
    else:
        print("[ENV] No .env found — using system environment")

# --------------------------------------------------------------
# App + Routers
# --------------------------------------------------------------

from src.backend.rag import run_rag
from src.common.logger import get_logger
from src.common.config import get_drive_root_folder_id
from src.ingestion.ingest import run_ingestion
from src.backend.auth import require_internal_key
from src.backend.drive_watcher import router as drive_router
from src.backend.drive_watch_refresh import router as refresh_router

logger = get_logger(__name__)
app = FastAPI(
    title="Legal RAG Assistant",
    docs_url=None if ENV == "prod" else "/docs",
    redoc_url=None if ENV == "prod" else "/redoc",
)

_ingest_lock = threading.Lock()

app.include_router(drive_router, dependencies=[Depends(require_internal_key)])
app.include_router(refresh_router, dependencies=[Depends(require_internal_key)])

# ---------------------------- Health Check --------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "env": ENV}

# ---------------------------- List local JSON docs --------------------------- #
@app.get("/docs", dependencies=[Depends(require_internal_key)])
def list_docs():
    files = glob.glob("./data/processed_json/**/*.json", recursive=True)
    return {"documents": [os.path.basename(f) for f in files]}

# ---------------------------- RAG Query --------------------------- #
@app.get("/query", dependencies=[Depends(require_internal_key)])
def query_api(q: str = Query(...), k: int = Query(5, ge=1, le=10)):
    return run_rag(q, k)

# ---------------------------- Manual Ingestion --------------------------- #
@app.post("/ingest", dependencies=[Depends(require_internal_key)])
def trigger_ingestion():
    acquired = _ingest_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=429, detail="Ingestion already running")
    try:
        root_id = get_drive_root_folder_id()
        if not root_id:
            raise HTTPException(
                status_code=500,
                detail="DRIVE_ROOT_FOLDER_ID not set"
            )

        logger.info(f"[INGEST] Running ingestion for root: {root_id}")
        run_ingestion(root_id)
        return {"status": "ok", "root": root_id}
    finally:
        _ingest_lock.release()

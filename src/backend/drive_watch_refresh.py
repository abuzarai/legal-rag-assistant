from fastapi import APIRouter
from googleapiclient.discovery import build
import google.auth
from google.oauth2 import service_account
import os
from src.common.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

SCOPES = ["https://www.googleapis.com/auth/drive"]

@router.post("/refresh-watch")
def refresh_watch():
    """
    Re-register the Drive webhook watch.

    Cloud Run → uses ADC (IAM)
    Local dev → uses GOOGLE_APPLICATION_CREDENTIALS
    """
    folder_id = os.getenv("DRIVE_ROOT_FOLDER_ID")
    webhook_url = os.getenv("DRIVE_WEBHOOK_URL")

    if not folder_id:
        raise RuntimeError("DRIVE_ROOT_FOLDER_ID environment variable not set.")
    if not webhook_url:
        raise RuntimeError("DRIVE_WEBHOOK_URL environment variable not set.")

    # Try ADC (Cloud Run)
    creds = None
    try:
        creds, _ = google.auth.default(scopes=SCOPES)
        logger.info("[WATCH] Using ADC credentials")
    except Exception:
        pass

    # Fallback: local service account file
    if creds is None:
        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
        logger.info(f"[WATCH] Using local credentials: {sa_path}")
        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=SCOPES
        )

    drive = build("drive", "v3", credentials=creds, cache_discovery=False)

    body = {
        "id": "legal-rag-folder-watch",
        "type": "web_hook",
        "address": webhook_url,
    }

    logger.info(f"[WATCH] Registering webhook for folder {folder_id} → {webhook_url}")
    resp = drive.files().watch(fileId=folder_id, body=body).execute()
    return {"watch": resp}

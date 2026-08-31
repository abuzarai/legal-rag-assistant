import base64
import json
import os
import time

from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from src.ingestion.ingest import run_ingestion, IngestBusyError

router = APIRouter()

# Pub/Sub push subscriptions can send ?token=<value>; require it when
# DRIVE_PUBSUB_TOKEN is configured.
_DEBOUNCE_SECONDS = float(os.getenv("DRIVE_EVENT_DEBOUNCE_SECONDS", "300"))
_last_trigger_ts = 0.0


@router.post("/drive-events")
async def handle_drive_event(
    request: Request, background_tasks: BackgroundTasks, token: str = ""
):
    """Handle Drive push notifications from Pub/Sub.

    Pub/Sub base64-encodes message.data; a raw-JSON parse of it always failed
    (500 + endless redelivery). Decode defensively, debounce repeated events,
    and let run_ingestion's shared lock arbitrate concurrent triggers.
    """
    global _last_trigger_ts

    expected = os.getenv("DRIVE_PUBSUB_TOKEN", "")
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid push token")

    try:
        envelope = await request.json()
    except Exception:
        # Ack malformed payloads so Pub/Sub stops redelivering them.
        return {"status": "ignored", "reason": "invalid envelope"}

    message = envelope.get("message", {})
    raw = message.get("data")
    if raw:
        try:
            decoded = base64.b64decode(raw).decode("utf-8")
            event = json.loads(decoded)
            print(f"[Drive Event] {event}")
        except Exception as exc:
            print(f"[Drive Event] unparseable payload: {exc}")

    # Debounce: a burst of file-change notifications must not each trigger a
    # full re-scan of the corpus.
    now = time.monotonic()
    if now - _last_trigger_ts < _DEBOUNCE_SECONDS:
        return {"status": "ignored", "reason": "debounced"}

    _last_trigger_ts = now
    background_tasks.add_task(_guarded_run)
    return {"status": "ingestion triggered"}


def _guarded_run():
    try:
        run_ingestion()
    except IngestBusyError:
        # A concurrent run already covers this; the per-file state saves make
        # this run fully resumable, so dropping the event is safe.
        print("[Drive Event] ingestion already running; skipping.")
    except Exception as exc:  # noqa: BLE001 — background task must not die silently
        print(f"[Drive Event] ingestion failed: {exc}")
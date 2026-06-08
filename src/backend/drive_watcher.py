from fastapi import APIRouter, Request, BackgroundTasks
from src.ingestion.ingest import run_ingestion
import json

router = APIRouter()

@router.post("/drive-events")
async def handle_drive_event(request: Request, background_tasks: BackgroundTasks):
    """Handle Drive push notifications from Pub/Sub."""
    envelope = await request.json()
    message = envelope.get("message", {})
    data = message.get("data")

    if data:
        event = json.loads(
            (data.encode("utf-8") if isinstance(data, str) else data)
        )
        print(f"[Drive Event] {event}")

    # Run ingestion in background to avoid blocking Pub/Sub
    background_tasks.add_task(run_ingestion)

    return {"status": "ingestion triggered"}

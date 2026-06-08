from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.backend.main import app
from src.backend import main as backend_main


client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint_returns_social_mode_for_greeting(monkeypatch):
    monkeypatch.setattr(
        "src.backend.rag.generate_social_reply",
        lambda _query: "Hi! Share your legal issue and I can guide you.",
    )

    response = client.get("/query", params={"q": "Hello", "k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "social"
    assert "legal issue" in body["answer"].lower()
    assert body["sources"] == []


def test_query_endpoint_returns_uncertain_mode(monkeypatch):
    monkeypatch.setattr(
        "src.backend.rag.uncertain_response", lambda: "Please share more facts."
    )

    response = client.get("/query", params={"q": "need help", "k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "uncertain"
    assert body["answer"] == "Please share more facts."


def test_query_endpoint_returns_invalid_on_missing_param():
    response = client.get("/query")

    assert response.status_code == 422


def test_ingest_endpoint_returns_500_when_drive_root_missing(monkeypatch):
    monkeypatch.setattr(backend_main, "get_drive_root_folder_id", lambda: "")

    response = client.post("/ingest")

    assert response.status_code == 500
    assert response.json()["detail"] == "DRIVE_ROOT_FOLDER_ID not set"


def test_ingest_endpoint_success(monkeypatch):
    called = {"root": None}

    monkeypatch.setattr(
        backend_main, "get_drive_root_folder_id", lambda: "root-folder-abc"
    )
    monkeypatch.setattr(
        backend_main, "run_ingestion", lambda root_id: called.update(root=root_id)
    )

    response = client.post("/ingest")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "root": "root-folder-abc"}
    assert called["root"] == "root-folder-abc"

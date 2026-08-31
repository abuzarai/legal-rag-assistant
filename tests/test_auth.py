"""Auth enforcement tests (INTERNAL_API_KEY on all routes except /health)."""

import os

os.environ["ENV"] = "prod"  # disable built-in docs; enforce the custom /docs auth
os.environ["INTERNAL_API_KEY"] = "test-internal-key"

from fastapi.testclient import TestClient  # noqa: E402

from src.backend.main import app  # noqa: E402

client = TestClient(app)


def test_health_stays_public():
    assert client.get("/health").status_code == 200


def test_docs_requires_key():
    assert client.get("/docs").status_code == 401


def test_query_requires_key():
    assert client.get("/query", params={"q": "test"}).status_code == 401


def test_ingest_requires_key():
    assert client.post("/ingest").status_code == 401


def test_valid_key_accepted():
    headers = {"x-internal-key": "test-internal-key"}
    assert client.get("/docs", headers=headers).status_code == 200


def test_wrong_key_rejected():
    assert client.get("/docs", headers={"x-internal-key": "nope"}).status_code == 401
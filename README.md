# Legal RAG Assistant

> **Final Year Project — AI Microservice** · Part of the [Insafdaar](https://github.com/abuzarai/insafdaar-webapp) legal case management platform.  
> A retrieval-augmented generation (RAG) service that answers legal questions grounded in Pakistani case law and CPC sections.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Weaviate](https://img.shields.io/badge/Weaviate-v4-5C3FFB?logo=weaviate&logoColor=white)](https://weaviate.io)
[![Gemini](https://img.shields.io/badge/Gemini-Vertex_AI-4285F4?logo=google&logoColor=white)](https://cloud.google.com/vertex-ai)

---

## 📖 What Is This?

This microservice ingests Pakistani legal documents (case law PDFs, CPC sections) from Google Drive, chunks and embeds them into a Weaviate vector store, and exposes a FastAPI endpoint that answers legal questions with cited sources via Gemini on Vertex AI.

It powers the **Legal Assistant Chat** inside the main Insafdaar webapp — advocates ask questions like *"What is Order VII Rule 11?"* and get a structured response with summary, legal analysis, and source citations.

---

## 🏗️ Architecture

```
Google Drive (PDFs, TXTs)
      │
      ▼
┌─────────────────────────────────────────────┐
│          INGESTION PIPELINE                  │
│                                             │
│  drive_fetcher.py       Recursive BFS scan  │
│  text_extractor.py      PyPDFLoader for PDF │
│  to_json.py             JSON snapshot backup│
│  embedder.py            Chunk → Embed →     │
│                         Upsert to Weaviate  │
│                         (1000-char chunks,  │
│                          100-char overlap)  │
│  state_manager.py       MD5 hash tracking,  │
│                         file/GCS backend    │
└─────────────┬───────────────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │     WEAVIATE v4     │
    │                     │
    │  Collection:        │
    │   "LegalChunk"      │
    │  Properties:        │
    │   content (text)    │
    │   source (text)     │
    │   page (text)       │
    │   drive_id (text)   │
    │   category (text)   │
    │  Vectorizer: BYOV   │
    │  (Gemini embeddings)│
    └──────────┬──────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│             BACKEND (FastAPI)                │
│                                              │
│  RAG Pipeline (rag.py):                      │
│                                              │
│  1. detect_mode(query)                       │
│     ├─ "social" → greeting/casual reply      │
│     ├─ "uncertain" → clarification prompt    │
│     └─ "legal" → full retrieval & generation │
│                                              │
│  2. similarity_search(query, k)              │
│     ├─ Embed query (text-embedding-005)      │
│     ├─ Hybrid search (Weaviate: alpha=0.5)   │
│     └─ Cosine-similarity reranking           │
│                                              │
│  3. is_retrieval_weak(docs)                  │
│     └─ Heuristic check (distance thresholds) │
│                                              │
│  4. Gemini (gemini-2.5-flash, temp=0.2)     │
│     └─ Prompt: Summary + Detailed Analysis   │
│        (Issue · Rule · Application ·         │
│         Next Step) + Citations               │
│                                              │
│  Endpoints:                                  │
│  GET  /query?q=...&k=5  → RAG query         │
│  POST /ingest            → Trigger ingestion │
│  POST /drive-events      → Drive Pub/Sub     │
│  POST /refresh-watch     → Renew webhook     │
│  GET  /health            → Health check      │
│  GET  /docs              → List docs         │
└──────────────────────────────────────────────┘
```

---

## 🔌 API Reference

### `GET /query`

Run a RAG query against the vector store.

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q` | string | **required** | Natural language legal question (max 2000 chars) |
| `k` | integer | `5` | Number of source documents to retrieve |

**Response:**

```json
{
  "query": "What is Order VII Rule 11?",
  "mode": "legal",
  "answer": "... raw Gemini response ...",
  "summary": "Order VII Rule 11 CPC empowers a court to reject a plaint on specific grounds...",
  "analysis": "1. **Issue:** Whether the plaint discloses a cause of action...\n2. **Rule:** Order VII Rule 11...\n3. **Application:** ...\n4. **Practical Next Step:** ...",
  "citations": ["Order VII Rule 11, CPC", "Muhammad Ashraf v. Federation of Pakistan (2023 SCMR 1234)"],
  "sources": [
    {"title": "Civil Procedure Code, 1908 -- Order VII -- p.45", "link": "https://drive.google.com/file/d/.../view"},
    {"title": "2023 SCMR 1234 -- p.12", "link": "https://drive.google.com/file/d/.../view"}
  ]
}
```

**Mode-dependent responses:**

| Mode | When | Returns |
|------|------|---------|
| `legal` | Query matches legal keywords (court, FIR, bail, cpc, appeal, etc.) | Full RAG with summary, analysis, citations, sources |
| `social` | Greeting/thanks/non-legal chat | Friendly message or non-legal redirect |
| `uncertain` | Vague/unclear query ("need help", "legal issue") | Prompt to share specific case facts |

### `POST /ingest`

Trigger a full re-ingestion of all documents from the configured Google Drive folder. Thread-safe — only one ingestion runs at a time.

**Response:** `{"status": "ok", "root": "<folder-id>"}` or `429` if already running.

### `POST /drive-events`

Receives Google Drive Pub/Sub push notifications for file changes and triggers ingestion as a background task.

### `POST /refresh-watch`

Re-registers the Drive webhook channel (channels expire after 7 days).

### `GET /health`

**Response:** `{"status": "ok", "env": "local"}`

### `GET /docs`

**Response:** `{"documents": ["filename1.json", "filename2.json", ...]}` — lists all processed JSON snapshots.

---

## 📥 Ingestion Pipeline

The ingestion pipeline converts legal documents from Google Drive into searchable vector embeddings:

### 1. Document Fetching (`drive_fetcher.py`)

- Recursive BFS traversal of all subfolders from a root Drive folder
- Supports shared drives
- Tracks files by `DriveFile` dataclass (id, name, mime_type, md5_checksum, category, relative_path)
- Folder hierarchy maps to document categories (e.g., `cpc-sections/`, `case-laws/`)

### 2. Text Extraction (`text_extractor.py`)

| Format | Tool | Output |
|--------|------|--------|
| PDF | `PyPDFLoader` (langchain) | Page-by-page `{page_content, metadata}` |
| TXT | Direct read | Single-entry `{page_content, metadata}` |

### 3. Chunking & Embedding (`embedder.py`)

- `RecursiveCharacterTextSplitter`: 1000-char chunks, 100-char overlap
- Embeddings: Gemini `text-embedding-005` via Vertex AI
- Rate-limit handling: Exponential backoff (up to 5 retries)
- Batch upload: 32 documents at a time to Weaviate, 10s delay between batches

### 4. State Management (`state_manager.py`)

Tracks which files have been processed and whether their embeddings are up-to-date:

```json
{
  "file_id": "1abc...",
  "hash": "md5-hash",
  "last_processed": "2026-01-27T12:00:00",
  "embeddings": {
    "model": "text-embedding-005",
    "done": true,
    "last_embedded": "2026-01-27T12:00:05"
  }
}
```

Supports two backends:
- **File** — `./artifacts/ingestion_state.json` (local dev)
- **GCS** — Google Cloud Storage blob (Cloud Run for persistence across restarts)

### 5. Repository-to-Category Mapping

Documents are organized by Drive folder structure into normalized categories:

| Drive Folder | Normalized Category |
|-------------|-------------------|
| `cpc/` or `cpc-sections/` | `cpc-sections` |
| `caselaws/` or `case law/` | `case-laws` |
| *(any other)* | Passed through as-is |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI + Uvicorn |
| **Vector Store** | Weaviate v4 (self-hosted on GCE or cloud) |
| **Embeddings** | Gemini `text-embedding-005` via Vertex AI |
| **LLM** | Gemini 2.5 Flash via Vertex AI (`temperature=0.2`) |
| **Document Source** | Google Drive API v3 (recursive folder scan) |
| **PDF Extraction** | PyPDFLoader (langchain) |
| **Text Splitting** | RecursiveCharacterTextSplitter (1000/100) |
| **Deployment** | Google Cloud Run (auto-deploy from GitHub) |
| **Language** | Python 3.13 |
| **Package Manager** | `uv` |

---

## 🚀 Local Development

### Prerequisites

- Python 3.13+ with [uv](https://docs.astral.sh/uv/)
- GCP service account with Vertex AI, Drive, and Storage access
- Weaviate instance (local Docker or remote)

### Setup

```bash
# Clone
git clone https://github.com/abuzarai/legal-rag-assistant.git
cd legal-rag-assistant

# Environment
cp .env.example .env
# Edit .env with your credentials

# Install
uv sync
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes | `service_account.json` | Path to GCP service account key |
| `GOOGLE_CLOUD_PROJECT` | Yes | — | GCP project ID for Vertex AI |
| `GOOGLE_VERTEX_LOCATION` | No | `asia-south1` | Vertex AI region |
| `WEAVIATE_URL` | Yes | — | Weaviate endpoint (e.g., `http://localhost:8080`) |
| `WEAVIATE_API_KEY` | Yes | — | Weaviate API key |
| `WEAVIATE_COLLECTION` | No | `LegalChunk` | Weaviate collection name |
| `WEAVIATE_GRPC_PORT` | No | `50051` | Weaviate gRPC port |
| `DRIVE_ROOT_FOLDER_ID` | Yes | — | Root Google Drive folder ID for ingestion |
| `DRIVE_ALLOWED_EXTS` | No | `pdf` | Comma-separated allowed file extensions |
| `INGESTION_STATE_BACKEND` | No | `file` | State storage: `file` or `gcs` |
| `GEMINI_API_KEY` | No | — | Only for local Gemini API (not Vertex AI) |

### Run

```bash
# 1. Ingest documents into Weaviate
uv run python -m src.ingestion.ingest

# 2. Start the API
uv run uvicorn src.backend.main:app --reload

# 3. Query
curl "http://localhost:8000/query?q=What+is+Order+VII+Rule+11&k=5"
```

### Test

```bash
uv run pytest tests/test_rag_api.py -v
```

---

## ☁️ Deployment

The service is deployed on **Google Cloud Run** with GitHub-connected auto-deploy. Key configuration:

- **Service**: Cloud Run (min-scale 0, max-scale 2, concurrency 10)
- **Environment**: `ENV=prod` (disables `.env` loading, uses Cloud Run env vars/secrets)
- **Ingress**: Internal + Cloud Load Balancing (private to the main webapp's VPC)
- **Auth**: IAM-based (service accounts) for `/ingest`; unauthenticated for `/query`
- **Auto-ingestion**: Cloud Scheduler triggers `POST /ingest` daily, or Drive Pub/Sub webhook triggers on change

### Self-hosting Weaviate

See [`docs/weaviate_gce.md`](docs/weaviate_gce.md) for setting up Weaviate on a GCE VM with Docker and API-key auth.

---

## 📂 Repository Structure

```
legal-rag-assistant/
├── src/
│   ├── backend/
│   │   ├── main.py                # FastAPI app, routes
│   │   ├── deps.py                # Embeddings, Weaviate search, reranker
│   │   ├── rag.py                 # RAG pipeline, mode detection, Gemini
│   │   ├── rerank.py              # Gemini-based reranker (experimental)
│   │   ├── drive_watcher.py       # Drive Pub/Sub webhook
│   │   └── drive_watch_refresh.py # Webhook channel renewal
│   ├── common/
│   │   ├── config.py              # Environment variable wrappers
│   │   ├── logger.py              # Structured logging
│   │   └── weaviate_client.py     # Weaviate v4 client + schema management
│   └── ingestion/
│       ├── ingest.py              # Main ingestion pipeline
│       ├── drive_fetcher.py       # Google Drive recursive BFS scanner
│       ├── text_extractor.py      # PDF/TXT text extraction
│       ├── to_json.py             # JSON snapshot writer
│       ├── state_manager.py       # Ingestion state (file or GCS)
│       └── embedder.py            # Chunking, embedding, Weaviate upsert
├── tests/
│   └── test_rag_api.py            # FastAPI TestClient tests
├── docs/
│   ├── weaviate_gce.md            # Self-hosted Weaviate setup guide
│   └── cloud_run_scheduler.md     # Cloud Scheduler /ingest trigger guide
├── Dockerfile                     # Cloud Run build
└── pyproject.toml                 # Dependencies & project metadata
```

---

## 📝 License

Licensed under the [Apache License 2.0](LICENSE).  

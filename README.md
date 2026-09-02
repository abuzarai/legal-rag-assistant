# Legal RAG Assistant

> Final Year Project, AI microservice · Part of the [Insafdaar](https://github.com/abuzarai/insafdaar-webapp) legal case management platform.  
> A retrieval-augmented generation (RAG) service that answers legal questions grounded in Pakistani case law and CPC sections.

[![License](https://img.shields.io/badge/License-PolyForm_Noncommercial-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Weaviate](https://img.shields.io/badge/Weaviate-v4-5C3FFB?logo=weaviate&logoColor=white)](https://weaviate.io)
[![Gemini](https://img.shields.io/badge/Gemini-API-4285F4?logo=google&logoColor=white)](https://ai.google.dev)

---

## What Is This?

This microservice ingests Pakistani legal documents (case law PDFs, CPC sections) from Google Drive, chunks and embeds them into a Weaviate vector store, and exposes a FastAPI endpoint that answers legal questions with cited sources using the Gemini API.

It powers the **Legal Assistant Chat** inside the main Insafdaar webapp. Users ask questions like *"What is Order VII Rule 11?"* and get a structured response with summary, legal analysis, and source citations.

---

## Architecture

```
Google Drive (PDFs, TXTs)
      │
      ▼
┌─────────────────────────────────────────────┐
│          INGESTION PIPELINE                  │
│                                             │
│  drive_fetcher.py       Recursive BFS scan  │
│  text_extractor.py      PDF/TXT extraction  │
│  embedder.py            Chunk → Embed →     │
│                         Upsert to Weaviate  │
│                         (1000-char chunks,  │
│                          100-char overlap)  │
│  state_manager.py       MD5 hash tracking,  │
│                         idempotent re-runs  │
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
    │  Vectorizer: none   │
    │  (bring-your-own    │
    │   Gemini embeddings)│
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
│     ├─ Embed query (gemini-embedding-001)    │
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
│  Endpoints (all gated on x-internal-key):    │
│  GET  /query?q=...&k=5  → RAG query         │
│  POST /ingest            → Trigger ingestion │
│  GET  /health            → Health check      │
│  GET  /docs              → Ingested file list│
└──────────────────────────────────────────────┘
```

---

<details>
<summary>API Reference</summary>

### `GET /query`

Run a RAG query against the vector store. Requires the `x-internal-key` header.

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

Trigger ingestion of new/changed documents from the configured Google Drive folder. Re-runs skip unchanged files (MD5-tracked), and only one ingestion runs at a time.

**Response:** `{"status": "ok", "root": "<folder-id>"}` or `429` if already running.

### `GET /health`

**Response:** `{"status": "ok", "env": "prod"}`

### `GET /docs`

**Response:** `{"documents": ["filename1.json", "filename2.json", ...]}` lists the ingested documents.

</details>

---

## Ingestion Pipeline

The ingestion pipeline converts legal documents from Google Drive into searchable vector embeddings:

### 1. Document Fetching

- Recursive BFS traversal of all subfolders from a root Drive folder
- Tracks files by `DriveFile` dataclass (id, name, mime_type, md5_checksum, category, relative_path)
- Folder hierarchy maps to document categories (e.g., `cpc-sections/`, `case-laws/`)

### 2. Text Extraction

| Format | Tool | Output |
|--------|------|--------|
| PDF | `PyPDFLoader` (langchain) | Page-by-page `{page_content, metadata}` |
| TXT | Direct read | Single-entry `{page_content, metadata}` |

### 3. Chunking & Embedding

- `RecursiveCharacterTextSplitter`: 1000-char chunks, 100-char overlap
- Embeddings: `gemini-embedding-001` via the Gemini API (768 dims, `output_dimensionality=768`)
- Rate-limit handling: exponential backoff (up to 5 retries, gentle pacing)
- Batch upload: 32 documents at a time to Weaviate, 10s delay between batches

### 4. State Management

Tracks which files have been processed and whether their embeddings are up-to-date:

```json
{
  "file_id": "1abc...",
  "hash": "md5-hash",
  "last_processed": "2026-01-27T12:00:00",
  "embeddings": {
    "model": "gemini-embedding-001",
    "done": true,
    "last_embedded": "2026-01-27T12:00:05"
  }
}
```

File-backed state (`./artifacts/ingestion_state.json`) with deterministic chunk UUIDs, so interrupted runs resume and re-runs never duplicate.

### 5. Repository-to-Category Mapping

Documents are organized by Drive folder structure into normalized categories:

| Drive Folder | Normalized Category |
|-------------|-------------------|
| `cpc/` or `cpc-sections/` | `cpc-sections` |
| `caselaws/` or `case law/` | `case-laws` |
| *(any other)* | Passed through as-is |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI + Uvicorn |
| **Vector Store** | Weaviate v4 (self-hosted) |
| **Embeddings** | Gemini `gemini-embedding-001` via the Gemini API |
| **LLM** | Gemini 2.5 Flash via the Gemini API (`temperature=0.2`) |
| **Document Source** | Google Drive API v3 (recursive folder scan, hourly refresh) |
| **PDF Extraction** | PyPDFLoader (langchain) |
| **Text Splitting** | RecursiveCharacterTextSplitter (1000/100) |
| **Deployment** | Container in the Insafdaar compose stack (Oracle Cloud Infrastructure) |
| **Language** | Python 3.13 |
| **Package Manager** | `uv` |

---

## Local Development

### Prerequisites

- Python 3.13+ with [uv](https://docs.astral.sh/uv/)
- Gemini API key ([AI Studio](https://aistudio.google.com/apikey))
- Google Drive service account (read access to the corpus folder)
- Weaviate instance (local Docker or compose)

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
| `GEMINI_API_KEY` | Yes | None | Gemini API key (embeddings + LLM) |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | LLM model name |
| `EMBEDDING_MODEL` | No | `gemini-embedding-001` | Embeddings model |
| `EMBEDDING_OUTPUT_DIMS` | No | `768` | Embedding dimensionality |
| `WEAVIATE_URL` | Yes | None | Weaviate endpoint (e.g., `http://localhost:8080`) |
| `WEAVIATE_API_KEY` | Yes | None | Weaviate API key |
| `WEAVIATE_COLLECTION` | No | `LegalChunk` | Weaviate collection name |
| `WEAVIATE_GRPC_PORT` | No | `50051` | Weaviate gRPC port |
| `DRIVE_ROOT_FOLDER_ID` | Yes | None | Root Google Drive folder ID for ingestion |
| `DRIVE_ALLOWED_EXTS` | No | `pdf` | Comma-separated allowed file extensions |
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes (Drive) | None | Path to the Drive service-account key |
| `INGESTION_STATE_BACKEND` | No | `file` | State storage backend (`file`) |
| `INTERNAL_API_KEY` | No | None | Shared secret the webapp sends with `x-internal-key` |

### Run

```bash
# 1. Ingest documents into Weaviate (resumable; re-run to pick up changes)
uv run python -m src.ingestion.ingest

# 2. Start the API
uv run uvicorn src.backend.main:app --reload

# 3. Query (needs the internal key header)
curl -H "x-internal-key: YOUR_KEY" "http://localhost:8000/query?q=What+is+Order+VII+Rule+11&k=5"
```

### Test

```bash
uv run pytest tests/test_rag_api.py -v
```

---

## Deployment

The service runs as a container in the Insafdaar compose stack. Key configuration:

- **Environment**: `ENV=prod` (disables `.env` loading, uses runtime env vars)
- **Auth**: every endpoint except `/health` enforces the shared `x-internal-key`
- **Ingestion cadence**: the host triggers `POST /ingest` hourly via cron; resumable state makes re-runs idempotent

Deploys are handled by the main webapp's pipeline (GitHub Actions builds the image on a runner, ships it, and applies the stack).

---

## Repository Structure

```
legal-rag-assistant/
├── src/
│   ├── backend/
│   │   ├── main.py                # FastAPI app, routes
│   │   ├── deps.py                # Embeddings, Weaviate search, reranker
│   │   ├── rag.py                 # RAG pipeline, mode detection, Gemini
│   │   └── rerank.py              # Local cosine reranker
│   ├── common/
│   │   ├── config.py              # Environment variable wrappers
│   │   ├── logger.py              # Structured logging
│   │   └── weaviate_client.py     # Weaviate v4 client + schema management
│   └── ingestion/
│       ├── ingest.py              # Main ingestion pipeline
│       ├── drive_fetcher.py       # Google Drive recursive BFS scanner
│       ├── text_extractor.py      # PDF/TXT text extraction
│       ├── state_manager.py       # Ingestion state (file backend)
│       └── embedder.py            # Chunking, embedding, Weaviate upsert
├── tests/
│   └── test_rag_api.py            # FastAPI TestClient tests
├── Dockerfile                     # Container build
└── pyproject.toml                 # Dependencies & project metadata
```

---

## License

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires written permission from the author.
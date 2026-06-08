# Legal RAG Assistant

> **Final Year Project — AI Microservice** · Part of the [Insafdaar](https://github.com/abuzarai/insafdaar-webapp) legal case management platform.  
> A retrieval-augmented generation (RAG) stack for Pakistani case law and CPC sections.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/Gemini-Vertex_AI-4285F4?logo=google&logoColor=white)](https://cloud.google.com/vertex-ai)

---

## 📖 What Is This?

This microservice ingests Pakistani legal documents (case law PDFs, CPC sections) from Google Drive, chunks and embeds them into a Weaviate vector store, and exposes a FastAPI endpoint that answers legal questions with grounded citations via Gemini on Vertex AI.

It powers the **Legal Assistant Chat** inside the main Insafdaar webapp — advocates can ask questions like *"What is Order VII Rule 11?"* and get answers with source citations.

### Key Components

- **`src/ingestion`** — Downloads PDFs from Google Drive, extracts text, chunks, embeds via Vertex AI, and upserts to Weaviate
- **`src/backend`** — FastAPI app that runs similarity search on Weaviate and feeds results to Gemini for grounded answers
- **`infra/`** — Cloud Run service definition and GCE Weaviate setup guide

---

## 🏗️ Architecture

```
Google Drive (PDFs)
     │
     ▼
Ingestion Pipeline ──► Weaviate (Vector Store)
     │                       │
     ▼                       ▼
  JSON Snapshots      FastAPI Query Endpoint
     │                       │
     └─────── Gemini ────────┘
              (Vertex AI)
```

---

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **Vector Store**: Weaviate (self-hosted on GCE)
- **Embeddings + LLM**: Gemini on Vertex AI
- **Data Source**: Google Drive API (recursive folder scan)
- **Deployment**: Google Cloud Run
- **Language**: Python 3.13

---

## 🚀 Local Development

### Prerequisites

- Python 3.13+ with [uv](https://docs.astral.sh/uv/)
- GCP service account with Vertex AI, Drive, and Storage access
- Weaviate instance (local or remote)

### Setup

```bash
# Clone
git clone https://github.com/abuzarai/legal-rag-assistant.git
cd legal-rag-assistant

# Environment
cp .env.example .env
# Edit .env with your GCP project, Weaviate URL, Drive folder ID, etc.

# Install dependencies
uv sync
```

### Environment Variables

```
ENV=local
GOOGLE_APPLICATION_CREDENTIALS=service_account.json
GOOGLE_CLOUD_PROJECT=your-gcp-project
GOOGLE_VERTEX_LOCATION=us-central1

WEAVIATE_URL=http://<weaviate-vm-ip>:8080
WEAVIATE_API_KEY=your-weaviate-api-key
WEAVIATE_COLLECTION=LegalChunk

DRIVE_ROOT_FOLDER_ID=your-google-drive-folder-id
```

### Run

```bash
# 1. Ingest documents into Weaviate
uv run python -m src.ingestion.ingest

# 2. Start the API
uv run uvicorn src.backend.main:app --reload

# 3. Query
curl "http://localhost:8000/query?q=What+is+Order+VII+Rule+11&k=5"
```

---

## ☁️ Deployment

Deployed on **Google Cloud Run** with GitHub-connected auto-deploy. The `POST /ingest` endpoint can be triggered on a schedule via Cloud Scheduler for continuous re-ingestion.

---

## 📝 License

Licensed under the [Apache License 2.0](LICENSE).  

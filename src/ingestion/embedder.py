import os
import time
import random
import datetime
import uuid
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from weaviate.exceptions import WeaviateBaseError
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter as WeaviateFilter
from src.common.logger import get_logger
from src.common.weaviate_client import get_weaviate_client, ensure_collection
from src.common.config import (
    get_weaviate_collection,
    get_embedding_output_dims,
    get_embedding_model,
)

load_dotenv(dotenv_path="./.env", override=True)
logger = get_logger(__name__)


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )


def get_embedder():
    from src.common.config import (
        get_embedding_model,
        get_embedding_output_dims,
        get_gemini_api_key,
    )

    dims = get_embedding_output_dims()
    api_key = get_gemini_api_key()
    if api_key:
        logger.info(
            "[EMBED] Gemini API embeddings (%s, dims=%s)", get_embedding_model(), dims
        )
        return GoogleGenerativeAIEmbeddings(
            model=get_embedding_model(),
            google_api_key=api_key,
            output_dimensionality=dims,
        )

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "Missing Gemini credentials: set GEMINI_API_KEY (or GOOGLE_CLOUD_PROJECT for Vertex AI)."
        )
    location = os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-005",
        project=project,
        location=location,
        vertexai=True,
        output_dimensionality=dims,
    )


def _batch_embed(embedder, texts: list, dims: int = 768) -> list[list[float]]:
    """
    Embed a list of texts in one request via google-genai embed_content.

    langchain's embed_documents uses the single-content path which hits the
    free-tier per-request limit. google-genai's embed_content accepts a list
    of contents in one request and returns one embedding per content, which
    stays under the per-minute ceiling for a typical batch.
    Falls back to embed_documents on any unexpected error.
    """
    client = getattr(embedder, "client", None)
    model = getattr(embedder, "model", None)
    if client is None or model is None:
        return embedder.embed_documents(texts)

    try:
        result = client.models.embed_content(
            model=(
                f"models/{model}"
                if not str(model).startswith("models/")
                else str(model)
            ),
            contents=[{"parts": [{"text": t}]} for t in texts],
            config={
                "outputDimensionality": dims,
                "taskType": "RETRIEVAL_DOCUMENT",
            },
        )
        return [list(e.values) for e in result.embeddings]
    except Exception:
        return embedder.embed_documents(texts)


def embed_with_retry(embedder, texts, max_retries=4):
    delay = 5
    transient_markers = [
        "429",
        "Resource has been exhausted",
        "Temporary failure in name resolution",
        "Name or service not known",
        "Connection reset",
        "Connection refused",
        "timed out",
        "Timeout",
        "getaddrinfo",
        "network is unreachable",
        "[Errno -3]",
        "[Errno 111]",
        "[Errno 110]",
    ]
    # The daily free-tier quota is NOT going to recover in 5-40s. If we see
    # the per-day quota violated, stop immediately with a clear message
    # instead of grinding retries that only burn more of tomorrow's budget.
    quota_exhausted_markers = [
        "EmbedContentRequestsPerDay",
        "you exceeded your current quota",
    ]
    for attempt in range(max_retries):
        try:
            return _batch_embed(embedder, texts, dims=get_embedding_output_dims())
        except Exception as e:
            msg = str(e)
            msg_lower = msg.lower()
            if any(m in msg_lower for m in quota_exhausted_markers):
                raise QuotaExhaustedError(
                    "Daily embedding quota exhausted (free tier). "
                    "Run again after midnight UTC; progress resumes from state."
                ) from e
            if any(m in msg for m in transient_markers):
                # Rate limits → exponential backoff + jitter; DNS/conn blips
                # → short capped backoff so transient flakes don't lose batches
                if "429" in msg or "Resource has been exhausted" in msg:
                    wait = delay * (2**attempt) + random.uniform(0, 2)
                    logger.warning(
                        f"[WARNING] Rate limited. Retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                else:
                    wait = min(2 + attempt * 2, 30) + random.uniform(0, 1)
                    logger.warning(
                        f"[WARNING] Transient network error, retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries}): {msg[:90]}"
                    )
                time.sleep(wait)
            else:
                logger.error(f"[ERROR] Non-transient embedding error: {msg[:200]}")
                return []
    logger.error(f"[ERROR] Failed to embed batch after {max_retries} retries.")
    return []


class QuotaExhaustedError(Exception):
    """Daily free-tier embedding quota is used up for today."""


def _prepare_metadata(chunk: Document, filepath: str, file_id: str | None) -> dict:
    metadata = chunk.metadata or {}
    metadata["source"] = metadata.get("source") or os.path.basename(filepath)
    page = metadata.get("page_label") or metadata.get("page") or "unknown"
    metadata["page"] = page
    if file_id:
        metadata["drive_id"] = file_id
    return metadata


def _upload_batch(client, collection_name: str, docs: List[Document], embeddings: List[List[float]]):
    """Upload a batch of document chunks and embeddings (v4 syntax)."""
    collection = client.collections.get(collection_name)
    logger.info(
        f"[INFO] Uploading {len(docs)} chunks to collection '{collection_name}'..."
    )

    objects = []
    for doc, vector in zip(docs, embeddings):
        metadata = doc.metadata or {}
        properties = {
            "content": doc.page_content,
            "source": metadata.get("source"),
            "page": str(metadata.get("page")),
            "drive_id": metadata.get("drive_id"),
            "category": metadata.get("category"),
        }
        # remove Nones
        properties = {k: v for k, v in properties.items() if v is not None}

        # Deterministic UUID per chunk: hash of source+page+content so a
        # re-run overwrites the same object instead of duplicating it.
        chunk_id = f"{properties.get('source')}|{properties.get('page')}|{properties.get('content')}"
        obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

        objects.append(DataObject(uuid=obj_uuid, properties=properties, vector=vector))

    collection.data.insert_many(objects)
    logger.info(f"[INFO] ✅ Uploaded {len(objects)} chunks successfully.")


def upsert_chunks(docs, state, filepath, file_id=None, batch_size=4, state_key=None):
    state_key = state_key if state_key is not None else filepath

    if not docs:
        logger.warning(f"[WARNING] No documents to embed for {filepath}")
        return state

    embedder = get_embedder()
    splitter = get_text_splitter()
    chunks = splitter.split_documents(docs)

    if not chunks:
        logger.warning(f"[WARNING] No chunks produced for {filepath}")
        return state

    client = get_weaviate_client()
    collection_name = get_weaviate_collection()
    ensure_collection(client, collection_name)

    # Inject metadata (source + page)
    for chunk in chunks:
        chunk.metadata = _prepare_metadata(chunk, filepath, file_id)

    # File-scoped upsert: this weaviate-client pin only supports batch delete
    # by filter, and `source` is unique per file — so delete the file's old
    # chunk set (also purging stale/renamed chunks) before inserting fresh
    # ones. Deterministic UUIDs keep inserts idempotent on crash-retry.
    file_source = None
    for chunk in chunks:
        s = (chunk.metadata or {}).get("source")
        if s:
            file_source = s
            break
    if file_source:
        try:
            collection = client.collections.get(collection_name)
            collection.data.delete_many(
                where=WeaviateFilter.by_property("source").equal(file_source)
            )
        except Exception as e:
            logger.warning(f"[WARN] Pre-insert delete for {file_source} failed (continuing): {e}")

    total_inserted = 0

    logger.info("[INFO] Embedding + uploading chunks to Weaviate...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [doc.page_content for doc in batch]

        embeddings = embed_with_retry(embedder, texts)
        if not embeddings:
            # Fail fast: if even one batch can't embed, the API is throttled
            # and the rest of this file won't either. Abort the file (state
            # saves on completion, so it will resume on the next run) instead
            # of cycling every batch through expensive backoff for hours.
            logger.warning(
                f"[WARNING] Batch {i // batch_size + 1} failed to embed; aborting file to save quota. Will resume next run."
            )
            break
        try:
            _upload_batch(client, collection_name, batch, embeddings)
            total_inserted += len(embeddings)
            logger.info(
                f"[INFO] Uploaded batch {i // batch_size + 1} with {len(embeddings)} chunks"
            )
            # Delay between batches to avoid rate limiting. 20s keeps us
            # under the per-minute ceiling (~3 batches/min) so we never
            # burst-block and never waste requests on retries.
            if i + batch_size < len(chunks):
                time.sleep(20)
        except QuotaExhaustedError as qe:
            # Daily free-tier budget is gone for today. Stop the whole run
            # now, persist what we have, and report clearly — resuming is
            # just re-running the same command later.
            logger.error(f"[STOP] {qe}")
            raise
        except (RuntimeError, WeaviateBaseError) as exc:
            logger.error(f"[ERROR] Failed to upsert batch {i // batch_size + 1}: {exc}")
            break

    if total_inserted:
        embeds = state[state_key].setdefault("embeddings", {})
        embeds["done"] = True
        embeds["last_embedded"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        embeds["model"] = get_embedding_model()
        embeds["vector_store"] = "weaviate"
        embeds["collection"] = collection_name
        logger.info(
            f"[INFO] Stored {total_inserted}/{len(chunks)} chunks for {filepath}."
        )
    else:
        logger.warning(f"[WARNING] No chunks stored for {filepath}.")

    # Never close the shared client here: it is also used by live /query
    # requests, and closing it mid-flight drops concurrent retrievals.
    return state

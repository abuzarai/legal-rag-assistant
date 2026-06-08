import os
import time
import random
import datetime
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from weaviate.exceptions import WeaviateBaseError
from weaviate.classes.data import DataObject
from src.common.logger import get_logger
from src.common.weaviate_client import get_weaviate_client, ensure_collection
from src.common.config import get_weaviate_collection

load_dotenv(dotenv_path="./.env", override=True)
logger = get_logger(__name__)


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )


def get_embedder():
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT for Vertex AI embeddings.")
    location = os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-005",
        project=project,
        location=location,
        vertexai=True,
    )


def embed_with_retry(embedder, texts, max_retries=5):
    delay = 2
    for attempt in range(max_retries):
        try:
            return embedder.embed_documents(texts)
        except Exception as e:
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                wait = delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"[WARNING] Rate limited. Retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
            else:
                logger.error(f"[ERROR] Unexpected embedding error: {e}")
                return []
    logger.error(f"[ERROR] Failed to embed batch after {max_retries} retries.")
    return []


def _prepare_metadata(chunk: Document, filepath: str, file_id: str | None) -> dict:
    metadata = chunk.metadata or {}
    metadata["source"] = metadata.get("source") or os.path.basename(filepath)
    page = metadata.get("page_label") or metadata.get("page") or "unknown"
    metadata["page"] = page
    if file_id:
        metadata["drive_id"] = file_id
    return metadata


def _upload_batch(
    client, collection_name: str, docs: List[Document], embeddings: List[List[float]]
):
    """Upload a batch of document chunks and embeddings (v4 syntax, fixed)."""
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

        # ✅ use DataObject instead of raw dict
        objects.append(DataObject(properties=properties, vector=vector))

    try:
        collection.data.insert_many(objects)
        logger.info(f"[INFO] ✅ Uploaded {len(objects)} chunks successfully.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to upload batch: {e}")
        raise


def upsert_chunks(docs, state, filepath, file_id=None, batch_size=32):
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

    total_inserted = 0

    logger.info("[INFO] Embedding + uploading chunks to Weaviate...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [doc.page_content for doc in batch]

        embeddings = embed_with_retry(embedder, texts)
        if not embeddings:
            logger.warning(
                f"[WARNING] Skipping batch {i // batch_size + 1}; embedding failed."
            )
            continue

        try:
            _upload_batch(client, collection_name, batch, embeddings)
            total_inserted += len(embeddings)
            logger.info(
                f"[INFO] Uploaded batch {i // batch_size + 1} with {len(embeddings)} chunks"
            )
            # Delay between batches to avoid rate limiting
            if i + batch_size < len(chunks):
                time.sleep(10)
        except (RuntimeError, WeaviateBaseError) as exc:
            logger.error(f"[ERROR] Failed to upsert batch {i // batch_size + 1}: {exc}")
            break

    if total_inserted:
        embeds = state[filepath].setdefault("embeddings", {})
        embeds["done"] = True
        embeds["last_embedded"] = datetime.datetime.utcnow().isoformat()
        embeds["model"] = "text-embedding-005"
        embeds["vector_store"] = "weaviate"
        embeds["collection"] = collection_name
        logger.info(
            f"[INFO] Stored {total_inserted}/{len(chunks)} chunks for {filepath}."
        )
    else:
        logger.warning(f"[WARNING] No chunks stored for {filepath}.")

    # Close connection cleanly
    try:
        client.close()
    except Exception:
        pass

    return state

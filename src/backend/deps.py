# src/backend/deps.py

import os

import numpy as np
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from weaviate.classes.query import Filter as WeaviateFilter

from src.common.config import get_weaviate_collection
from src.common.logger import get_logger
from src.common.weaviate_client import ensure_collection, get_weaviate_client

logger = get_logger(__name__)

_embeddings = None
_collection = None


# ---------------------- Embeddings ----------------------
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        from src.common.config import (
            get_embedding_model,
            get_embedding_output_dims,
            get_gemini_api_key,
        )

        dims = get_embedding_output_dims()
        api_key = get_gemini_api_key()
        if api_key:
            logger.info(
                "Loading Gemini API embeddings (%s, dims=%s)...",
                get_embedding_model(),
                dims,
            )
            _embeddings = GoogleGenerativeAIEmbeddings(
                model=get_embedding_model(),
                google_api_key=api_key,
                output_dimensionality=dims,
            )
            return _embeddings

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "Missing Gemini credentials: set GEMINI_API_KEY (or GOOGLE_CLOUD_PROJECT for Vertex AI)."
            )
        location = os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")
        logger.info("Loading Vertex AI embeddings for Weaviate queries...")
        _embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-005",
            project=project,
            location=location,
            vertexai=True,
            output_dimensionality=dims,
        )
    return _embeddings


# ---------------------- Collection ----------------------
def get_collection():
    global _collection
    if _collection is None:
        client = get_weaviate_client()
        collection_name = get_weaviate_collection()
        ensure_collection(client, collection_name)
        logger.info(f"Using Weaviate collection '{collection_name}' for retrieval...")
        _collection = client.collections.get(collection_name)
    return _collection


# ---------------------- Utility: Reranker ----------------------
def rerank_results(query_vector: list[float], docs: list[Document]) -> list[Document]:
    """
    Re-rank retrieved documents by cosine similarity between
    query and doc vectors. Requires vectors from Weaviate metadata.
    """

    def cosine_similarity(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    scored = []
    for doc in docs:
        vec = doc.metadata.get("_vector")
        if vec is not None:
            score = cosine_similarity(query_vector, vec)
            scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    reranked = [doc for _, doc in scored]
    return reranked or docs


# ---------------------- Core Search ----------------------
def similarity_search(
    query: str,
    k: int = 5,
    category: str | None = None,
    use_hybrid: bool = True,
    rerank: bool = True,
) -> list[Document]:
    """
    Perform semantic/hybrid search in Weaviate with optional reranking and filtering.
    """
    embedder = get_embeddings()
    query_vector = embedder.embed_query(query)
    collection = get_collection()

    # --- optional filter by category ---
    # Weaviate v4 requires a Filter object (v3-era dicts are rejected).
    where_filter = None
    if category:
        where_filter = WeaviateFilter.by_property("category").equal(category)

    # --- Hybrid (BM25 + Vector) search ---
    if use_hybrid:
        logger.info(f"🔍 Running hybrid search for '{query}' (alpha=0.5)...")
        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=0.5,
            limit=max(k * 3, 10),  # get more for reranking
            filters=where_filter,
            return_properties=["content", "source", "page", "drive_id"],
            return_metadata=["distance"],
            include_vector=True,  # vector lets the local reranker work
        )
    else:
        logger.info(f"🔍 Running pure vector search for '{query}'...")
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=max(k * 3, 10),
            filters=where_filter,
            return_properties=["content", "source", "page", "drive_id"],
            return_metadata=["distance"],
            include_vector=True,
        )

    objects = getattr(response, "objects", None) or []
    if not objects:
        logger.warning(f"[Weaviate] No results found for query: {query}")
        return []

    docs: list[Document] = []
    for obj in objects:
        props = obj.properties or {}
        content = props.get("content")
        if not content:
            continue

        metadata = {
            "source": props.get("source"),
            "page": props.get("page"),
            "drive_id": props.get("drive_id"),
            "distance": getattr(getattr(obj, "metadata", None), "distance", None),
            "_vector": (obj.vector or {}).get("default") if isinstance(obj.vector, dict) else obj.vector,
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        docs.append(Document(page_content=content, metadata=metadata))

    # --- rerank locally by embedding similarity ---
    if rerank:
        logger.info("Re-ranking retrieved documents by semantic similarity...")
        docs = rerank_results(query_vector, docs)

    # --- return top k after reranking ---
    top_docs = docs[:k]
    logger.info(f"✅ Retrieved {len(top_docs)} relevant chunks.")
    return top_docs

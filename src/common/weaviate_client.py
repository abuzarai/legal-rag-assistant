import weaviate
from typing import Optional
from urllib.parse import urlparse
from weaviate.exceptions import WeaviateBaseError
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, Property, DataType
from src.common.config import get_weaviate_settings, get_weaviate_collection
from src.common.logger import get_logger

logger = get_logger(__name__)

_client: Optional[weaviate.WeaviateClient] = None


# -------------------- Connection helpers --------------------
def _build_headers(raw: Optional[dict]) -> Optional[dict]:
    """Drop None values from optional headers dict."""
    if not raw:
        return None
    headers = {k: v for k, v in raw.items() if v}
    return headers or None


def _is_cloud_endpoint(url: str) -> bool:
    """Check if the target URL is a managed Weaviate Cloud endpoint."""
    return any(token in url for token in (".weaviate.network", ".weaviate.cloud", ".semi.tech"))


def _parse_host(url: str):
    parsed = urlparse(url if "://" in url else f"http://{url}")
    scheme = parsed.scheme or "http"
    host = parsed.hostname or parsed.path
    if not host:
        raise RuntimeError(f"Unable to parse host from WEAVIATE_URL='{url}'")
    port = parsed.port or (443 if scheme == "https" else 8080)
    secure = scheme == "https"
    return host, port, secure


# -------------------- Client creation --------------------
def _create_client() -> weaviate.WeaviateClient:
    """Create and return a v4 Weaviate client."""
    settings = get_weaviate_settings()
    url = settings.get("url")
    if not url:
        raise RuntimeError("WEAVIATE_URL is not set. Please configure your Weaviate cluster URL.")

    api_key = settings.get("api_key")
    headers = _build_headers(settings.get("headers"))
    auth = AuthApiKey(api_key) if api_key else None

    logger.info(f"Connecting to Weaviate at {url}...")

    if _is_cloud_endpoint(url):
        client = weaviate.connect_to_wcs(
            cluster_url=url,
            auth_credentials=auth,
            headers=headers,
        )
    else:
        http_host, http_port, http_secure = _parse_host(url)
        client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=http_host,
            grpc_port=settings.get("grpc_port", 50051),
            grpc_secure=http_secure,
            headers=headers,
            auth_credentials=auth,
            skip_init_checks=True, 
        )

    logger.info("✅ Connected to Weaviate successfully.")
    return client


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Return a singleton Weaviate client; reconnect if closed."""
    global _client
    if _client is None:
        _client = _create_client()
    elif not _client.is_connected():
        _client.connect()
    return _client


# -------------------- Schema management --------------------
def ensure_collection(client: Optional[weaviate.WeaviateClient] = None, collection_name: Optional[str] = None):
    """
    Ensure the Weaviate collection exists and contains the latest schema.
    Adds 'category' property if missing (safe migration).
    """
    client = client or get_weaviate_client()
    collection_name = collection_name or get_weaviate_collection()

    existing_collections = client.collections.list_all()

    # ✅ If already exists, check for missing 'category' property and add if needed
    if collection_name in existing_collections:
        logger.info(f"Collection '{collection_name}' already exists.")

        try:
            coll = client.collections.get(collection_name)
            props = coll.config.get().properties or []
            prop_names = {p.name for p in props}

            if "category" not in prop_names:
                logger.info("Adding missing 'category' property to schema...")
                coll.config.add_property(Property(name="category", data_type=DataType.TEXT))
                logger.info("✅ 'category' property added successfully.")

        except Exception as e:
            logger.warning(f"⚠️ Could not verify or add 'category' property: {e}")
        return

    # ✅ If not exists, create collection fresh
    logger.info(f"Creating Weaviate collection '{collection_name}'...")

    try:
        client.collections.create(
            name=collection_name,
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.TEXT),
                Property(name="drive_id", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),  
            ],
            vectorizer_config=Configure.Vectorizer.none(),  # BYOV (Gemini embeddings)
            description="Legal document chunks ingested from Google Drive (CPC + case laws)",
        )
        logger.info(f"✅ Collection '{collection_name}' created successfully with 'category' property.")
    except WeaviateBaseError as exc:
        logger.error(f"❌ Failed to create collection '{collection_name}': {exc}")
        raise

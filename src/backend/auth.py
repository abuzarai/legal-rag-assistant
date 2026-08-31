"""Inbound auth for the RAG API (audit #10).

Every route except /health requires a valid `x-internal-key` header matching
INTERNAL_API_KEY. When the env var is set the service enforces it; when it is
unset (bare local dev) the dependency passes — but the compose stack always
sets it, so the deployed service is never open.
"""

import hmac
import os

from fastapi import Header, HTTPException


def require_internal_key(x_internal_key: str = Header(default="")) -> None:
    expected = os.getenv("INTERNAL_API_KEY", "")
    if not expected:
        return  # not configured: permissive fallback for bare local dev
    if not hmac.compare_digest(x_internal_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid internal key")
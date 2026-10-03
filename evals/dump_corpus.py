"""Dump every chunk in the live Weaviate collection to a local JSONL cache.

Read-only against production. Run once:
    python -m evals.dump_corpus
"""

from __future__ import annotations

import json

from evals import config


def main() -> None:
    config.init_credentials()
    config.ensure_import_path()

    from src.common.config import get_weaviate_collection
    from src.common.weaviate_client import get_weaviate_client

    client = get_weaviate_client()
    coll = client.collections.get(get_weaviate_collection())

    config.DATA.mkdir(parents=True, exist_ok=True)
    out = config.DATA / "corpus.jsonl"

    total = 0
    by_cat: dict[str, int] = {}
    with out.open("w") as fh:
        for obj in coll.iterator(include_vector=True):
            props = obj.properties or {}
            vec = obj.vector
            if isinstance(vec, dict):
                vec = vec.get("default")
            rec = {
                "chunk_id": str(obj.uuid),
                "content": props.get("content") or "",
                "source": props.get("source") or "unknown",
                "page": str(props.get("page") or "?"),
                "category": props.get("category") or "unclassified",
                "vector": list(vec) if vec else None,
            }
            fh.write(json.dumps(rec) + "\n")
            total += 1
            by_cat[rec["category"]] = by_cat.get(rec["category"], 0) + 1

    client.close()
    print(f"wrote {total} chunks to {out}")
    print("by category:", by_cat)
    docs = {}
    for rec in config.load_corpus():
        docs[rec["source"]] = docs.get(rec["source"], 0) + 1
    print(f"{len(docs)} source documents")
    for src, n in sorted(docs.items()):
        print(f"  {n:4d}  {src}")


if __name__ == "__main__":
    main()

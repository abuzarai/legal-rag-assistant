"""
This module will:

    - Take the top-N chunks from Weaviate (e.g., 20–30)

    - Send them to Gemini 2.5 Pro

    - Gemini scores each chunk based on relevance to the query

    - Return the top-k reranked chunks"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from typing import List
import os

project = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")

# Gemini client for re-ranking
rerank_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project=project,
    location=location,
    vertexai=True,
    temperature=0.0,
)


def rerank_chunks(query: str, chunks: List[Document], top_k: int = 5) -> List[Document]:
    """
    Use Gemini to re-score and reorder retrieved chunks based on relevance.
    Input: top-N chunks from Weaviate
    Output: best top_k after re-ranking
    """

    if not chunks:
        return []

    # Build structured ranking prompt
    ranking_list = "\n".join(
        [f"[{i}] {chunks[i].page_content[:500]}" for i in range(len(chunks))]
    )

    prompt = f"""
You are ranking legal text chunks based on how useful they are
for answering this question:

Question: "{query}"

Below are {len(chunks)} chunks retrieved from a legal database.
Rank them **ONLY** by relevance to the question. Ignore style, length, flowery language.

Return output *strictly* in this JSON format:
[
  {{"index": 3, "score": 0.92}},
  {{"index": 0, "score": 0.85}},
  ...
]

Chunks:
{ranking_list}
"""

    response = rerank_llm.invoke(prompt)

    import json

    try:
        ranking = json.loads(response.content)
    except Exception:
        # fallback: return the original order
        return chunks[:top_k]

    # Sort by score
    ranking = sorted(ranking, key=lambda x: x["score"], reverse=True)

    best_docs = []
    for item in ranking[:top_k]:
        idx = item["index"]
        if 0 <= idx < len(chunks):
            best_docs.append(chunks[idx])

    return best_docs

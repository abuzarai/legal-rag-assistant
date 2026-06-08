import os
import re
from typing import Iterable
from pathlib import PurePosixPath
from langchain_google_genai import ChatGoogleGenerativeAI
from src.backend.deps import similarity_search
from src.common.logger import get_logger

logger = get_logger(__name__)

MAX_QUERY_CHARS = 2000
MIN_CONTEXT_CHARS = 120
MAX_SOCIAL_REPLY_CHARS = 280


# -------------------------------------------------------
# Gemini Initialization via Vertex AI (ADC)
# -------------------------------------------------------
def get_llm():
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT for Vertex AI Gemini.")
    location = os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")

    logger.info("[GEMINI] Using Vertex AI authentication")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=project,
        location=location,
        vertexai=True,
        temperature=0.2,
    )


# Cached instance (lazy load)
_llm_instance = None


def llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = get_llm()
    return _llm_instance


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------


def _pretty_title(path: str, page: str | int | None) -> str:
    filename = PurePosixPath(path).name if path else "Unknown source"
    filename = filename.replace("-", " ").replace("_", " ")
    page_display = f"p.{page}" if page else "page ?"
    return f"{filename} — {page_display}"


def format_sources(docs):
    seen = set()
    formatted = []
    for doc in docs:
        meta = doc.metadata or {}
        source_path = meta.get("source", "unknown.pdf")
        page = meta.get("page_label") or meta.get("page") or "?"
        file_id = meta.get("drive_id")

        title = _pretty_title(source_path, page)
        link = (
            f"https://drive.google.com/file/d/{file_id}/view#page={page}"
            if file_id
            else None
        )

        key = (file_id, page, title)
        if key not in seen:
            seen.add(key)
            formatted.append({"title": title, "link": link})
    return formatted


def _compile_section_patterns(section: str) -> list[re.Pattern]:
    aliases = {
        "Detailed Analysis": ["Detailed Analysis", "Analysis", "Reasoning"],
        "Summary": ["Summary", "Answer", "Short Answer"],
        "Citations": ["Citations", "References", "Authorities"],
    }
    labels = aliases.get(section, [section])
    escaped = "|".join(re.escape(item) for item in labels)

    return [
        re.compile(
            rf"^\s*\*\*\s*(?:{escaped})\s*:?\s*\*\*\s*(.*?)\s*(?=^\s*\*\*\s*[A-Za-z][^\n]*\*\*\s*:?\s*$|\Z)",
            flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            rf"^\s*(?:{escaped})\s*:\s*(.*?)\s*(?=^\s*[A-Za-z][^\n]*:\s*$|\Z)",
            flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
        ),
    ]


def extract_section(text: str, section: str) -> str:
    for pattern in _compile_section_patterns(section):
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            if value:
                return value.strip("-").strip()
    return ""


def extract_citations(text: str) -> list[str]:
    citations_section = extract_section(text, "Citations")
    if not citations_section:
        return []
    lines = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        for line in citations_section.splitlines()
    ]
    deduped = []
    seen = set()
    for line in lines:
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return deduped[:8]


def normalize_query(query: str) -> str:
    normalized = re.sub(r"\s+", " ", (query or "").strip())
    return normalized[:MAX_QUERY_CHARS]


_CASUAL_PATTERNS = [
    re.compile(
        r"^\s*(hi|hello|hey|salam|assalam\s*o\s*alaikum|aoa)\s*[!.?]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(thanks?|thank\s+you|ok|okay|great|cool)\s*[!.?]*\s*$", re.IGNORECASE
    ),
    re.compile(
        r"^\s*(good\s*(morning|afternoon|evening)|bye|goodbye)\s*[!.?]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(yo|sup|hey there|hmm|hmmm)\s*[!.?]*\s*$", re.IGNORECASE),
]

_NON_LEGAL_PATTERNS = [
    re.compile(
        r"\b(weather|temperature|rain|sports?|movie|song|recipe|restaurant)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(code|python|javascript|react|bug fix|docker|sql query)\b", re.IGNORECASE
    ),
]

_UNCERTAIN_PATTERNS = [
    re.compile(
        r"^\s*(help|need help|please help|guide me|advice)\s*[!.?]*\s*$", re.IGNORECASE
    ),
    re.compile(r"^\s*(what can you do|how can you help)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(
        r"^\s*(property issue|family issue|court issue|legal issue)\s*[!.?]*\s*$",
        re.IGNORECASE,
    ),
]

_STATUTE_REF_PATTERN = re.compile(
    r"\b(order\s*[ivx\d]+\s*rule\s*\d+[a-z]?|section\s*\d+[a-z]?|article\s*\d+)\b",
    re.IGNORECASE,
)

_LEGAL_KEYWORDS = {
    "legal",
    "law",
    "lawyer",
    "court",
    "judge",
    "appeal",
    "plaint",
    "plaintiff",
    "defendant",
    "fir",
    "bail",
    "petition",
    "injunction",
    "notice",
    "contract",
    "agreement",
    "divorce",
    "khula",
    "custody",
    "nikah",
    "inheritance",
    "property",
    "land",
    "tenant",
    "rent",
    "eviction",
    "cpc",
    "order",
    "rule",
    "section",
    "writ",
    "civil",
    "criminal",
    "suit",
}


def _has_non_legal_signal(query: str) -> bool:
    return any(pattern.search(query) for pattern in _NON_LEGAL_PATTERNS)


def _has_legal_signal(query: str, tokens: Iterable[str]) -> bool:
    if _STATUTE_REF_PATTERN.search(query):
        return True
    return any(token in _LEGAL_KEYWORDS for token in tokens)


def detect_mode(query: str) -> str:
    normalized = normalize_query(query)
    if not normalized:
        return "social"

    for pattern in _CASUAL_PATTERNS:
        if pattern.match(normalized):
            return "social"

    for pattern in _UNCERTAIN_PATTERNS:
        if pattern.match(normalized):
            return "uncertain"

    tokens = re.findall(r"[a-zA-Z]+", normalized.lower())
    has_legal = _has_legal_signal(normalized, tokens)
    has_non_legal = _has_non_legal_signal(normalized)

    if has_non_legal and not has_legal:
        return "social"

    if len(tokens) <= 3 and not has_legal:
        return "uncertain"

    return "legal"


def casual_response(query: str) -> str:
    normalized = normalize_query(query).lower()
    if re.match(r"^\s*(thanks?|thank\s+you)\b", normalized):
        return "You're welcome. If you share your legal issue, I can guide you with the next practical steps."
    if re.match(r"^\s*(bye|goodbye)\b", normalized):
        return "Take care. If you need legal help later, just message me anytime."
    return (
        "Hi! I am Insafdaar Assistant. "
        "I can help with FIRs, property disputes, family matters, contracts, and court procedure. "
        "What legal issue would you like help with?"
    )


def uncertain_response() -> str:
    return (
        "I can help with that. Please share key facts: what happened, when it happened, and what outcome you want. "
        "Then I will give a focused legal response."
    )


def sanitize_social_text(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(
        r"\*\*\s*(Sources?|Citations?)\s*:?\s*\*\*.*$", "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"\b(Sources?|Citations?)\s*:\s*.*$", "", cleaned, flags=re.IGNORECASE
    )
    cleaned = cleaned.strip("- ").strip()
    if len(cleaned) > MAX_SOCIAL_REPLY_CHARS:
        cleaned = cleaned[:MAX_SOCIAL_REPLY_CHARS].rstrip()
    return cleaned


def generate_social_reply(query: str) -> str:
    prompt = f"""
You are Insafdaar Assistant for a legal platform in Pakistan.

The user sent social or small-talk text. Reply naturally and warmly in English.

Rules:
- Keep it to 1-2 short sentences.
- Do not provide legal advice in this mode.
- Do not include citations, sources, markdown headings, or bullet points.
- Invite the user to share their legal issue in one simple line.

User message:
{query}
"""

    try:
        response = llm().invoke(prompt)
        raw_content = response.content
        if isinstance(raw_content, list):
            text = " ".join(str(item) for item in raw_content)
        else:
            text = str(raw_content or "")
        sanitized = sanitize_social_text(text)
        if sanitized:
            return sanitized
    except Exception as err:
        logger.warning("[GEMINI] Social reply generation failed: %s", err)

    return casual_response(query)


def non_legal_redirect_response() -> str:
    return (
        "I am focused on legal guidance for Pakistani law. "
        "If you share a legal issue, I can help with practical next steps and relevant legal references."
    )


def is_retrieval_weak(docs) -> bool:
    if not docs:
        return True

    strong_docs = [
        d for d in docs if len((d.page_content or "").strip()) >= MIN_CONTEXT_CHARS
    ]
    if not strong_docs:
        return True

    distances = [
        float(doc.metadata.get("distance"))
        for doc in docs
        if doc.metadata and doc.metadata.get("distance") is not None
    ]

    if len(distances) == 1:
        return distances[0] > 0.62

    if len(distances) < 2:
        return False

    avg_distance = sum(distances) / len(distances)
    return avg_distance > 0.52 or min(distances) > 0.58


def _fallback_summary_from_answer(answer: str) -> str:
    clean = re.sub(r"\s+", " ", answer).strip()
    if not clean:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    return " ".join(sentences[:2])[:420].strip()


def _join_context(docs) -> str:
    chunks = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source_path = meta.get("source", "unknown.pdf")
        page = meta.get("page_label") or meta.get("page") or "?"
        header = (
            f"[Source {idx} | {PurePosixPath(str(source_path)).name} | page {page}]"
        )
        chunks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(chunks)


# -------------------------------------------------------
# Core RAG pipeline
# -------------------------------------------------------


def run_rag(query: str, k: int = 5) -> dict:
    normalized_query = normalize_query(query)
    logger.info(f"🔍 Searching Weaviate for: {normalized_query}")

    mode = detect_mode(normalized_query)
    if mode == "social":
        if _has_non_legal_signal(normalized_query):
            answer = non_legal_redirect_response()
        else:
            answer = generate_social_reply(normalized_query)
        return {
            "query": normalized_query,
            "mode": "social",
            "answer": answer,
            "summary": "",
            "analysis": "",
            "citations": [],
            "sources": [],
        }

    if mode == "uncertain":
        return {
            "query": normalized_query,
            "mode": "uncertain",
            "answer": uncertain_response(),
            "summary": "",
            "analysis": "",
            "citations": [],
            "sources": [],
        }

    docs = similarity_search(normalized_query, k=k)

    if is_retrieval_weak(docs):
        return {
            "query": normalized_query,
            "mode": "legal",
            "answer": (
                "I need a bit more detail to answer this accurately. "
                "Please share key facts (what happened, when, where, and what relief you want), "
                "and I will provide a focused legal response."
            ),
            "summary": "",
            "analysis": "",
            "citations": [],
            "sources": [],
        }

    context = _join_context(docs)

    prompt = f"""
You are Insafdaar Assistant, a legal research assistant for Pakistani law.

Task:
- Use ONLY the retrieved context below. Do not add authorities or facts not in context.
- Give a rigorous legal response in English, but keep it practical and readable.
- Resolve ambiguity by stating assumptions explicitly.
- If context conflicts, explain both views and why one is stronger.
- If context is insufficient, ask one targeted follow-up question instead of guessing.

Output format:
**Summary:** <2-3 sentence answer>

**Detailed Analysis:**
1. Issue
2. Rule
3. Application
4. Practical Next Step

**Citations:**
- <CPC Order/Rule, section, or case title from context>
- <second reference if available>

Context:
{context}

Question:
{normalized_query}
"""

    logger.info("[GEMINI] Generating legal answer...")
    try:
        response = llm().invoke(prompt)
    except Exception as err:
        logger.exception("[GEMINI] Generation failed: %s", err)
        return {
            "query": normalized_query,
            "mode": "legal",
            "answer": (
                "I could not generate a complete legal analysis right now. "
                "Please retry in a moment."
            ),
            "summary": "",
            "analysis": "",
            "citations": [],
            "sources": format_sources(docs),
        }

    raw_content = response.content
    if isinstance(raw_content, list):
        raw = "\n".join(str(item) for item in raw_content).strip()
    else:
        raw = str(raw_content or "").strip()

    summary = extract_section(raw, "Summary")
    analysis = extract_section(raw, "Detailed Analysis")
    citations = extract_citations(raw)

    if not summary:
        summary = _fallback_summary_from_answer(raw)

    if not analysis and summary:
        analysis = "The retrieved context supports the summary above."

    return {
        "query": normalized_query,
        "mode": "legal",
        "answer": raw,
        "summary": summary,
        "analysis": analysis,
        "citations": citations,
        "sources": format_sources(docs),
    }

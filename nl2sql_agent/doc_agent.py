"""
Document Sub-Agent (very simple RAG).

  Index (once):  PDF pages -> text -> overlapping chunks -> text-embedding-3-large vectors (cached)
  Query:         question -> embedding -> cosine similarity -> top-k chunks -> LLM answer with [p. N] citations
"""

import json
import math
import time
from pypdf import PdfReader
import llm
from config import PDF_PATH, PDF_PAGE_START, PDF_PAGE_END, CACHE_DIR, CURRENT_PROMPTS
from tool_runner import answer_with_tools
from tracelog import Trace

AGENT = "Document Agent"
INDEX_FILE = CACHE_DIR / "doc_index.json"
CHUNK_SIZE = 900        # characters
CHUNK_OVERLAP = 150
TOP_K = 4

CHUNKS = []             # [{"id", "page", "text", "vector"}]


def _chunk_page(text: str):
    text = " ".join(text.split())
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, max(len(text) - CHUNK_OVERLAP, 1), step)]


def build_index() -> None:
    """Load chunks + vectors from cache, or build them from the PDF."""
    global CHUNKS
    if INDEX_FILE.exists():
        CHUNKS = json.loads(INDEX_FILE.read_text())
        return

    reader = PdfReader(str(PDF_PATH))
    chunks = []
    for pdf_index in range(PDF_PAGE_START, PDF_PAGE_END):
        book_page = pdf_index - PDF_PAGE_START + 1
        for piece in _chunk_page(reader.pages[pdf_index].extract_text() or ""):
            chunks.append({"id": f"p{book_page}-c{len(chunks)}", "page": book_page, "text": piece})

    vectors = []
    for i in range(0, len(chunks), 64):                 # embed in batches
        batch, _ = llm.embed([c["text"] for c in chunks[i:i + 64]])
        vectors += batch
    for c, v in zip(chunks, vectors):
        c["vector"] = v

    INDEX_FILE.write_text(json.dumps(chunks))
    CHUNKS = chunks


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)


def retrieve(question: str, trace: Trace) -> list:
    t0 = time.time()
    [query_vec], cost = llm.embed([question])
    scored = sorted(CHUNKS, key=lambda c: cosine(query_vec, c["vector"]), reverse=True)[:TOP_K]
    hits = [{"id": c["id"], "page": c["page"], "text": c["text"],
             "score": round(cosine(query_vec, c["vector"]), 4)} for c in scored]
    trace.add(AGENT, "retrieval", f"Dense retrieval (top {TOP_K})", input=question,
              output=[{"id": h["id"], "page": h["page"], "score": h["score"]} for h in hits], cost=cost, started=t0)
    return hits


def run(question: str, trace: Trace) -> dict:
    hits = retrieve(question, trace)
    context = "\n\n".join(f"[p. {h['page']}] {h['text']}" for h in hits)
    system = CURRENT_PROMPTS["doc_answer_prompt"].format(context=context)
    answer = answer_with_tools(AGENT, system, question, trace)
    return {"answer": answer, "chunks": hits}

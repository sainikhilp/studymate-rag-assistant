"""Retrieve relevant chunks and generate grounded answers via OpenAI chat."""

import pickle

import faiss
import numpy as np
from openai import OpenAI

from src.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    CHAT_MODEL,
    TOP_K,
    SMALL_DOC_CHUNK_THRESHOLD,
    TEMPERATURE,
    OPENAI_API_KEY,
)
from src.embedder import embed_query

_client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = """You are a helpful teaching assistant. Answer questions using ONLY the context provided below.
When you use information from the context, cite it inline as [source_file p.N] where source_file is the document name and N is the page number.
If the provided context does not contain enough information to answer the question, respond with: "I don't know based on the provided materials."
Do not use outside knowledge. Be concise and precise."""


def load_index() -> tuple[faiss.Index, list[dict]]:
    """Load the persisted FAISS index and metadata from disk.

    Returns:
        Tuple of ``(faiss_index, metadata_list)``.

    Raises:
        FileNotFoundError: If the index files do not exist. Run build_index first.
    """
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Index not found at {FAISS_INDEX_PATH}. "
            "Run `python -m src.build_index` to build the index first."
        )
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata: list[dict] = pickle.load(f)
    return index, metadata


def retrieve(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    k: int = TOP_K,
    small_doc_threshold: int = SMALL_DOC_CHUNK_THRESHOLD,
) -> list[dict]:
    """Return the most relevant chunks for ``query`` with small-doc guarantees.

    Strategy:
    - Small documents (chunk count <= small_doc_threshold, e.g. a 12-chunk
      syllabus): ALL chunks are always included. This prevents key facts buried
      in low-ranked chunks (due to query-phrasing variance) from being missed.
    - Large documents (e.g. the textbook): top-k globally ranked chunks only.

    The final list is sorted by score so the LLM sees the most relevant context
    first, regardless of source.

    Args:
        query: User's question string.
        index: Loaded FAISS index.
        metadata: Aligned metadata list (same row order as the index).
        k: Number of chunks to include from large documents.
        small_doc_threshold: Sources with <= this many chunks are fully included.

    Returns:
        List of metadata dicts sorted by score descending, each with a
        ``"score"`` field.
    """
    query_vec = embed_query(query).reshape(1, -1)

    # Count chunks per source to identify small vs large documents
    source_chunk_counts: dict[str, int] = {}
    for m in metadata:
        source_chunk_counts[m["source_file"]] = source_chunk_counts.get(m["source_file"], 0) + 1

    small_sources = {s for s, cnt in source_chunk_counts.items() if cnt <= small_doc_threshold}
    large_sources = {s for s in source_chunk_counts if s not in small_sources}

    # Collect all small-doc chunks (scored via FAISS over the full index)
    # Search enough candidates to cover all small-doc chunks + k large-doc chunks
    total_small = sum(source_chunk_counts[s] for s in small_sources)
    search_k = min(total_small + k + 10, index.ntotal)
    scores_arr, indices_arr = index.search(query_vec, search_k)

    seen_ids: set[str] = set()
    result: list[dict] = []
    large_count = 0

    for score, idx in zip(scores_arr[0], indices_arr[0]):
        if idx == -1:
            continue
        entry = dict(metadata[idx])
        entry["score"] = float(score)
        src = entry["source_file"]

        if src in small_sources:
            # Always include; will be de-duped after full scan below
            pass  # collected in second pass
        else:
            # Large doc: include only up to k
            if large_count < k and entry["id"] not in seen_ids:
                result.append(entry)
                seen_ids.add(entry["id"])
                large_count += 1

    # Add every chunk from small documents (all scored)
    # We need scores for all small-doc chunks — search the full index
    scores_full, indices_full = index.search(query_vec, index.ntotal)
    for score, idx in zip(scores_full[0], indices_full[0]):
        if idx == -1:
            continue
        entry = dict(metadata[idx])
        entry["score"] = float(score)
        if entry["source_file"] in small_sources and entry["id"] not in seen_ids:
            result.append(entry)
            seen_ids.add(entry["id"])

    result.sort(key=lambda c: c["score"], reverse=True)
    return result


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a labelled context block."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk['source_file']} | Page: {chunk['page_number']}\n"
            f"{chunk['document']}"
        )
    return "\n\n---\n\n".join(parts)


def answer(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    history: list[dict] | None = None,
    k: int = TOP_K,
) -> dict:
    """Retrieve relevant chunks and return a grounded answer.

    Args:
        query: The user's question.
        index: Loaded FAISS index.
        metadata: Aligned metadata list.
        history: Optional list of prior ``{role, content}`` message dicts
                 for multi-turn conversation support.
        k: Number of chunks to retrieve.

    Returns:
        Dict with keys:
            - ``"answer"`` (str): The model's response.
            - ``"sources"`` (list[dict]): Retrieved chunks with scores.
    """
    sources = retrieve(query, index, metadata, k=k)
    context_block = _build_context_block(sources)

    user_message = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}"
    )

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )
    answer_text = response.choices[0].message.content or ""

    return {"answer": answer_text, "sources": sources}

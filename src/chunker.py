"""Recursive character chunking with tiktoken-based token counting."""

import re
import tiktoken
from pathlib import Path
from src.config import (
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    MIN_CHUNK_TOKENS,
    TIKTOKEN_ENCODING,
)

_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " "]

_enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)


def _token_count(text: str) -> int:
    """Return the number of tokens in ``text`` using cl100k_base."""
    return len(_enc.encode(text))


def _split_recursive(text: str, separators: list[str]) -> list[str]:
    """Split ``text`` recursively until all pieces are under CHUNK_SIZE_TOKENS."""
    if _token_count(text) <= CHUNK_SIZE_TOKENS:
        return [text]

    sep = separators[0] if separators else " "
    parts = text.split(sep)

    # Re-join parts into groups that fit within the chunk size
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for part in parts:
        part_tokens = _token_count(part)
        join_tokens = _token_count(sep) if current_parts else 0

        if current_tokens + join_tokens + part_tokens > CHUNK_SIZE_TOKENS and current_parts:
            chunk_text = sep.join(current_parts)
            if _token_count(chunk_text) > CHUNK_SIZE_TOKENS and len(separators) > 1:
                chunks.extend(_split_recursive(chunk_text, separators[1:]))
            else:
                chunks.append(chunk_text)
            current_parts = []
            current_tokens = 0

        current_parts.append(part)
        current_tokens += join_tokens + part_tokens

    if current_parts:
        chunk_text = sep.join(current_parts)
        if _token_count(chunk_text) > CHUNK_SIZE_TOKENS and len(separators) > 1:
            chunks.extend(_split_recursive(chunk_text, separators[1:]))
        else:
            chunks.append(chunk_text)

    return chunks


def _apply_overlap(chunks: list[str]) -> list[str]:
    """Re-join consecutive chunks with a token overlap window.

    Each chunk (except the first) is prepended with the trailing
    CHUNK_OVERLAP_TOKENS tokens from the previous chunk.
    """
    if len(chunks) <= 1:
        return chunks

    result: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = _enc.encode(chunks[i - 1])
        overlap_tokens = prev_tokens[-CHUNK_OVERLAP_TOKENS:] if len(prev_tokens) > CHUNK_OVERLAP_TOKENS else prev_tokens
        overlap_text = _enc.decode(overlap_tokens)
        merged = overlap_text + " " + chunks[i]
        result.append(merged)
    return result


def chunk_pages(
    pages: list[dict],
    source_file: str,
) -> list[dict]:
    """Convert a list of pages into overlapping text chunks with metadata.

    Args:
        pages: Output of ``pdf_loader.load_pdf`` — list of dicts with
               ``page_number`` and ``text``.
        source_file: Filename of the source PDF (e.g. ``"textbook.pdf"``).

    Returns:
        List of chunk dicts ready to be embedded. ``total_chunks`` is set
        after all chunks are collected so every dict gets the final count.
    """
    raw_chunks: list[tuple[str, int]] = []  # (text, starting_page)

    for page in pages:
        page_text = re.sub(r"[ \t]+", " ", page["text"]).strip()
        if not page_text:
            continue
        pieces = _split_recursive(page_text, _SPLIT_SEPARATORS)
        for piece in pieces:
            raw_chunks.append((piece.strip(), page["page_number"]))

    overlapped = _apply_overlap([t for t, _ in raw_chunks])
    pages_for_chunks = [p for _, p in raw_chunks]

    source_stem = Path(source_file).stem
    metadata_chunks: list[dict] = []
    chunk_index = 0

    for text, page_number in zip(overlapped, pages_for_chunks):
        tokens = _token_count(text)
        if tokens < MIN_CHUNK_TOKENS:
            continue
        metadata_chunks.append(
            {
                "id": f"{source_stem}_chunk_{chunk_index:05d}",
                "document": text,
                "source_file": source_file,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "total_chunks": 0,  # filled below
                "embedding_model": "",  # filled by embedder
                "token_count": tokens,
            }
        )
        chunk_index += 1

    total = len(metadata_chunks)
    for chunk in metadata_chunks:
        chunk["total_chunks"] = total

    return metadata_chunks

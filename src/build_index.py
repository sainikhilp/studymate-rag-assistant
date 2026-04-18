"""Orchestrate: load PDFs -> chunk -> embed -> save FAISS index + metadata."""

import argparse
import pickle
import sys
import os
from collections import defaultdict

# Fix Windows console encoding so Unicode prints don't crash
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[attr-defined]

import faiss
import numpy as np

from src.config import (
    DATA_DIR,
    INDEX_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_DIM,
    EMBED_COST_PER_MILLION_TOKENS,
)
from src.pdf_loader import load_all_pdfs
from src.chunker import chunk_pages
from src.embedder import embed_chunks


def _index_exists() -> bool:
    return FAISS_INDEX_PATH.exists() and METADATA_PATH.exists()


def build(rebuild: bool = False) -> None:
    """Build and persist the FAISS index.

    Args:
        rebuild: When ``True``, regenerate even if an index already exists.
    """
    if _index_exists() and not rebuild:
        print("Index exists. Pass --rebuild to regenerate.")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading PDFs from {DATA_DIR} ...")
    pdf_docs = load_all_pdfs(DATA_DIR)

    all_chunks: list[dict] = []
    chunks_per_file: dict[str, int] = {}

    for pdf_path, pages in pdf_docs:
        print(f"  Chunking {pdf_path.name} ({len(pages)} pages) ...")
        chunks = chunk_pages(pages, pdf_path.name)
        chunks_per_file[pdf_path.name] = len(chunks)
        all_chunks.extend(chunks)
        print(f"    -> {len(chunks)} chunks")

    if not all_chunks:
        print("No chunks produced. Check that PDFs contain extractable text.")
        sys.exit(1)

    print(f"\nEmbedding {len(all_chunks)} chunks ...")
    vectors: np.ndarray = embed_chunks(all_chunks)

    print("Building FAISS index ...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    # Summary
    total_tokens = sum(c["token_count"] for c in all_chunks)
    estimated_cost = (total_tokens / 1_000_000) * EMBED_COST_PER_MILLION_TOKENS

    print("\n-- Build summary -------------------------------------------------")
    for fname, count in chunks_per_file.items():
        file_tokens = sum(c["token_count"] for c in all_chunks if c["source_file"] == fname)
        print(f"  {fname}: {count} chunks, {file_tokens:,} tokens")
    print(f"  Total chunks : {len(all_chunks):,}")
    print(f"  Total tokens : {total_tokens:,}")
    print(f"  Estimated embedding cost: ${estimated_cost:.4f}")
    print(f"  Index saved  : {FAISS_INDEX_PATH}")
    print(f"  Metadata saved: {METADATA_PATH}")
    print("-----------------------------------------------------------------")


def main() -> None:
    """CLI entry point for building the FAISS index."""
    parser = argparse.ArgumentParser(description="Build the RAG FAISS index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force regeneration even if an index already exists.",
    )
    args = parser.parse_args()
    build(rebuild=args.rebuild)


if __name__ == "__main__":
    main()

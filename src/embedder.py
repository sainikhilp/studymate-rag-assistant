"""Batched OpenAI embedding calls with exponential-backoff retry."""

import time
import numpy as np
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm

from src.config import (
    EMBEDDING_MODEL,
    EMBED_BATCH_SIZE,
    OPENAI_API_KEY,
)

_client = OpenAI(api_key=OPENAI_API_KEY)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row so inner product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _embed_batch(texts: list[str], retries: int = 5) -> np.ndarray:
    """Embed a single batch with exponential backoff on transient errors.

    Args:
        texts: List of strings to embed (len <= EMBED_BATCH_SIZE).
        retries: Maximum number of retry attempts.

    Returns:
        Float32 numpy array of shape ``(len(texts), embedding_dim)``.
    """
    delay = 1.0
    for attempt in range(retries):
        try:
            response = _client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            vectors = np.array(
                [item.embedding for item in response.data], dtype=np.float32
            )
            return vectors
        except RateLimitError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
        except APIError as exc:
            if attempt == retries - 1:
                raise
            # Retry on 5xx errors only
            if hasattr(exc, "status_code") and exc.status_code and exc.status_code < 500:
                raise
            time.sleep(delay)
            delay *= 2
    # Unreachable, but satisfies type checkers
    raise RuntimeError("Embedding failed after all retries.")


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """Embed all chunks in batches and return a normalized matrix.

    Args:
        chunks: List of chunk metadata dicts produced by ``chunker.chunk_pages``.
                Each dict must have a ``"document"`` key.

    Returns:
        Normalized float32 array of shape ``(len(chunks), embedding_dim)``.
        Also mutates each chunk dict to set ``"embedding_model"``.
    """
    texts = [c["document"] for c in chunks]
    all_vectors: list[np.ndarray] = []

    for start in tqdm(
        range(0, len(texts), EMBED_BATCH_SIZE),
        desc="Embedding batches",
        unit="batch",
    ):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vectors = _embed_batch(batch)
        all_vectors.append(vectors)

    matrix = np.vstack(all_vectors).astype(np.float32)
    matrix = _normalize(matrix)

    for chunk in chunks:
        chunk["embedding_model"] = EMBEDDING_MODEL

    return matrix


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string and return a normalized 1-D vector.

    Args:
        query: The user's question or search string.

    Returns:
        Normalized float32 array of shape ``(embedding_dim,)``.
    """
    vector = _embed_batch([query])[0]
    vector = vector / max(np.linalg.norm(vector), 1e-9)
    return vector.astype(np.float32)

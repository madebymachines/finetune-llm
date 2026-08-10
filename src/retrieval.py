from __future__ import annotations

import numpy as np

_MODEL_CACHE: dict[str, object] = {}


def get_embedder(model_name: str):
    """Cached sentence-transformers model, loaded on CPU so it never competes
    with the 4-bit-loaded LLM for GPU memory. Indexing a catalog/document is
    an infrequent, small-batch operation, so CPU is fast enough."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _MODEL_CACHE[model_name]


def embed_passages(embedder, texts: list[str]) -> np.ndarray:
    """"passage: " prefix is required by e5-family models for correct
    retrieval quality (asymmetric query/passage encoding)."""
    return embedder.encode([f"passage: {t}" for t in texts], normalize_embeddings=True, convert_to_numpy=True)


def embed_query(embedder, text: str) -> np.ndarray:
    return embedder.encode([f"query: {text}"], normalize_embeddings=True, convert_to_numpy=True)[0]


def retrieve_top_chunks(
    query_vec: np.ndarray,
    chunk_vecs: np.ndarray,
    chunks: list[dict],
    top_k: int = 5,
    max_chars: int = 2000,
) -> list[dict]:
    """Brute-force cosine similarity (vectors are normalized, so dot product
    == cosine similarity) — exact, not approximate. Fine at knowledge-base
    scale (hundreds-thousands of chunks); FAISS/Chroma only pay off at a
    scale this app isn't targeting. `chunks[i]` is `{"text", "source"}`,
    aligned by row index with `chunk_vecs`. Stops adding chunks once
    `max_chars` is reached, so the retrieved context stays bounded."""
    scores = chunk_vecs @ query_vec
    order = scores.argsort()[::-1]
    selected: list[dict] = []
    total = 0
    for i in order[:top_k]:
        chunk = chunks[int(i)]
        if selected and total + len(chunk["text"]) > max_chars:
            break
        selected.append({**chunk, "score": float(scores[i])})
        total += len(chunk["text"])
    return selected

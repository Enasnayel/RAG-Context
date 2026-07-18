"""
embed_index.py — frozen embedder + exhaustive FAISS index.

- jina-embeddings-v2 (frozen), auto-uses CUDA when available.
- Embeddings cached to disk per (corpus, cell): embed once, ever.
  A rerun, crash recovery, or metric recomputation never re-embeds.
- FAISS IndexFlatIP on normalized vectors = exact cosine retrieval,
  deterministic, no ANN approximation confound.
"""

from pathlib import Path

import numpy as np

from config import EMBED_MODEL, CACHE_DIR

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    return _model


def _normalize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embed_corpus(serialized_texts, cache_key: str, batch_size: int = 64):
    """Embed the serialized records for one cell, with disk cache."""
    path = Path(CACHE_DIR) / f"emb_{cache_key}.npy"
    if path.exists():
        vecs = np.load(path)
        if len(vecs) == len(serialized_texts):
            return vecs
    path.parent.mkdir(parents=True, exist_ok=True)
    vecs = get_model().encode(serialized_texts, batch_size=batch_size,
                              show_progress_bar=True, convert_to_numpy=True)
    vecs = _normalize(vecs).astype(np.float32)
    np.save(path, vecs)
    return vecs


def build_index(vecs):
    import faiss
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def embed_query(question: str):
    v = get_model().encode([question], convert_to_numpy=True)
    return _normalize(v).astype(np.float32)

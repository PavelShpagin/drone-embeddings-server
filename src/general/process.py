"""
Low-level matching primitives used by higher-level pipelines.
"""

from typing import Optional, Dict, Any
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def find_closest_patch(query_embedding: np.ndarray, session_data, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Find the closest patch in session data to the query embedding."""
    if not session_data or not session_data.patches:
        return None

    best_similarity = -1.0
    best_patch = None

    for patch in session_data.patches:
        # Extract patch embedding robustly from dict or legacy field
        patch_emb = None
        if hasattr(patch, "embedding_data") and isinstance(patch.embedding_data, dict):
            patch_emb = patch.embedding_data.get("embedding")
        if patch_emb is None:
            patch_emb = getattr(patch, "embedding", None)
        if patch_emb is None:
            continue

        # Ensure both are numpy arrays of float32
        patch_emb = np.asarray(patch_emb, dtype=np.float32)
        qe = np.asarray(query_embedding, dtype=np.float32)

        # Some methods may pass dict from embedder; accept {"embedding": ...}
        if isinstance(query_embedding, dict):
            qe_raw = query_embedding.get("embedding")
            if qe_raw is None:
                continue
            qe = np.asarray(qe_raw, dtype=np.float32)

        similarity = cosine_similarity(qe, patch_emb)
        if similarity > best_similarity:
            best_similarity = similarity
            best_patch = patch

    if best_patch is None:
        return None

    return {
        "lat": best_patch.lat,
        "lng": best_patch.lng,
        "similarity": float(best_similarity),
        "patch_coords": best_patch.patch_coords,
        "confidence": "high" if best_similarity > 0.8 else "medium" if best_similarity > 0.6 else "low",
    }



"""Approximate vector index with FAISS optional fallback."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


class VectorIndex:
    """ANN-like index for item embeddings."""

    def __init__(self) -> None:
        self.item_ids: list[int] = []
        self.vectors: np.ndarray | None = None
        self.faiss_index = None
        self.nn_index: NearestNeighbors | None = None

    def fit(self, item_ids: list[int], vectors: np.ndarray) -> None:
        if len(item_ids) == 0:
            self.item_ids = []
            self.vectors = None
            self.faiss_index = None
            self.nn_index = None
            return

        vec = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vec = vec / norms

        self.item_ids = [int(x) for x in item_ids]
        self.vectors = vec

        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatIP(vec.shape[1])  # cosine via normalized dot product
            self.faiss_index.add(vec)
            self.nn_index = None
            return

        self.nn_index = NearestNeighbors(metric="cosine", algorithm="auto")
        self.nn_index.fit(vec)
        self.faiss_index = None

    def query_by_item(self, item_id: int, top_n: int) -> list[tuple[int, float]]:
        if self.vectors is None or item_id not in self.item_ids:
            return []
        idx = self.item_ids.index(int(item_id))
        return self.query_by_vector(self.vectors[idx], top_n=top_n, exclude_item_id=item_id)

    def query_by_vector(
        self, vector: np.ndarray, top_n: int, exclude_item_id: int | None = None
    ) -> list[tuple[int, float]]:
        if self.vectors is None:
            return []

        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        q = q / q_norm

        k = min(top_n + 1, len(self.item_ids))
        results: list[tuple[int, float]] = []

        if self.faiss_index is not None:
            scores, idxs = self.faiss_index.search(q, k)
            for score, idx in zip(scores[0], idxs[0]):
                item_id = self.item_ids[int(idx)]
                if exclude_item_id is not None and item_id == exclude_item_id:
                    continue
                results.append((item_id, float(score)))
            return results[:top_n]

        if self.nn_index is None:
            return []
        distances, idxs = self.nn_index.kneighbors(q, n_neighbors=k)
        for dist, idx in zip(distances[0], idxs[0]):
            item_id = self.item_ids[int(idx)]
            if exclude_item_id is not None and item_id == exclude_item_id:
                continue
            score = 1.0 - float(dist)
            results.append((item_id, score))
        return results[:top_n]

"""Machine learning recommendation engine implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.vector_index import VectorIndex


@dataclass
class RecommendationResult:
    item_id: int
    score: float
    source: str


class RecommenderEngine:
    """Recommender engine with collaborative, content, and hybrid methods."""

    def __init__(self) -> None:
        self.user_item_matrix = pd.DataFrame()
        self.user_similarity = pd.DataFrame()
        self.item_similarity = pd.DataFrame()
        self.svd_predictions = pd.DataFrame()
        self.items_df = pd.DataFrame()
        self.popularity = pd.Series(dtype=float)
        self.content_similarity: np.ndarray | None = None
        self.item_index: dict[int, int] = {}
        self.ready = False
        self.user_activity = pd.Series(dtype=float)
        self.item_popularity = pd.Series(dtype=float)
        self.item_genre_map: dict[int, str] = {}
        self.tfidf_matrix = None
        self.vector_index = VectorIndex()

    def fit(self, users_df: pd.DataFrame, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> None:
        """Fit all recommendation models from tabular data."""
        self.items_df = items_df.copy()
        self.item_genre_map = {
            int(row["id"]): str(row.get("genre", "") or "")
            for _, row in items_df.iterrows()
        }
        self.popularity = (
            ratings_df.groupby("item_id")["rating"].mean().sort_values(ascending=False)
            if not ratings_df.empty
            else pd.Series(dtype=float)
        )
        self.item_index = {item_id: idx for idx, item_id in enumerate(items_df["id"].tolist())}

        if ratings_df.empty:
            self.user_item_matrix = pd.DataFrame()
            self.user_similarity = pd.DataFrame()
            self.item_similarity = pd.DataFrame()
            self.svd_predictions = pd.DataFrame()
            self._fit_content_model(items_df)
            self.ready = True
            return

        processed = self._preprocess_interactions(ratings_df)
        self.user_activity = processed.groupby("user_id")["adjusted_score"].count().astype(float)
        self.item_popularity = (
            processed.groupby("item_id")["adjusted_score"].sum().sort_values(ascending=False)
        )

        adjusted_matrix = processed.pivot_table(
            index="user_id", columns="item_id", values="adjusted_score", fill_value=0.0
        )
        self.user_item_matrix = adjusted_matrix

        user_sim = cosine_similarity(adjusted_matrix)
        self.user_similarity = pd.DataFrame(
            user_sim, index=adjusted_matrix.index, columns=adjusted_matrix.index
        )

        item_sim = cosine_similarity(adjusted_matrix.T)
        self.item_similarity = pd.DataFrame(
            item_sim, index=adjusted_matrix.columns, columns=adjusted_matrix.columns
        )

        self._fit_svd(adjusted_matrix)
        self._fit_content_model(items_df)
        self.ready = True

    def _preprocess_interactions(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Apply implicit signal, confidence weights, and time-decay."""
        processed = ratings_df.copy()
        if "created_at" not in processed.columns:
            processed["created_at"] = pd.Timestamp.utcnow()
        processed["created_at"] = pd.to_datetime(processed["created_at"], errors="coerce").fillna(
            pd.Timestamp.utcnow()
        )

        now = pd.Timestamp.utcnow()
        age_days = (now - processed["created_at"]).dt.total_seconds() / 86400.0
        decay = np.exp(-0.015 * age_days.clip(lower=0))

        # Normalize explicit ratings (0-5 -> 0-1), add implicit feedback signal.
        explicit_norm = (processed["rating"].clip(lower=0.0, upper=5.0) / 5.0).astype(float)
        implicit_signal = (processed["rating"] > 0).astype(float)

        user_interaction_count = processed.groupby("user_id")["item_id"].transform("count").astype(float)
        confidence = 1.0 + np.log1p(user_interaction_count) * 0.35

        adjusted = (0.75 * explicit_norm + 0.25 * implicit_signal) * confidence * decay
        processed["adjusted_score"] = adjusted.astype(float)
        return processed

    def _fit_svd(self, matrix: pd.DataFrame) -> None:
        if matrix.empty:
            self.svd_predictions = pd.DataFrame()
            return

        num_features = min(20, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if num_features <= 1:
            self.svd_predictions = matrix.copy()
            return

        svd = TruncatedSVD(n_components=num_features, random_state=42)
        transformed = svd.fit_transform(matrix)
        reconstructed = np.dot(transformed, svd.components_)
        self.svd_predictions = pd.DataFrame(
            reconstructed, index=matrix.index, columns=matrix.columns
        )

    def _fit_content_model(self, items_df: pd.DataFrame) -> None:
        if items_df.empty:
            self.content_similarity = np.array([])
            self.tfidf_matrix = None
            self.vector_index.fit([], np.array([]))
            return

        text_series = (
            items_df["genre"].fillna("")
            + " "
            + items_df["tags"].fillna("")
            + " "
            + items_df["description"].fillna("")
        )
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(text_series)
        self.tfidf_matrix = tfidf_matrix
        self.content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.vector_index.fit(
            item_ids=[int(x) for x in items_df["id"].tolist()],
            vectors=tfidf_matrix.toarray().astype(np.float32),
        )

    def recommend_user_based(self, user_id: int, top_n: int = 10) -> list[RecommendationResult]:
        """User-based collaborative recommendations."""
        if user_id not in self.user_item_matrix.index or self.user_similarity.empty:
            return []

        similar_users = self.user_similarity.loc[user_id].drop(index=user_id, errors="ignore")
        similar_users = similar_users[similar_users > 0].sort_values(ascending=False).head(20)

        scores: dict[int, float] = {}
        already_rated = set(
            self.user_item_matrix.columns[self.user_item_matrix.loc[user_id] > 0].tolist()
        )

        for neighbor_id, sim_score in similar_users.items():
            neighbor_ratings = self.user_item_matrix.loc[neighbor_id]
            for item_id, rating in neighbor_ratings[neighbor_ratings > 0].items():
                if item_id in already_rated:
                    continue
                scores[item_id] = scores.get(item_id, 0.0) + sim_score * float(rating)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [RecommendationResult(item_id=i, score=s, source="user_cf") for i, s in ranked]

    def recommend_item_based(self, user_id: int, top_n: int = 10) -> list[RecommendationResult]:
        """Item-based collaborative recommendations."""
        if user_id not in self.user_item_matrix.index or self.item_similarity.empty:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        scores: dict[int, float] = {}

        for item_id, rating in rated_items.items():
            similar_items = self.item_similarity[item_id].sort_values(ascending=False).head(30)
            for candidate_id, sim in similar_items.items():
                if candidate_id == item_id or user_ratings.get(candidate_id, 0) > 0:
                    continue
                scores[candidate_id] = scores.get(candidate_id, 0.0) + float(sim) * float(rating)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [RecommendationResult(item_id=i, score=s, source="item_cf") for i, s in ranked]

    def recommend_svd(self, user_id: int, top_n: int = 10) -> list[RecommendationResult]:
        """Latent-factor recommendations using TruncatedSVD reconstruction."""
        if user_id not in self.svd_predictions.index or self.svd_predictions.empty:
            return []

        pred_scores = self.svd_predictions.loc[user_id].sort_values(ascending=False)
        rated_items = set(
            self.user_item_matrix.columns[self.user_item_matrix.loc[user_id] > 0].tolist()
        )
        recs = [
            RecommendationResult(item_id=int(item_id), score=float(score), source="svd")
            for item_id, score in pred_scores.items()
            if item_id not in rated_items
        ]
        return recs[:top_n]

    def recommend_content_for_user(self, user_id: int, top_n: int = 10) -> list[RecommendationResult]:
        """Content-based recommendations based on user's historical liked items."""
        if self.content_similarity is None or self.items_df.empty or self.tfidf_matrix is None:
            return []
        if user_id not in self.user_item_matrix.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        threshold = float(user_ratings[user_ratings > 0].median()) if (user_ratings > 0).any() else 0.0
        liked_items = user_ratings[user_ratings >= max(threshold, 0.2)].index.tolist()
        if not liked_items:
            return []

        aggregate_scores = np.zeros(len(self.items_df), dtype=np.float32)
        profile_weights = []
        profile_vectors = []
        for item_id in liked_items:
            idx = self.item_index.get(int(item_id))
            if idx is not None:
                profile_vectors.append(self.tfidf_matrix[idx].toarray().flatten())
                profile_weights.append(float(user_ratings.get(item_id, 1.0)))
        if profile_vectors:
            w = np.array(profile_weights, dtype=np.float32)
            w = w / (w.sum() if w.sum() > 0 else 1.0)
            user_profile = np.average(np.array(profile_vectors, dtype=np.float32), axis=0, weights=w)
            ann_candidates = self.vector_index.query_by_vector(user_profile, top_n=top_n * 3)
            rated_items = set(user_ratings[user_ratings > 0].index.tolist())
            ann_results = [
                RecommendationResult(item_id=item_id, score=score, source="ann_content")
                for item_id, score in ann_candidates
                if item_id not in rated_items
            ]
            if ann_results:
                return ann_results[:top_n]

        for item_id in liked_items:
            idx = self.item_index.get(int(item_id))
            if idx is not None:
                aggregate_scores += self.content_similarity[idx]

        rated_items = set(user_ratings[user_ratings > 0].index.tolist())
        candidates = []
        for idx, score in enumerate(aggregate_scores):
            item_id = int(self.items_df.iloc[idx]["id"])
            if item_id in rated_items:
                continue
            candidates.append(RecommendationResult(item_id=item_id, score=float(score), source="content"))
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_n]

    def recommend_ann_for_user(self, user_id: int, top_n: int = 10) -> list[RecommendationResult]:
        """ANN retrieval using profile vector over item embeddings."""
        if self.tfidf_matrix is None or user_id not in self.user_item_matrix.index:
            return []
        user_ratings = self.user_item_matrix.loc[user_id]
        liked_items = user_ratings[user_ratings > 0].index.tolist()
        if not liked_items:
            return []
        profile_vectors = []
        weights = []
        for item_id in liked_items:
            idx = self.item_index.get(int(item_id))
            if idx is None:
                continue
            profile_vectors.append(self.tfidf_matrix[idx].toarray().flatten())
            weights.append(float(user_ratings.get(item_id, 1.0)))
        if not profile_vectors:
            return []
        w = np.array(weights, dtype=np.float32)
        w = w / (w.sum() if w.sum() > 0 else 1.0)
        user_profile = np.average(np.array(profile_vectors, dtype=np.float32), axis=0, weights=w)
        rated_items = set(user_ratings[user_ratings > 0].index.tolist())
        ann = self.vector_index.query_by_vector(user_profile, top_n=top_n * 2)
        return [
            RecommendationResult(item_id=item_id, score=score, source="ann")
            for item_id, score in ann
            if item_id not in rated_items
        ][:top_n]

    def recommend_popular(self, top_n: int = 10) -> list[RecommendationResult]:
        """Cold-start fallback: return globally popular items."""
        results: list[RecommendationResult] = []
        if not self.item_popularity.empty:
            for item_id, score in self.item_popularity.head(top_n).items():
                results.append(
                    RecommendationResult(item_id=int(item_id), score=float(score), source="popular")
                )
            return results
        if not self.popularity.empty:
            for item_id, score in self.popularity.head(top_n).items():
                results.append(
                    RecommendationResult(item_id=int(item_id), score=float(score), source="popular")
                )
            return results

        for _, row in self.items_df.head(top_n).iterrows():
            results.append(
                RecommendationResult(item_id=int(row["id"]), score=1.0, source="popular")
            )
        return results

    def similar_items(self, item_id: int, top_n: int = 10) -> list[RecommendationResult]:
        """Return similar items using item-cf, fallback to content similarity."""
        ann_similar = self.vector_index.query_by_item(item_id=item_id, top_n=top_n)
        if ann_similar:
            return [
                RecommendationResult(item_id=int(candidate_id), score=float(score), source="ann")
                for candidate_id, score in ann_similar
            ]

        if self.item_similarity is not None and item_id in self.item_similarity.columns:
            sims = self.item_similarity[item_id].sort_values(ascending=False)
            sims = sims.drop(index=item_id, errors="ignore").head(top_n)
            return [
                RecommendationResult(item_id=int(candidate), score=float(score), source="item_cf")
                for candidate, score in sims.items()
            ]

        idx = self.item_index.get(item_id)
        if idx is None or self.content_similarity is None or len(self.content_similarity) == 0:
            return []
        scores = self.content_similarity[idx]
        ranked = np.argsort(scores)[::-1]
        results = []
        for candidate_idx in ranked:
            candidate_id = int(self.items_df.iloc[candidate_idx]["id"])
            if candidate_id == item_id:
                continue
            results.append(
                RecommendationResult(
                    item_id=candidate_id,
                    score=float(scores[candidate_idx]),
                    source="content",
                )
            )
            if len(results) >= top_n:
                break
        return results

    def recommend_hybrid(self, user_id: int, top_n: int = 10) -> list[RecommendationResult]:
        """Hybrid recommendations combining user-cf, item-cf, content, and SVD."""
        if not self.ready:
            return []

        if user_id not in self.user_item_matrix.index:
            return self.recommend_popular(top_n=top_n)

        user_history = self.user_item_matrix.loc[user_id]
        rated_items = user_history[user_history > 0]
        activity_count = int(len(rated_items))
        user_weights = self._dynamic_weights(user_id=user_id, activity_count=activity_count)

        candidate_lists = {
            "user_cf": self.recommend_user_based(user_id, top_n=top_n * 3),
            "item_cf": self.recommend_item_based(user_id, top_n=top_n * 3),
            "content": self.recommend_content_for_user(user_id, top_n=top_n * 3),
            "svd": self.recommend_svd(user_id, top_n=top_n * 3),
            "ann": self.recommend_ann_for_user(user_id, top_n=top_n * 3),
        }
        candidates = self._generate_candidates(candidate_lists)
        combined = self._rank_candidates(candidates, candidate_lists, user_weights)

        if not combined:
            return self.recommend_popular(top_n=top_n)

        ranked_items = self._apply_diversity_filter(combined, top_n=top_n)
        return [RecommendationResult(item_id=item_id, score=score, source="hybrid") for item_id, score in ranked_items]

    def recommend_hybrid_v1(self, user_id: int, top_n: int = 10) -> list[RecommendationResult]:
        """Baseline hybrid without advanced re-ranking (A/B control variant)."""
        if not self.ready:
            return []
        if user_id not in self.user_item_matrix.index:
            return self.recommend_popular(top_n=top_n)

        rec_lists = [
            (self.recommend_user_based(user_id, top_n=top_n * 2), 0.25),
            (self.recommend_item_based(user_id, top_n=top_n * 2), 0.25),
            (self.recommend_content_for_user(user_id, top_n=top_n * 2), 0.20),
            (self.recommend_svd(user_id, top_n=top_n * 2), 0.30),
        ]
        combined: dict[int, float] = {}
        for recs, weight in rec_lists:
            for rec in recs:
                combined[rec.item_id] = combined.get(rec.item_id, 0.0) + rec.score * weight

        if not combined:
            return self.recommend_popular(top_n=top_n)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [RecommendationResult(item_id=item_id, score=score, source="hybrid_v1") for item_id, score in ranked]

    def _dynamic_weights(self, user_id: int, activity_count: int) -> dict[str, float]:
        """Choose hybrid weights based on interaction density and profile richness."""
        if activity_count < 3:
            return {
                "user_cf": 0.10,
                "item_cf": 0.10,
                "content": 0.30,
                "ann": 0.20,
                "svd": 0.10,
                "popular": 0.20,
            }
        if activity_count < 8:
            return {
                "user_cf": 0.20,
                "item_cf": 0.20,
                "content": 0.20,
                "ann": 0.15,
                "svd": 0.15,
                "popular": 0.10,
            }
        return {
            "user_cf": 0.20,
            "item_cf": 0.20,
            "content": 0.10,
            "ann": 0.20,
            "svd": 0.25,
            "popular": 0.05,
        }

    def _generate_candidates(
        self, candidate_lists: dict[str, list[RecommendationResult]]
    ) -> set[int]:
        """Candidate generation stage from all retrieval methods."""
        candidates: set[int] = set()
        for recs in candidate_lists.values():
            candidates.update(rec.item_id for rec in recs)
        return candidates

    def _rank_candidates(
        self,
        candidates: set[int],
        candidate_lists: dict[str, list[RecommendationResult]],
        user_weights: dict[str, float],
    ) -> dict[int, float]:
        """Ranking stage combining method scores with popularity novelty balancing."""
        score_maps: dict[str, dict[int, float]] = {}
        for source, recs in candidate_lists.items():
            score_maps[source] = {rec.item_id: float(rec.score) for rec in recs}

        popularity_max = float(self.item_popularity.max()) if not self.item_popularity.empty else 1.0
        combined: dict[int, float] = {}
        for item_id in candidates:
            score = 0.0
            for source in ("user_cf", "item_cf", "content", "svd", "ann"):
                score += user_weights[source] * score_maps.get(source, {}).get(item_id, 0.0)

            popularity_score = (
                float(self.item_popularity.get(item_id, 0.0)) / popularity_max if popularity_max > 0 else 0.0
            )
            # Slight novelty boost to avoid over-concentrating only popular items.
            novelty_boost = 1.0 - min(popularity_score, 1.0)
            score += user_weights.get("popular", 0.0) * popularity_score
            score += 0.05 * novelty_boost
            combined[item_id] = score
        return combined

    def _apply_diversity_filter(self, combined: dict[int, float], top_n: int) -> list[tuple[int, float]]:
        """Simple diversity re-ranking: limit same-genre streak in top results."""
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        selected: list[tuple[int, float]] = []
        genre_counts: dict[str, int] = {}

        for item_id, score in ranked:
            genre = self.item_genre_map.get(item_id, "").strip().lower()
            if genre and genre_counts.get(genre, 0) >= 2:
                continue
            selected.append((item_id, score))
            if genre:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            if len(selected) >= top_n:
                return selected

        for item_id, score in ranked:
            if len(selected) >= top_n:
                break
            if (item_id, score) not in selected:
                selected.append((item_id, score))
        return selected[:top_n]

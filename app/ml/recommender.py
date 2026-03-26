"""Machine learning recommendation engine implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

    def fit(self, users_df: pd.DataFrame, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> None:
        """Fit all recommendation models from tabular data."""
        self.items_df = items_df.copy()
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

        matrix = ratings_df.pivot_table(
            index="user_id", columns="item_id", values="rating", fill_value=0.0
        )
        self.user_item_matrix = matrix

        user_sim = cosine_similarity(matrix)
        self.user_similarity = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)

        item_sim = cosine_similarity(matrix.T)
        self.item_similarity = pd.DataFrame(
            item_sim, index=matrix.columns, columns=matrix.columns
        )

        self._fit_svd(matrix)
        self._fit_content_model(items_df)
        self.ready = True

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
        self.content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

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
        if self.content_similarity is None or self.items_df.empty:
            return []
        if user_id not in self.user_item_matrix.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        liked_items = user_ratings[user_ratings >= 3.5].index.tolist()
        if not liked_items:
            return []

        aggregate_scores = np.zeros(len(self.items_df))
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

    def recommend_popular(self, top_n: int = 10) -> list[RecommendationResult]:
        """Cold-start fallback: return globally popular items."""
        results: list[RecommendationResult] = []
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

        rec_lists = [
            (self.recommend_user_based(user_id, top_n=top_n * 2), 0.25),
            (self.recommend_item_based(user_id, top_n=top_n * 2), 0.25),
            (self.recommend_content_for_user(user_id, top_n=top_n * 2), 0.20),
            (self.recommend_svd(user_id, top_n=top_n * 2), 0.30),
        ]

        combined: dict[int, float] = {}
        source_track: dict[int, str] = {}
        for recs, weight in rec_lists:
            for rec in recs:
                combined[rec.item_id] = combined.get(rec.item_id, 0.0) + rec.score * weight
                source_track[rec.item_id] = "hybrid"

        if not combined:
            return self.recommend_popular(top_n=top_n)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            RecommendationResult(item_id=item_id, score=score, source=source_track[item_id])
            for item_id, score in ranked
        ]

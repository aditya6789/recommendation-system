"""Learning-to-rank service (pointwise) for final recommendation re-ranking."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class LTRRanker:
    """Pointwise ranker trained from historical rating interactions."""

    def __init__(self) -> None:
        self.model = GradientBoostingRegressor(random_state=42)
        self.is_trained = False
        self.user_stats = pd.DataFrame()
        self.item_stats = pd.DataFrame()
        self.global_mean = 0.0
        self.model_path = Path("artifacts/ltr_ranker.joblib")

    def fit(self, ratings_df: pd.DataFrame) -> bool:
        if ratings_df.empty or len(ratings_df) < 10:
            self.is_trained = False
            return False

        df = ratings_df.copy()
        df["rating"] = df["rating"].astype(float)
        self.global_mean = float(df["rating"].mean())

        self.user_stats = df.groupby("user_id")["rating"].agg(["mean", "count"]).rename(
            columns={"mean": "user_mean", "count": "user_count"}
        )
        self.item_stats = df.groupby("item_id")["rating"].agg(["mean", "count"]).rename(
            columns={"mean": "item_mean", "count": "item_count"}
        )

        train = df.merge(self.user_stats, left_on="user_id", right_index=True, how="left").merge(
            self.item_stats, left_on="item_id", right_index=True, how="left"
        )
        X = train[["user_mean", "user_count", "item_mean", "item_count"]].fillna(
            {
                "user_mean": self.global_mean,
                "user_count": 1.0,
                "item_mean": self.global_mean,
                "item_count": 1.0,
            }
        )
        y = train["rating"].astype(float)
        self.model.fit(X, y)
        self.is_trained = True

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "user_stats": self.user_stats,
                "item_stats": self.item_stats,
                "global_mean": self.global_mean,
            },
            self.model_path,
        )
        return True

    def load_if_available(self) -> bool:
        if not self.model_path.exists():
            return False
        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.user_stats = payload["user_stats"]
        self.item_stats = payload["item_stats"]
        self.global_mean = float(payload["global_mean"])
        self.is_trained = True
        return True

    def rerank(
        self,
        user_id: int,
        candidates: list[tuple[int, float, str]],
        top_n: int,
    ) -> list[tuple[int, float, str]]:
        if not candidates:
            return []
        if not self.is_trained:
            return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]

        user_row = (
            self.user_stats.loc[user_id]
            if user_id in self.user_stats.index
            else pd.Series({"user_mean": self.global_mean, "user_count": 1.0})
        )
        rows = []
        for item_id, base_score, source in candidates:
            item_row = (
                self.item_stats.loc[item_id]
                if item_id in self.item_stats.index
                else pd.Series({"item_mean": self.global_mean, "item_count": 1.0})
            )
            rows.append(
                {
                    "item_id": item_id,
                    "source": source,
                    "base_score": float(base_score),
                    "user_mean": float(user_row["user_mean"]),
                    "user_count": float(user_row["user_count"]),
                    "item_mean": float(item_row["item_mean"]),
                    "item_count": float(item_row["item_count"]),
                }
            )
        X = pd.DataFrame(rows)[["user_mean", "user_count", "item_mean", "item_count"]]
        preds = self.model.predict(X)
        scored = []
        for row, pred in zip(rows, preds):
            final_score = 0.6 * row["base_score"] + 0.4 * float(pred) / 5.0
            scored.append((int(row["item_id"]), float(final_score), str(row["source"])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


ltr_ranker = LTRRanker()

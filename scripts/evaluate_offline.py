"""Offline evaluation for recommendation quality (Precision/Recall/MAP/NDCG/Coverage@K)."""

import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import select

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.db.database import SessionLocal
from app.models import Item, Rating, User
from app.services.recommendation_service import recommendation_service


def train_test_split_leave_one_out(ratings_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out one random item per eligible user for evaluation."""
    test_rows = []
    train_rows = []
    random.seed(42)

    grouped = ratings_df.groupby("user_id")
    for _, group in grouped:
        if len(group) < 2:
            train_rows.extend(group.to_dict("records"))
            continue
        test_idx = random.choice(list(group.index))
        for idx, row in group.iterrows():
            if idx == test_idx:
                test_rows.append(row.to_dict())
            else:
                train_rows.append(row.to_dict())
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def evaluate(k: int = 10) -> dict:
    db = SessionLocal()
    try:
        users = db.execute(select(User)).scalars().all()
        items = db.execute(select(Item)).scalars().all()
        ratings = db.execute(select(Rating)).scalars().all()
        if not ratings:
            return {"message": "No ratings found for evaluation."}

        ratings_df = pd.DataFrame(
            [
                {
                    "user_id": r.user_id,
                    "item_id": r.item_id,
                    "rating": r.rating,
                    "created_at": r.created_at,
                }
                for r in ratings
            ]
        )
        train_df, test_df = train_test_split_leave_one_out(ratings_df)

        users_df = pd.DataFrame([{"id": u.id, "name": u.name} for u in users])
        items_df = pd.DataFrame(
            [
                {
                    "id": i.id,
                    "title": i.title,
                    "genre": i.genre or "",
                    "tags": i.tags or "",
                    "description": i.description or "",
                }
                for i in items
            ]
        )
        recommendation_service.engine.fit(users_df, items_df, train_df)

        truth_map = defaultdict(set)
        for _, row in test_df.iterrows():
            truth_map[int(row["user_id"])].add(int(row["item_id"]))

        precision_sum = 0.0
        recall_sum = 0.0
        map_sum = 0.0
        ndcg_sum = 0.0
        covered_items = set()
        evaluated_users = 0

        for user_id, relevant_items in truth_map.items():
            recs = recommendation_service.engine.recommend_hybrid(user_id=user_id, top_n=k)
            rec_item_ids = [r.item_id for r in recs]
            if not rec_item_ids:
                continue

            hit_count = len(set(rec_item_ids).intersection(relevant_items))
            precision_sum += hit_count / k
            recall_sum += hit_count / max(1, len(relevant_items))
            map_sum += average_precision_at_k(rec_item_ids, relevant_items, k)
            ndcg_sum += ndcg_at_k(rec_item_ids, relevant_items, k)
            covered_items.update(rec_item_ids)
            evaluated_users += 1

        total_items = len(items_df) if not items_df.empty else 1
        return {
            "k": k,
            "evaluated_users": evaluated_users,
            "precision_at_k": round(precision_sum / max(1, evaluated_users), 4),
            "recall_at_k": round(recall_sum / max(1, evaluated_users), 4),
            "map_at_k": round(map_sum / max(1, evaluated_users), 4),
            "ndcg_at_k": round(ndcg_sum / max(1, evaluated_users), 4),
            "coverage_at_k": round(len(covered_items) / total_items, 4),
        }
    finally:
        db.close()


def average_precision_at_k(pred_items: list[int], relevant_items: set[int], k: int) -> float:
    """Compute AP@K."""
    if not relevant_items:
        return 0.0
    ap = 0.0
    hits = 0
    for idx, item_id in enumerate(pred_items[:k], start=1):
        if item_id in relevant_items:
            hits += 1
            ap += hits / idx
    return ap / min(len(relevant_items), k)


def ndcg_at_k(pred_items: list[int], relevant_items: set[int], k: int) -> float:
    """Compute NDCG@K with binary relevance."""
    dcg = 0.0
    for i, item_id in enumerate(pred_items[:k], start=1):
        rel = 1.0 if item_id in relevant_items else 0.0
        dcg += rel / np.log2(i + 1)
    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


if __name__ == "__main__":
    print(evaluate(k=10))

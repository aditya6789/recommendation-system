"""Service layer connecting DB entities with ML recommender engine."""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.recommender import RecommenderEngine
from app.models import Item, Rating, User
from app.schemas.recommendation import RecommendationItem
from app.services.cache import CacheClient

logger = logging.getLogger(__name__)


class RecommendationService:
    """Coordinates data retrieval, model rebuilds, and recommendation serving."""

    def __init__(self) -> None:
        self.engine = RecommenderEngine()
        self.cache = CacheClient()

    def rebuild(self, db: Session) -> None:
        """Recompute recommendation artifacts from current database state."""
        users = db.execute(select(User)).scalars().all()
        items = db.execute(select(Item)).scalars().all()
        ratings = db.execute(select(Rating)).scalars().all()

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
        ratings_df = pd.DataFrame(
            [{"user_id": r.user_id, "item_id": r.item_id, "rating": r.rating} for r in ratings]
        )

        if items_df.empty:
            logger.info("No items available; recommender state reset")
            self.engine = RecommenderEngine()
            return

        self.engine.fit(users_df=users_df, items_df=items_df, ratings_df=ratings_df)
        self.cache.delete_pattern("recommendations:*")
        logger.info("Recommender rebuilt with %s users, %s items, %s ratings", len(users), len(items), len(ratings))

    def _item_title_map(self, db: Session) -> dict[int, str]:
        items = db.execute(select(Item.id, Item.title)).all()
        return {item_id: title for item_id, title in items}

    def recommend(self, db: Session, user_id: int, top_n: int = 10) -> list[RecommendationItem]:
        """Return hybrid recommendations for a user."""
        cache_key = f"recommendations:user:{user_id}:n:{top_n}"
        cached = self.cache.get_json(cache_key)
        if cached is not None:
            return [RecommendationItem(**item) for item in cached]

        recs = self.engine.recommend_hybrid(user_id=user_id, top_n=top_n)
        title_map = self._item_title_map(db)
        response = [
            RecommendationItem(
                item_id=rec.item_id,
                title=title_map.get(rec.item_id, f"Item {rec.item_id}"),
                score=round(float(rec.score), 4),
                source=rec.source,
            )
            for rec in recs
        ]
        self.cache.set_json(cache_key, [item.model_dump() for item in response], ttl=600)
        return response

    def similar_items(self, db: Session, item_id: int, top_n: int = 10) -> list[RecommendationItem]:
        """Return similar items for a given item."""
        recs = self.engine.similar_items(item_id=item_id, top_n=top_n)
        title_map = self._item_title_map(db)
        return [
            RecommendationItem(
                item_id=rec.item_id,
                title=title_map.get(rec.item_id, f"Item {rec.item_id}"),
                score=round(float(rec.score), 4),
                source=rec.source,
            )
            for rec in recs
        ]


recommendation_service = RecommendationService()

"""Service layer connecting DB entities with ML recommender engine."""

from __future__ import annotations

import logging
import random
from hashlib import sha256

import pandas as pd
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.db.config import get_settings
from app.ml.recommender import RecommenderEngine
from app.models import ABAssignment, Item, ModelVersion, Rating, RecommendationEvent, User, UserFeature
from app.schemas.recommendation import RecommendationItem
from app.services.cache import CacheClient
from app.services.ltr_ranker import ltr_ranker
from app.services.metrics import RECOMMENDATION_IMPRESSIONS

logger = logging.getLogger(__name__)


class RecommendationService:
    """Coordinates data retrieval, model rebuilds, and recommendation serving."""

    def __init__(self) -> None:
        settings = get_settings()
        self.engine = RecommenderEngine()
        self.cache = CacheClient()
        self.experiment_name = "hybrid_ranker_experiment_v1"
        self.active_model_version_id: int | None = None
        self.enable_bandit_auto = settings.enable_bandit_auto
        ltr_ranker.load_if_available()

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

        if items_df.empty:
            logger.info("No items available; recommender state reset")
            self.engine = RecommenderEngine()
            return

        self.engine.fit(users_df=users_df, items_df=items_df, ratings_df=ratings_df)
        self.active_model_version_id = self._register_model_version(
            db=db,
            variant="hybrid_v2",
            parameters={
                "signals": ["user_cf", "item_cf", "content", "svd"],
                "features": ["time_decay", "confidence_weighting", "diversity_rerank"],
            },
            metrics={
                "users": len(users),
                "items": len(items),
                "ratings": len(ratings),
            },
        )
        self.cache.delete_pattern("recommendations:*")
        ltr_ranker.fit(ratings_df=ratings_df)
        logger.info("Recommender rebuilt with %s users, %s items, %s ratings", len(users), len(items), len(ratings))

    def _register_model_version(
        self, db: Session, variant: str, parameters: dict, metrics: dict
    ) -> int:
        """Create new active model version and deactivate older entries."""
        db.execute(update(ModelVersion).values(is_active=False))
        version = ModelVersion(
            model_name="hybrid_recommender",
            variant=variant,
            parameters=parameters,
            metrics=metrics,
            is_active=True,
        )
        db.add(version)
        db.commit()
        db.refresh(version)
        return int(version.id)

    def _assign_variant(self, db: Session, user_id: int) -> str:
        """Assign user variant using bandit-auto or deterministic A/B."""
        if self.enable_bandit_auto:
            return self._thompson_sample_variant(db)

        # Deterministic fallback A/B.
        assignment = db.execute(
            select(ABAssignment).where(
                ABAssignment.user_id == user_id,
                ABAssignment.experiment_name == self.experiment_name,
            )
        ).scalar_one_or_none()
        if assignment:
            return assignment.variant

        seed = f"{self.experiment_name}:{user_id}".encode("utf-8")
        bucket = int(sha256(seed).hexdigest(), 16) % 100
        variant = "v1" if bucket < 50 else "v2"
        assignment = ABAssignment(
            user_id=user_id, experiment_name=self.experiment_name, variant=variant
        )
        db.add(assignment)
        db.commit()
        return variant

    def _thompson_sample_variant(self, db: Session) -> str:
        """Adaptive variant selection using Thompson Sampling over click-through."""
        stats = self._variant_event_stats(db)
        variants = ["v1", "v2"]
        best_variant = "v2"
        best_sample = -1.0
        for variant in variants:
            impressions = stats.get(variant, {}).get("impression", 0)
            clicks = stats.get(variant, {}).get("click", 0)
            alpha = 1 + clicks
            beta = 1 + max(0, impressions - clicks)
            sample = random.betavariate(alpha, beta)
            if sample > best_sample:
                best_sample = sample
                best_variant = variant
        return best_variant

    def _variant_event_stats(self, db: Session) -> dict[str, dict[str, int]]:
        rows = db.execute(
            select(
                RecommendationEvent.experiment_variant,
                RecommendationEvent.event_type,
                func.count(RecommendationEvent.id),
            ).group_by(RecommendationEvent.experiment_variant, RecommendationEvent.event_type)
        ).all()
        by_variant: dict[str, dict[str, int]] = {}
        for variant, event_type, count in rows:
            if variant not in by_variant:
                by_variant[variant] = {"impression": 0, "click": 0, "watch": 0}
            by_variant[variant][event_type] = int(count)
        return by_variant

    def bandit_snapshot(self, db: Session) -> dict:
        """Expose current adaptive assignment priors and posterior means."""
        stats = self._variant_event_stats(db)
        out: dict[str, dict[str, float | int]] = {}
        for variant in ("v1", "v2"):
            impressions = stats.get(variant, {}).get("impression", 0)
            clicks = stats.get(variant, {}).get("click", 0)
            alpha = 1 + clicks
            beta = 1 + max(0, impressions - clicks)
            posterior_mean = alpha / (alpha + beta)
            out[variant] = {
                "impressions": impressions,
                "clicks": clicks,
                "alpha": alpha,
                "beta": beta,
                "posterior_mean_ctr": round(posterior_mean, 5),
            }
        return {"enabled": self.enable_bandit_auto, "variants": out}

    def _log_impressions(
        self, db: Session, user_id: int, variant: str, recommendations: list[RecommendationItem]
    ) -> None:
        events = [
            RecommendationEvent(
                user_id=user_id,
                item_id=item.item_id,
                model_version_id=self.active_model_version_id,
                experiment_variant=variant,
                event_type="impression",
                score=item.score,
            )
            for item in recommendations
        ]
        db.add_all(events)
        db.commit()
        RECOMMENDATION_IMPRESSIONS.labels(variant=variant).inc(len(recommendations))

    def _item_title_map(self, db: Session) -> dict[int, str]:
        items = db.execute(select(Item.id, Item.title)).all()
        return {item_id: title for item_id, title in items}

    def recommend(
        self, db: Session, user_id: int, top_n: int = 10, strategy: str = "auto"
    ) -> tuple[list[RecommendationItem], str, int | None]:
        """Return recommendations for a user with optional A/B strategy selection."""
        variant = self._assign_variant(db, user_id) if strategy == "auto" else strategy
        cache_key = f"recommendations:user:{user_id}:n:{top_n}:variant:{variant}"
        cached = self.cache.get_json(cache_key)
        if cached is not None:
            recs = [RecommendationItem(**item) for item in cached]
            self._log_impressions(db, user_id=user_id, variant=variant, recommendations=recs)
            return recs, variant, self.active_model_version_id

        if variant == "v1":
            recs = self.engine.recommend_hybrid_v1(user_id=user_id, top_n=top_n)
        else:
            recs = self.engine.recommend_hybrid(user_id=user_id, top_n=top_n)
            reranked = ltr_ranker.rerank(
                user_id=user_id,
                candidates=[(r.item_id, r.score, r.source) for r in recs],
                top_n=top_n,
            )
            recs = [type(recs[0])(item_id=i, score=s, source=f"{src}+ltr") for i, s, src in reranked] if recs else []
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
        self._log_impressions(db, user_id=user_id, variant=variant, recommendations=response)
        return response, variant, self.active_model_version_id

    def precompute_for_active_users(self, db: Session, top_n: int = 20, limit_users: int = 200) -> dict:
        """Precompute recommendations for active users and cache hot segments."""
        active_user_ids = db.execute(
            select(UserFeature.user_id).order_by(UserFeature.last_active_at.desc()).limit(limit_users)
        ).scalars().all()
        warmed = 0
        for user_id in active_user_ids:
            for variant in ("v1", "v2"):
                recs, _, _ = self.recommend(
                    db=db,
                    user_id=int(user_id),
                    top_n=top_n,
                    strategy=variant,
                )
                cache_key = f"recommendations:user:{user_id}:n:{top_n}:variant:{variant}"
                self.cache.set_json(cache_key, [item.model_dump() for item in recs], ttl=1800)
            warmed += 1

        # Segment cache fallback for cold users.
        segment_key = f"recommendations:segment:popular:n:{top_n}"
        popular = [
            RecommendationItem(item_id=r.item_id, title="", score=round(float(r.score), 4), source=r.source)
            for r in self.engine.recommend_popular(top_n=top_n)
        ]
        self.cache.set_json(segment_key, [item.model_dump() for item in popular], ttl=1800)
        return {"active_users_warmed": warmed, "segment_cache_key": segment_key}

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

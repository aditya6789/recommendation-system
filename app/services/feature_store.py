"""Online feature updater shared by API ingestion and stream consumers."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import UserFeature
from app.services.cache import CacheClient


class FeatureStoreService:
    """Maintains rolling online user features."""

    def __init__(self) -> None:
        self.cache = CacheClient()

    def apply_event(
        self,
        db: Session,
        user_id: int,
        event_type: str,
        watch_seconds: float = 0.0,
        event_ts: datetime | None = None,
    ) -> None:
        feature = db.execute(
            select(UserFeature).where(UserFeature.user_id == user_id)
        ).scalar_one_or_none()
        if feature is None:
            feature = UserFeature(user_id=user_id)
            db.add(feature)

        if event_type == "impression":
            feature.impression_count += 1
        elif event_type == "click":
            feature.click_count += 1
        elif event_type == "watch":
            feature.watch_seconds += watch_seconds or 0.0

        feature.last_active_at = event_ts or datetime.now(timezone.utc)
        db.commit()

        if self.cache.enabled and self.cache.client is not None:
            key = f"user_features:{user_id}"
            self.cache.client.hset(
                key,
                mapping={
                    "click_count": feature.click_count,
                    "impression_count": feature.impression_count,
                    "watch_seconds": feature.watch_seconds,
                    "last_active_at": feature.last_active_at.isoformat(),
                },
            )
            self.cache.client.expire(key, 86400)


feature_store_service = FeatureStoreService()

"""Real-time interaction ingestion and online feature updates."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import RecommendationEvent
from app.schemas.interaction import InteractionEventIn
from app.services.feature_store import feature_store_service
from app.services.kafka_producer import event_producer
from app.services.metrics import INTERACTION_EVENTS


class InteractionService:
    """Ingest user behavior events and update online features."""

    def ingest_event(self, db: Session, payload: InteractionEventIn) -> dict:
        """Persist event, publish stream event, and update online feature state."""
        event = RecommendationEvent(
            user_id=payload.user_id,
            item_id=payload.item_id,
            model_version_id=payload.model_version_id,
            experiment_variant=payload.experiment_variant,
            event_type=payload.event_type,
            score=payload.score,
        )
        db.add(event)
        db.commit()

        feature_store_service.apply_event(
            db=db,
            user_id=payload.user_id,
            event_type=payload.event_type,
            watch_seconds=payload.watch_seconds or 0.0,
            event_ts=datetime.now(timezone.utc),
        )
        INTERACTION_EVENTS.labels(event_type=payload.event_type).inc()
        event_producer.send_event(
            {
                "user_id": payload.user_id,
                "item_id": payload.item_id,
                "event_type": payload.event_type,
                "experiment_variant": payload.experiment_variant,
                "watch_seconds": payload.watch_seconds,
                "feature_applied": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return {"status": "ok", "event_type": payload.event_type}


interaction_service = InteractionService()

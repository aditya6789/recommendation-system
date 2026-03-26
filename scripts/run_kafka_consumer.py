"""Kafka consumer worker for rolling feature updates from event stream."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from kafka import KafkaConsumer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.db.config import get_settings
from app.db.database import Base, SessionLocal, engine
from app.models import Item, User
from app.services.feature_store import feature_store_service


def _parse_ts(raw_ts: str | None) -> datetime:
    if not raw_ts:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def run_consumer() -> None:
    settings = get_settings()
    if not settings.enable_kafka:
        print("Kafka consumer disabled: set ENABLE_KAFKA=true")
        return

    Base.metadata.create_all(bind=engine)
    consumer = KafkaConsumer(
        settings.kafka_topic_events,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    print(
        f"Kafka consumer started on {settings.kafka_bootstrap_servers}, "
        f"topic={settings.kafka_topic_events}, group={settings.kafka_consumer_group}"
    )

    for msg in consumer:
        payload = msg.value
        if bool(payload.get("feature_applied", False)):
            # Avoid double-apply when event originated from API ingestion path.
            continue
        user_id = int(payload.get("user_id", 0))
        item_id = int(payload.get("item_id", 0))
        event_type = str(payload.get("event_type", "impression"))
        watch_seconds = float(payload.get("watch_seconds", 0.0) or 0.0)
        event_ts = _parse_ts(payload.get("timestamp"))
        if user_id <= 0 or item_id <= 0:
            continue

        db = SessionLocal()
        try:
            if not db.get(User, user_id) or not db.get(Item, item_id):
                continue
            feature_store_service.apply_event(
                db=db,
                user_id=user_id,
                event_type=event_type,
                watch_seconds=watch_seconds,
                event_ts=event_ts,
            )
        finally:
            db.close()


if __name__ == "__main__":
    run_consumer()

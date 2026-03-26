"""Optional Kafka producer for streaming recommendation events."""

from __future__ import annotations

import json
import logging
from typing import Any

from kafka import KafkaProducer

from app.db.config import get_settings

logger = logging.getLogger(__name__)


class EventProducer:
    """Kafka event producer with graceful no-op fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = settings.enable_kafka
        self.topic = settings.kafka_topic_events
        self._producer: KafkaProducer | None = None
        if not self.enabled:
            return
        try:
            self._producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            logger.info("Kafka producer enabled on %s", settings.kafka_bootstrap_servers)
        except Exception as exc:
            logger.warning("Kafka unavailable, disabling producer: %s", exc)
            self.enabled = False

    def send_event(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self._producer is None:
            return
        try:
            self._producer.send(self.topic, payload)
            self._producer.flush(timeout=0.2)
        except Exception as exc:
            logger.warning("Failed to publish Kafka event: %s", exc)


event_producer = EventProducer()

"""Optional Redis cache wrapper."""

import json
import logging
from typing import Any

import redis

from app.db.config import get_settings

logger = logging.getLogger(__name__)


class CacheClient:
    """Thin cache abstraction with Redis fallback to no-op mode."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = settings.enable_redis_cache
        self.client: redis.Redis | None = None
        if self.enabled:
            try:
                self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                self.client.ping()
                logger.info("Redis cache enabled")
            except Exception as exc:
                logger.warning("Redis unavailable, disabling cache: %s", exc)
                self.enabled = False

    def get_json(self, key: str) -> Any | None:
        if not self.enabled or self.client is None:
            return None
        raw = self.client.get(key)
        return json.loads(raw) if raw else None

    def set_json(self, key: str, value: Any, ttl: int = 300) -> None:
        if not self.enabled or self.client is None:
            return
        self.client.setex(key, ttl, json.dumps(value))

    def delete(self, key: str) -> None:
        if self.enabled and self.client is not None:
            self.client.delete(key)

    def delete_pattern(self, pattern: str) -> None:
        if not self.enabled or self.client is None:
            return
        keys = list(self.client.scan_iter(match=pattern))
        if keys:
            self.client.delete(*keys)

"""Application settings and environment configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings loaded from environment variables."""

    database_url: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5432/recommendation_db"
    )
    redis_url: str = "redis://localhost:6379/0"
    enable_redis_cache: bool = False
    log_level: str = "INFO"
    enable_kafka: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_events: str = "recommendation.events"
    kafka_consumer_group: str = "recsys-feature-updater"
    enable_bandit_auto: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()

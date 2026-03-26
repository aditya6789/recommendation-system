"""Schemas for real-time interaction events."""

from pydantic import BaseModel, Field


class InteractionEventIn(BaseModel):
    user_id: int
    item_id: int
    event_type: str = Field(pattern="^(impression|click|watch)$")
    experiment_variant: str = "unknown"
    model_version_id: int | None = None
    score: float | None = None
    watch_seconds: float | None = 0.0

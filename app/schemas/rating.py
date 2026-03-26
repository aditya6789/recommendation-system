"""Rating request/response schemas."""

from pydantic import BaseModel, Field


class RatingCreate(BaseModel):
    user_id: int
    item_id: int
    rating: float = Field(ge=0.0, le=5.0)


class RatingRead(BaseModel):
    id: int
    user_id: int
    item_id: int
    rating: float

    model_config = {"from_attributes": True}

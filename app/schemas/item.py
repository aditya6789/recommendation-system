"""Item request/response schemas."""

from pydantic import BaseModel


class ItemCreate(BaseModel):
    title: str
    genre: str | None = None
    tags: str | None = None
    description: str | None = None


class ItemRead(BaseModel):
    id: int
    title: str
    genre: str | None = None
    tags: str | None = None
    description: str | None = None

    model_config = {"from_attributes": True}

"""Recommendation response schemas."""

from pydantic import BaseModel


class RecommendationItem(BaseModel):
    item_id: int
    title: str
    score: float
    source: str


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]


class SimilarItemsResponse(BaseModel):
    item_id: int
    similar_items: list[RecommendationItem]

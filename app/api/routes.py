"""HTTP API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models import Item, Rating, User
from app.schemas import (
    ItemCreate,
    ItemRead,
    RatingCreate,
    RatingRead,
    RecommendationResponse,
    SimilarItemsResponse,
    UserCreate,
    UserRead,
)
from app.services.recommendation_service import recommendation_service

router = APIRouter()


@router.post("/user", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate, db: Session = Depends(get_db)) -> User:
    user = User(name=payload.name, email=payload.email)
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already exists") from exc
    recommendation_service.rebuild(db)
    return user


@router.post("/item", response_model=ItemRead, status_code=status.HTTP_201_CREATED)
def create_item(payload: ItemCreate, db: Session = Depends(get_db)) -> Item:
    item = Item(
        title=payload.title,
        genre=payload.genre,
        tags=payload.tags,
        description=payload.description,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    recommendation_service.rebuild(db)
    return item


@router.post("/rate", response_model=RatingRead, status_code=status.HTTP_201_CREATED)
def rate_item(payload: RatingCreate, db: Session = Depends(get_db)) -> Rating:
    user = db.get(User, payload.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    item = db.get(Item, payload.item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    existing = db.execute(
        select(Rating).where(
            Rating.user_id == payload.user_id,
            Rating.item_id == payload.item_id,
        )
    ).scalar_one_or_none()

    if existing:
        existing.rating = payload.rating
        rating = existing
    else:
        rating = Rating(user_id=payload.user_id, item_id=payload.item_id, rating=payload.rating)
        db.add(rating)

    db.commit()
    db.refresh(rating)
    recommendation_service.rebuild(db)
    return rating


@router.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend_for_user(
    user_id: int,
    n: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> RecommendationResponse:
    if not db.get(User, user_id):
        raise HTTPException(status_code=404, detail="User not found")
    recommendations = recommendation_service.recommend(db=db, user_id=user_id, top_n=n)
    return RecommendationResponse(user_id=user_id, recommendations=recommendations)


@router.get("/similar/{item_id}", response_model=SimilarItemsResponse)
def similar_items(
    item_id: int,
    n: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> SimilarItemsResponse:
    if not db.get(Item, item_id):
        raise HTTPException(status_code=404, detail="Item not found")
    recommendations = recommendation_service.similar_items(db=db, item_id=item_id, top_n=n)
    return SimilarItemsResponse(item_id=item_id, similar_items=recommendations)

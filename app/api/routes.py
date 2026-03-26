"""HTTP API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models import Item, ModelVersion, Rating, RecommendationEvent, User, UserFeature
from app.schemas import (
    InteractionEventIn,
    ItemCreate,
    ItemRead,
    RatingCreate,
    RatingRead,
    RecommendationResponse,
    SimilarItemsResponse,
    UserCreate,
    UserRead,
)
from app.services.interaction_service import interaction_service
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
    strategy: str = Query(default="auto", pattern="^(auto|v1|v2)$"),
    db: Session = Depends(get_db),
) -> RecommendationResponse:
    if not db.get(User, user_id):
        raise HTTPException(status_code=404, detail="User not found")
    recommendations, variant, model_version_id = recommendation_service.recommend(
        db=db,
        user_id=user_id,
        top_n=n,
        strategy=strategy,
    )
    return RecommendationResponse(
        user_id=user_id,
        experiment_variant=variant,
        model_version_id=model_version_id,
        recommendations=recommendations,
    )


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


@router.get("/experiments/summary")
def experiment_summary(db: Session = Depends(get_db)) -> dict:
    """Return simple A/B impression summary and active model version."""
    rows = db.execute(
        select(
            RecommendationEvent.experiment_variant,
            func.count(RecommendationEvent.id),
        ).group_by(RecommendationEvent.experiment_variant)
    ).all()
    active = db.execute(
        select(ModelVersion).where(ModelVersion.is_active.is_(True)).order_by(ModelVersion.id.desc())
    ).scalars().first()
    return {
        "active_model_version": active.id if active else None,
        "active_variant": active.variant if active else None,
        "impressions_by_variant": {variant: count for variant, count in rows},
    }


@router.get("/experiments/performance")
def experiment_performance(db: Session = Depends(get_db)) -> dict:
    """Return online KPI metrics by variant (CTR and watch conversion)."""
    rows = db.execute(
        select(
            RecommendationEvent.experiment_variant,
            RecommendationEvent.event_type,
            func.count(RecommendationEvent.id),
        ).group_by(RecommendationEvent.experiment_variant, RecommendationEvent.event_type)
    ).all()

    by_variant: dict[str, dict[str, int]] = {}
    for variant, event_type, count in rows:
        if variant not in by_variant:
            by_variant[variant] = {"impression": 0, "click": 0, "watch": 0}
        by_variant[variant][event_type] = int(count)

    response: dict[str, dict] = {}
    for variant, counts in by_variant.items():
        impressions = counts.get("impression", 0)
        clicks = counts.get("click", 0)
        watches = counts.get("watch", 0)
        ctr = clicks / impressions if impressions > 0 else 0.0
        watch_conversion = watches / impressions if impressions > 0 else 0.0
        response[variant] = {
            "impressions": impressions,
            "clicks": clicks,
            "watches": watches,
            "ctr": round(ctr, 4),
            "watch_conversion_rate": round(watch_conversion, 4),
        }
    return {"variants": response}


@router.get("/experiments/bandit")
def experiment_bandit(db: Session = Depends(get_db)) -> dict:
    """Return current bandit priors/posteriors used by auto strategy."""
    return recommendation_service.bandit_snapshot(db)


@router.post("/events/interaction")
def ingest_interaction(payload: InteractionEventIn, db: Session = Depends(get_db)) -> dict:
    """Ingest real-time interaction event and update online features."""
    if not db.get(User, payload.user_id):
        raise HTTPException(status_code=404, detail="User not found")
    if not db.get(Item, payload.item_id):
        raise HTTPException(status_code=404, detail="Item not found")
    return interaction_service.ingest_event(db=db, payload=payload)


@router.get("/features/{user_id}")
def get_user_features(user_id: int, db: Session = Depends(get_db)) -> dict:
    """Get latest user online feature snapshot."""
    feature = db.get(UserFeature, user_id)
    if feature is None:
        raise HTTPException(status_code=404, detail="Feature snapshot not found")
    ctr = feature.click_count / feature.impression_count if feature.impression_count > 0 else 0.0
    return {
        "user_id": user_id,
        "click_count": feature.click_count,
        "impression_count": feature.impression_count,
        "watch_seconds": feature.watch_seconds,
        "ctr": round(ctr, 4),
        "last_active_at": feature.last_active_at.isoformat(),
    }


@router.post("/jobs/precompute")
def precompute_recommendations(
    n: int = Query(default=20, ge=5, le=100),
    limit_users: int = Query(default=200, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> dict:
    """Warm recommendation caches for active users and segment fallback."""
    return recommendation_service.precompute_for_active_users(
        db=db,
        top_n=n,
        limit_users=limit_users,
    )

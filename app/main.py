"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router
from app.db.database import Base, SessionLocal, engine
from app.services.logger import configure_logging
from app.services.recommendation_service import recommendation_service

configure_logging()

app = FastAPI(
    title="Production Recommendation System",
    description="Netflix/Amazon style recommendation backend with hybrid ML methods.",
    version="1.0.0",
)
app.include_router(router)


@app.on_event("startup")
def startup_event() -> None:
    """Initialize database and warm recommender."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        recommendation_service.rebuild(db)
    finally:
        db.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

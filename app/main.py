"""FastAPI application entry point."""

from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.db.database import Base, SessionLocal, engine
from app.services.logger import configure_logging
from app.services.metrics import REQUEST_COUNT, REQUEST_LATENCY, render_metrics
from app.services.recommendation_service import recommendation_service

configure_logging()

app = FastAPI(
    title="Production Recommendation System",
    description="Netflix/Amazon style recommendation backend with hybrid ML methods.",
    version="1.0.0",
)
app.include_router(router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    latency = perf_counter() - start
    endpoint = request.url.path
    method = request.method
    REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(latency)
    REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=str(response.status_code)).inc()
    return response


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


@app.get("/metrics", response_class=Response, response_model=None)
def metrics() -> Response:
    return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")


@app.get("/", response_model=None)
def frontend_home():
    """Serve frontend app if available."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend not found. Use API docs at /docs"}

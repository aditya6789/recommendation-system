# Production Recommendation System (FastAPI + Python ML)

A production-style recommendation backend inspired by Netflix/Amazon with:

- User-based collaborative filtering
- Item-based collaborative filtering
- Content-based filtering (genre/tags/description)
- Matrix factorization (SVD)
- Hybrid ranking model
- Cold-start handling for new users/items
- FastAPI API endpoints
- PostgreSQL via SQLAlchemy ORM
- Optional Redis caching
- Optional Streamlit UI for quick testing

## Project Structure

```text
app/
  main.py
  api/
    routes.py
  db/
    config.py
    database.py
  models/
    user.py
    item.py
    rating.py
  schemas/
    user.py
    item.py
    rating.py
    recommendation.py
  services/
    cache.py
    logger.py
    recommendation_service.py
  ml/
    recommender.py
scripts/
  load_sample_data.py
streamlit_app.py
requirements.txt
.env.example
README.md
```

## Setup

1. Create virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Create PostgreSQL database:

- DB name: `recommendation_db`
- Update `.env` from `.env.example` with your credentials.

3. (Optional) Start Redis and set:

- `ENABLE_REDIS_CACHE=true`
- `REDIS_URL=redis://localhost:6379/0`

## Database Migrations (Alembic)

Run all migrations:

```bash
alembic upgrade head
```

Create a new migration:

```bash
alembic revision -m "your change summary"
```

## Run Backend

```bash
uvicorn app.main:app --reload
```

API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Load Sample Data

```bash
python scripts/load_sample_data.py
```

## Run Streamlit UI (Optional)

```bash
streamlit run streamlit_app.py
```

## API Endpoints

- `POST /user` -> create user
- `POST /item` -> add item
- `POST /rate` -> rate item (upserts rating)
- `GET /recommend/{user_id}?n=10` -> hybrid Top-N recommendations
- `GET /similar/{item_id}?n=10` -> similar items
- `GET /experiments/summary` -> A/B impression summary and active model version
- `GET /experiments/performance` -> variant-wise CTR and watch conversion
- `GET /experiments/bandit` -> Thompson Sampling priors/posteriors
- `POST /events/interaction` -> ingest real-time behavior events
- `GET /features/{user_id}` -> online user feature snapshot
- `GET /metrics` -> Prometheus metrics
- `POST /jobs/precompute?n=20&limit_users=200` -> cache warmer job for active users

## ML Design

- Builds sparse user-item matrix from ratings.
- Applies time-decay on interactions to prioritize recent behavior.
- Uses confidence weighting by user interaction density.
- Mixes explicit rating with implicit signal for robust sparse learning.
- Computes cosine similarity:
  - User-user matrix for user-CF.
  - Item-item matrix for item-CF.
- Learns latent factors with `TruncatedSVD` for matrix factorization.
- Builds content vectors via TF-IDF from `genre + tags + description`.
- Hybrid score combines weighted components with dynamic user-aware weights.
- Uses 2-stage recommendation flow:
  - candidate generation (user-CF, item-CF, content, SVD)
  - re-ranking (novelty + popularity balance + diversity filter)
- Cold start:
  - New user: popular items fallback.
  - New item: content-based similarity fallback.
- Level 3 additions:
  - Model version registry (`model_versions`)
  - A/B assignment framework (`ab_assignments`) with `auto|v1|v2` strategy
  - Recommendation event logging (`recommendation_events`)
  - Offline evaluator (`scripts/evaluate_offline.py`)

### A/B Testing Quick Check

- `GET /recommend/1?n=10&strategy=auto` (assigned variant)
- `GET /recommend/1?n=10&strategy=v1` (control)
- `GET /recommend/1?n=10&strategy=v2` (advanced ranker)
- `GET /experiments/summary`

### Offline Evaluation

```bash
python scripts/evaluate_offline.py
```

## Phase 1 Real-Time Foundation

- Real-time interaction ingestion:
  - event types: `impression`, `click`, `watch`
- Online feature updates:
  - `click_count`, `impression_count`, `watch_seconds`, `last_active_at`
  - persisted in `user_features` + cached in Redis (when enabled)
- Kafka event streaming (optional):
  - set `ENABLE_KAFKA=true`
  - configure `KAFKA_BOOTSTRAP_SERVERS` and `KAFKA_TOPIC_EVENTS`
  - run consumer worker for stream-driven feature updates:
    - `python scripts/run_kafka_consumer.py`
- Observability:
  - Prometheus counters/histograms at `/metrics`

## Phase 2 Retrieval at Scale

- ANN/vector retrieval:
  - item embeddings indexed through `app/ml/vector_index.py`
  - FAISS used automatically if available, else sklearn fallback
- Candidate generation upgraded:
  - hybrid candidate pool now includes ANN retrieval candidates
- Similar items optimized:
  - `/similar/{item_id}` now prefers vector index neighbors first
- Precompute and warm cache:
  - `/jobs/precompute` warms per-user recommendation cache for active users
  - also builds segment fallback cache (`recommendations:segment:popular:*`)

## Phase 3 Ranking + Experiment Intelligence

- Learning-to-rank layer:
  - `app/services/ltr_ranker.py` (GradientBoostingRegressor)
  - trained on historical ratings
  - used to rerank `v2` candidate list before final response
- Ranker persistence:
  - saved at `artifacts/ltr_ranker.joblib`
  - loaded on service startup if available
- Advanced offline evaluation:
  - `scripts/evaluate_offline.py` now reports:
    - Precision@K
    - Recall@K
    - MAP@K
    - NDCG@K
    - Coverage@K
- Online KPI reporting:
  - `/experiments/performance` for variant metrics:
    - impressions
    - clicks
    - watches
    - CTR
    - watch conversion rate
- Adaptive experiment selection:
  - `strategy=auto` now supports Thompson Sampling when `ENABLE_BANDIT_AUTO=true`
  - posterior snapshot endpoint: `/experiments/bandit`

### Phase 3 Utility Scripts

```bash
python scripts/train_ranker.py
python scripts/evaluate_offline.py
```

## Docker Compose Stack

Run complete local stack (API + Postgres + Redis + Kafka + Prometheus + Grafana):

```bash
docker compose up --build
```

Detached mode:

```bash
docker compose up -d --build
```

Service URLs:

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

## Production Notes

- Move model rebuild to async/background workers for high write throughput.
- Add migration tooling (Alembic), CI tests, and auth for production deployment.
- Consider periodic batch precomputation and Redis warming for low-latency reads.
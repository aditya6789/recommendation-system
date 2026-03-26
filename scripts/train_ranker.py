"""Train and persist the LTR ranker model from ratings data."""

import os
import sys

import pandas as pd
from sqlalchemy import select

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.db.database import SessionLocal
from app.models import Rating
from app.services.ltr_ranker import ltr_ranker


def train_ranker() -> dict:
    db = SessionLocal()
    try:
        ratings = db.execute(select(Rating)).scalars().all()
        df = pd.DataFrame(
            [
                {"user_id": r.user_id, "item_id": r.item_id, "rating": r.rating, "created_at": r.created_at}
                for r in ratings
            ]
        )
        ok = ltr_ranker.fit(df)
        return {"trained": ok, "rows": len(df), "model_path": str(ltr_ranker.model_path)}
    finally:
        db.close()


if __name__ == "__main__":
    print(train_ranker())

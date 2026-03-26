"""User feature snapshot model for online/offline feature consistency."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class UserFeature(Base):
    """Aggregated user behavior features updated by interaction events."""

    __tablename__ = "user_features"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True, index=True)
    click_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    impression_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    watch_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

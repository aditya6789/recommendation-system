"""A/B experiment user assignment table."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class ABAssignment(Base):
    """Stores deterministic A/B bucket for each user and experiment."""

    __tablename__ = "ab_assignments"
    __table_args__ = (UniqueConstraint("user_id", "experiment_name", name="uq_user_experiment"),)

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    experiment_name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    variant: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

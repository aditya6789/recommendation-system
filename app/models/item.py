"""Item database model."""

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Item(Base):
    """Recommendable item such as a movie/product."""

    __tablename__ = "items"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    genre: Mapped[str] = mapped_column(String(120), nullable=True)
    tags: Mapped[str] = mapped_column(Text, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)

    ratings = relationship("Rating", back_populates="item", cascade="all, delete-orphan")

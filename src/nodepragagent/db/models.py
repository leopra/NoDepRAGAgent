"""SQLAlchemy declarative models for the transactional database."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import List

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


class Customer(Base):
    """Customer profile data."""

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    purchases: Mapped[List["Purchase"]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
    )


class Item(Base):
    """Catalog of products available for purchase."""

    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)

    purchases: Mapped[List["Purchase"]] = relationship(
        back_populates="item",
        cascade="all, delete-orphan",
    )


class Purchase(Base):
    """Link table capturing customer purchases."""

    __tablename__ = "purchases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    purchased_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    customer: Mapped[Customer] = relationship(back_populates="purchases")
    item: Mapped[Item] = relationship(back_populates="purchases")

"""SQLAlchemy declarative models for the transactional database."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

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
    addresses: Mapped[List["CustomerAddress"]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
    )


class Item(Base):
    """Catalog of products available for purchase."""

    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    category_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("categories.id", ondelete="SET NULL"),
        nullable=True,
    )

    purchases: Mapped[List["Purchase"]] = relationship(
        back_populates="item",
        cascade="all, delete-orphan",
    )
    category: Mapped[Optional["Category"]] = relationship(back_populates="items")
    supplier_links: Mapped[List["ItemSupplier"]] = relationship(
        back_populates="item",
        cascade="all, delete-orphan",
    )
    suppliers: Mapped[List["Supplier"]] = relationship(
        secondary="item_suppliers",
        back_populates="items",
    )


class Category(Base):
    """Grouping of catalog items."""

    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)

    items: Mapped[List[Item]] = relationship(back_populates="category")


class Supplier(Base):
    """Company supplying items to the catalog."""

    __tablename__ = "suppliers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    contact_email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    item_links: Mapped[List["ItemSupplier"]] = relationship(
        back_populates="supplier",
        cascade="all, delete-orphan",
    )
    items: Mapped[List[Item]] = relationship(
        secondary="item_suppliers",
        back_populates="suppliers",
    )


class ItemSupplier(Base):
    """Association table linking suppliers to items they provide."""

    __tablename__ = "item_suppliers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(
        ForeignKey("items.id", ondelete="CASCADE"),
        nullable=False,
    )
    supplier_id: Mapped[int] = mapped_column(
        ForeignKey("suppliers.id", ondelete="CASCADE"),
        nullable=False,
    )
    wholesale_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    lead_time_days: Mapped[int] = mapped_column(Integer, nullable=False, default=7)

    item: Mapped[Item] = relationship(
        back_populates="supplier_links",
        overlaps="suppliers,items",
    )
    supplier: Mapped[Supplier] = relationship(
        back_populates="item_links",
        overlaps="items,suppliers",
    )


class CustomerAddress(Base):
    """Postal addresses associated with a customer."""

    __tablename__ = "customer_addresses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
    )
    label: Mapped[str] = mapped_column(String(50), nullable=False)
    street: Mapped[str] = mapped_column(String(255), nullable=False)
    city: Mapped[str] = mapped_column(String(120), nullable=False)
    state: Mapped[str] = mapped_column(String(120), nullable=False)
    postal_code: Mapped[str] = mapped_column(String(20), nullable=False)
    country: Mapped[str] = mapped_column(String(120), nullable=False)

    customer: Mapped[Customer] = relationship(back_populates="addresses")
    purchases: Mapped[List["Purchase"]] = relationship(back_populates="shipping_address")


class Purchase(Base):
    """Link table capturing customer purchases."""

    __tablename__ = "purchases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), nullable=False
    )
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    purchased_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    shipping_address_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("customer_addresses.id", ondelete="SET NULL"),
        nullable=True,
    )

    customer: Mapped[Customer] = relationship(back_populates="purchases")
    item: Mapped[Item] = relationship(back_populates="purchases")
    shipping_address: Mapped[Optional[CustomerAddress]] = relationship(
        back_populates="purchases"
    )

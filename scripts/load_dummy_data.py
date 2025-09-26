"""Populate the SQL database with dummy customer, item, and purchase data."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, create_engine, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for ORM models."""


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    purchases: Mapped[list["Purchase"]] = relationship(back_populates="customer", cascade="all, delete-orphan")


class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)

    purchases: Mapped[list["Purchase"]] = relationship(back_populates="item", cascade="all, delete-orphan")


class Purchase(Base):
    __tablename__ = "purchases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    purchased_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    customer: Mapped[Customer] = relationship(back_populates="purchases")
    item: Mapped[Item] = relationship(back_populates="purchases")


def _ensure_sqlite_path(database_url: str) -> None:
    """Create parent directories for SQLite files so the engine can initialize."""

    if not database_url.startswith("sqlite"):
        return

    if database_url == "sqlite:///:memory:":
        return

    _, _, file_path = database_url.partition("///")
    if not file_path:
        return

    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_engine():
    database_url = os.getenv(
        "DATABASE_URL",
        "sqlite:///data/rag.db",
    )
    _ensure_sqlite_path(database_url)
    return create_engine(database_url, echo=False, future=True)


def load_dummy_data(session: Session) -> None:
    """Insert a deterministic set of customers, items, and purchase records."""

    # Start from a clean slate so repeated runs do not duplicate rows.
    session.execute(delete(Purchase))
    session.execute(delete(Customer))
    session.execute(delete(Item))

    customers = [
        Customer(name="Alice Johnson", email="alice@example.com"),
        Customer(name="Brian Lee", email="brian@example.com"),
        Customer(name="Carla Mendes", email="carla@example.com"),
    ]

    items = [
        Item(name="Wireless Mouse", price=Decimal("29.99")),
        Item(name="Mechanical Keyboard", price=Decimal("129.50")),
        Item(name="27-inch Monitor", price=Decimal("249.00")),
        Item(name="USB-C Hub", price=Decimal("59.95")),
    ]

    session.add_all([*customers, *items])
    session.flush()

    purchases = [
        Purchase(customer_id=customers[0].id, item_id=items[0].id, quantity=1, total_amount=items[0].price),
        Purchase(
            customer_id=customers[0].id,
            item_id=items[2].id,
            quantity=1,
            total_amount=items[2].price,
        ),
        Purchase(
            customer_id=customers[1].id,
            item_id=items[1].id,
            quantity=2,
            total_amount=items[1].price * 2,
        ),
        Purchase(
            customer_id=customers[1].id,
            item_id=items[3].id,
            quantity=1,
            total_amount=items[3].price,
        ),
        Purchase(
            customer_id=customers[2].id,
            item_id=items[0].id,
            quantity=3,
            total_amount=items[0].price * 3,
        ),
    ]

    session.add_all(purchases)
    session.commit()


def main() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        load_dummy_data(session)
    print("Dummy data loaded successfully.")


if __name__ == "__main__":
    main()

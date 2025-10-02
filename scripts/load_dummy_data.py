"""Populate the SQL database with dummy customer, item, and purchase data."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from nodepragagent.db.models import Base, Customer, Item, Purchase

from nodepragagent.tools import create_postgres_engine


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
        Purchase(
            customer_id=customers[0].id,
            item_id=items[0].id,
            quantity=1,
            total_amount=items[0].price,
        ),
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
    engine: Session.bind.__class__ | None = None  # type: ignore[attr-defined]
    try:
        engine = create_postgres_engine()
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            load_dummy_data(session)
    except SQLAlchemyError as exc:  # pragma: no cover - depends on external DB
        raise SystemExit(f"Failed to load dummy data: {exc}")
    finally:
        if engine is not None:
            engine.dispose()

    db_url = engine.url.render_as_string(hide_password=True)
    print(f"Dummy data loaded successfully into {db_url}.")


if __name__ == "__main__":
    main()

"""Populate the SQL database with dummy customer, item, and purchase data."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from nodepragagent.db.models import (
    Base,
    Category,
    Customer,
    CustomerAddress,
    Item,
    ItemSupplier,
    Purchase,
    Supplier,
)

from nodepragagent.tools import create_postgres_engine


def load_dummy_data(session: Session) -> None:
    """Insert a deterministic set of customers, items, and purchase records."""

    # Start from a clean slate so repeated runs do not duplicate rows.
    session.execute(delete(Purchase))
    session.execute(delete(ItemSupplier))
    session.execute(delete(CustomerAddress))
    session.execute(delete(Customer))
    session.execute(delete(Item))
    session.execute(delete(Supplier))
    session.execute(delete(Category))

    categories = [
        Category(name="Peripherals"),
        Category(name="Monitors"),
        Category(name="Accessories"),
    ]

    suppliers = [
        Supplier(name="Acme Distribution", contact_email="sales@acme.com"),
        Supplier(name="Brightline Wholesale", contact_email="hello@brightline.com"),
    ]

    customers = [
        Customer(name="Alice Johnson", email="alice@example.com"),
        Customer(name="Brian Lee", email="brian@example.com"),
        Customer(name="Carla Mendes", email="carla@example.com"),
    ]

    session.add_all([*categories, *suppliers, *customers])
    session.flush()

    addresses = [
        CustomerAddress(
            customer_id=customers[0].id,
            label="Home",
            street="123 Maple Street",
            city="Springfield",
            state="IL",
            postal_code="62704",
            country="USA",
        ),
        CustomerAddress(
            customer_id=customers[0].id,
            label="Office",
            street="1 Innovation Way",
            city="Chicago",
            state="IL",
            postal_code="60601",
            country="USA",
        ),
        CustomerAddress(
            customer_id=customers[1].id,
            label="Home",
            street="500 Ocean Avenue",
            city="San Francisco",
            state="CA",
            postal_code="94107",
            country="USA",
        ),
        CustomerAddress(
            customer_id=customers[2].id,
            label="Home",
            street="90 Greenway Plaza",
            city="Austin",
            state="TX",
            postal_code="73301",
            country="USA",
        ),
    ]

    items = [
        Item(
            name="Wireless Mouse",
            price=Decimal("29.99"),
            category_id=categories[0].id,
        ),
        Item(
            name="Mechanical Keyboard",
            price=Decimal("129.50"),
            category_id=categories[0].id,
        ),
        Item(
            name="27-inch Monitor",
            price=Decimal("249.00"),
            category_id=categories[1].id,
        ),
        Item(
            name="USB-C Hub",
            price=Decimal("59.95"),
            category_id=categories[2].id,
        ),
        Item(
            name="Noise-Cancelling Headset",
            price=Decimal("199.00"),
            category_id=categories[0].id,
        ),
    ]

    session.add_all([*addresses, *items])
    session.flush()

    item_suppliers = [
        ItemSupplier(
            item_id=items[0].id,
            supplier_id=suppliers[0].id,
            wholesale_price=Decimal("19.99"),
            lead_time_days=5,
        ),
        ItemSupplier(
            item_id=items[1].id,
            supplier_id=suppliers[0].id,
            wholesale_price=Decimal("95.00"),
            lead_time_days=10,
        ),
        ItemSupplier(
            item_id=items[2].id,
            supplier_id=suppliers[1].id,
            wholesale_price=Decimal("210.00"),
            lead_time_days=12,
        ),
        ItemSupplier(
            item_id=items[3].id,
            supplier_id=suppliers[1].id,
            wholesale_price=Decimal("42.50"),
            lead_time_days=7,
        ),
        ItemSupplier(
            item_id=items[4].id,
            supplier_id=suppliers[0].id,
            wholesale_price=Decimal("150.00"),
            lead_time_days=9,
        ),
        ItemSupplier(
            item_id=items[4].id,
            supplier_id=suppliers[1].id,
            wholesale_price=Decimal("155.00"),
            lead_time_days=6,
        ),
    ]

    session.add_all(item_suppliers)
    session.flush()

    purchases = [
        Purchase(
            customer_id=customers[0].id,
            item_id=items[0].id,
            quantity=1,
            total_amount=items[0].price,
            shipping_address_id=addresses[0].id,
        ),
        Purchase(
            customer_id=customers[0].id,
            item_id=items[2].id,
            quantity=1,
            total_amount=items[2].price,
            shipping_address_id=addresses[1].id,
        ),
        Purchase(
            customer_id=customers[1].id,
            item_id=items[1].id,
            quantity=2,
            total_amount=items[1].price * 2,
            shipping_address_id=addresses[2].id,
        ),
        Purchase(
            customer_id=customers[1].id,
            item_id=items[3].id,
            quantity=1,
            total_amount=items[3].price,
            shipping_address_id=addresses[2].id,
        ),
        Purchase(
            customer_id=customers[2].id,
            item_id=items[0].id,
            quantity=3,
            total_amount=items[0].price * 3,
            shipping_address_id=addresses[3].id,
        ),
        Purchase(
            customer_id=customers[2].id,
            item_id=items[4].id,
            quantity=1,
            total_amount=items[4].price,
            shipping_address_id=addresses[3].id,
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

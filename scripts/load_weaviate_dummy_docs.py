"""Populate the Weaviate instance with item and company context documents."""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import weaviate  # type: ignore[import-untyped]
from weaviate import AuthApiKey  # type: ignore[import-untyped]
from weaviate.exceptions import WeaviateBaseError

CLASS_NAME = "ProductInsight"


def _client() -> weaviate.Client:
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    api_key = os.getenv("WEAVIATE_API_KEY")
    auth = AuthApiKey(api_key) if api_key else None
    return weaviate.Client(url=url, auth_client_secret=auth)


def _class_exists(client: weaviate.Client, class_name: str) -> bool:
    schema = client.schema.get()
    classes: Iterable[dict] = schema.get("classes", []) or []
    return any(cls.get("class") == class_name for cls in classes)


def _reset_schema(client: weaviate.Client) -> None:
    if _class_exists(client, CLASS_NAME):
        client.schema.delete_class(CLASS_NAME)

    client.schema.create_class(
        {
            "class": CLASS_NAME,
            "description": "Product catalogue facts and company economic notes",
            "vectorizer": "none",
            "properties": [
                {"name": "title", "dataType": ["text"], "description": "Document title"},
                {
                    "name": "category",
                    "dataType": ["text"],
                    "description": "High-level grouping such as item or company",
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Full text with metrics and qualitative notes",
                },
            ],
        }
    )


def _documents() -> tuple[list[dict[str, str]], list[Sequence[float]]]:
    docs = [
        {
            "title": "Wireless Mouse Overview",
            "category": "item",
            "content": (
                "Wireless Mouse retail details: silent-scroll design, bundled USB receiver, "
                "inventory of 120 units. List price USD 29.99, margin target 35 percent."
            ),
        },
        {
            "title": "Mechanical Keyboard Sales",
            "category": "item",
            "content": (
                "Mechanical Keyboard with hot-swappable switches and RGB backlight. "
                "Premium accessory positioned at USD 129.50, typical basket attachment "
                "rate 18 percent in B2B accounts."
            ),
        },
        {
            "title": "27-inch Monitor Performance",
            "category": "item",
            "content": (
                "27-inch Monitor, 1440p IPS panel, warranty three years. Price point USD 249.00 "
                "supports bundles with docking stations, accessory sell-through 1.4 add-ons per sale."
            ),
        },
        {
            "title": "USB-C Hub Attachment",
            "category": "item",
            "content": (
                "USB-C Hub with eight ports, shipping with braided cable. MSRP USD 59.95, discountable "
                "in education verticals with minimum margin 22 percent."
            ),
        },
        {
            "title": "Quarterly Financial Snapshot",
            "category": "company",
            "content": (
                "Company posted Q2 revenue of USD 4.2M with gross margin 41 percent. Operating expenses "
                "trended flat quarter-over-quarter as marketing shifted toward digital campaigns. Cash "
                "reserves cover 14 months of runway even with continued R&D investment."
            ),
        },
    ]

    vectors: list[Sequence[float]] = [
        [0.12, 0.31, 0.08],
        [0.15, 0.28, 0.11],
        [0.18, 0.22, 0.14],
        [0.11, 0.34, 0.06],
        [0.42, 0.17, 0.29],
    ]
    return docs, vectors


def _load_documents(client: weaviate.Client) -> None:
    docs, vectors = _documents()
    client.batch.configure(batch_size=5)
    with client.batch as batch:  # type: ignore[attr-defined]
        for doc, vector in zip(docs, vectors, strict=True):
            batch.add_data_object(doc, CLASS_NAME, vector=vector)


def main() -> None:
    client = _client()
    try:
        _reset_schema(client)
        _load_documents(client)
    except WeaviateBaseError as exc:
        raise SystemExit(f"Failed to load documents: {exc}")
    print("Dummy documents loaded into Weaviate.")


if __name__ == "__main__":
    main()

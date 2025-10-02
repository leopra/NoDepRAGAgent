"""Populate the Weaviate instance with item and company context documents."""

from __future__ import annotations

import asyncio

import weaviate
from weaviate.exceptions import WeaviateBaseError

from nodepragagent.embeddings import embed_contents
from nodepragagent.tools import _weaviate_client
from nodepragagent.vllm import VLLMConfig

CLASS_NAME = "ProductInsight"


async def _reset_schema(client: weaviate.WeaviateAsyncClient) -> None:
    if await client.collections.exists(CLASS_NAME):
        await client.collections.delete(CLASS_NAME)

    await client.collections.create_from_dict(
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



def _documents() -> list[dict[str, str]]:
    return [
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


async def _load_documents(client: weaviate.WeaviateAsyncClient) -> None:
    docs = _documents()
    try:
        vectors = await embed_contents((doc["content"] for doc in docs), config=VLLMConfig())
    except Exception as exc:  # pragma: no cover - embedding request failure
        raise SystemExit(f"Failed to compute embeddings: {exc}")

    collection = client.collections.get(CLASS_NAME)
    for doc, vector in zip(docs, vectors, strict=True):
        await collection.data.insert(doc, vector=vector)


async def _main_async() -> None:
    client = _weaviate_client()
    try:
        await client.connect()
        await _reset_schema(client)
        await _load_documents(client)
    except WeaviateBaseError as exc:
        raise SystemExit(f"Failed to load documents: {exc}")
    finally:
        await client.close()
    print("Dummy documents loaded into Weaviate.")


if __name__ == "__main__":
    asyncio.run(_main_async())

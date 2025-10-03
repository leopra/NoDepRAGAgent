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
                "inventory of 120 units. List price USD 29.99, margin target 35 percent, "
                "and break-even volume currently projected at 82 units per month. Packaging refresh "
                "scheduled for Q3 to highlight recyclable materials and reduced plastic footprint."
            ),
        },
        {
            "title": "Mechanical Keyboard Sales",
            "category": "item",
            "content": (
                "Mechanical Keyboard with hot-swappable switches and RGB backlight. "
                "Premium accessory positioned at USD 129.50, typical basket attachment "
                "rate 18 percent in B2B accounts. Channel partners report a 12 percent upsell "
                "lift when bundled with ergonomic wrist rests, and supply chain indicates a "
                "two-week lead time for Kailh switch replenishment."
            ),
        },
        {
            "title": "27-inch Monitor Performance",
            "category": "item",
            "content": (
                "27-inch Monitor, 1440p IPS panel, warranty three years. Price point USD 249.00 "
                "supports bundles with docking stations, accessory sell-through 1.4 add-ons per sale. "
                "Customer feedback praises color calibration presets and notes desire for USB-C "
                "power delivery in future revisions. Returns sit below 1.8 percent thanks to on-site setup videos."
            ),
        },
        {
            "title": "USB-C Hub Attachment",
            "category": "item",
            "content": (
                "USB-C Hub with eight ports, shipping with braided cable. MSRP USD 59.95, discountable "
                "in education verticals with minimum margin 22 percent. Firmware includes automated load balancing "
                "to prevent brownouts on high-draw ports, and packaging inserts promote companion warranty subscriptions."
            ),
        },
        {
            "title": "Quarterly Financial Snapshot",
            "category": "company",
            "content": (
                "Company posted Q2 revenue of USD 4.2M with gross margin 41 percent. Operating expenses "
                "trended flat quarter-over-quarter as marketing shifted toward digital campaigns. Cash "
                "reserves cover 14 months of runway even with continued R&D investment. Board directives emphasize "
                "maintaining positive net retention above 118 percent while allocating 12 percent of revenue to innovation labs."
            ),
        },
        {
            "title": "Remote Work Policy Highlights",
            "category": "policy",
            "content": (
                "Remote-first staffing policy mandates core hours 10:00-15:00 GMT, ergonomic stipend USD 450 "
                "per teammate annually, and quarterly home office audits to ensure compliance with safety standards. "
                "Teams coordinate through asynchronous standups, and managers receive training on inclusive facilitation "
                "to reduce meeting overload and time zone friction."
            ),
        },
        {
            "title": "Sustainability Roadmap 2025",
            "category": "policy",
            "content": (
                "Blue Horizon Tech targets carbon-neutral operations by FY2025 through renewable energy contracts, "
                "vendor audits, and mandatory eco-design reviews for new hardware. KPIs include 30 percent waste "
                "reduction and ISO 14001 certification. Sustainability dashboard publishes quarterly updates, "
                "and procurement must prioritize suppliers with verified low-emission logistics."
            ),
        },
        {
            "title": "Customer Success Playbook",
            "category": "practice",
            "content": (
                "Three-tier account framework with proactive health scoring, executive business reviews on a 90-day "
                "cadence, and escalation matrix linking product, support, and finance for high-risk renewals. Playbook "
                "recommends quarterly voice-of-customer synthesis and mandates closing the loop on top-five feature requests."
            ),
        },
        {
            "title": "AI Enablement Program",
            "category": "practice",
            "content": (
                "Internal enablement tracks upskill teams on prompt design, ethical guardrails, and model observability. "
                "Pilot projects receive 8-week sandbox access and must publish impact retrospectives before scaling. "
                "Curriculum includes shadow reviews with the Ethical AI Council and a shared repository of reusable prompt patterns."
            ),
        },
        {
            "title": "Blue Horizon Business Plan 2024-2027",
            "category": "strategy",
            "content": (
                "Three-year growth plan prioritizes mid-market expansion, embedded analytics, and recurring revenue at 78 percent. "
                "Assumes Series C raise in Q1 2025 to fuel acquisitions of boutique AI workflow startups. Year-by-year milestones "
                "outline geographic expansion into DACH and APAC regions, coupled with product-led growth experiments."
            ),
        },
        {
            "title": "Data Governance Standards",
            "category": "policy",
            "content": (
                "Company maintains tiered data classification, quarterly SOC 2 type II audits, and zero-trust access with "
                "mandatory hardware keys. Product telemetry anonymized within 15 minutes under privacy-by-design charter. "
                "Incident response runbooks stored in a signed vault, and data stewards convene monthly to review anomalies."
            ),
        },
        {
            "title": "Sales Enablement Quarterly Goals",
            "category": "strategy",
            "content": (
                "Revenue operations to deliver 12 modular playbooks, refresh ROI calculators, and cut onboarding ramp by 25 percent. "
                "Regional pods trial blended coaching combining async libraries with live deal reviews. Additional KPIs track demo-to-close "
                "conversion improvements and require weekly insights sharing between product marketing and frontline sellers."
            ),
        },
        {
            "title": "People Operations Inclusion Guidelines",
            "category": "policy",
            "content": (
                "Hiring managers apply requirement libraries to eliminate biased language, conduct structured interviews, and "
                "publish promotion dossiers for transparency. Inclusion council owns quarterly sentiment pulse checks and "
                "sponsors mentorship pairings focused on cross-functional skill growth."
            ),
        },
        {
            "title": "Product Lifecycle Review",
            "category": "practice",
            "content": (
                "Every feature exits discovery only after dual-track validation, quantified user value scoring, and a live "
                "operability drill. End-of-life protocol includes 9-month notice and migration kits. Change advisory board reviews "
                "implementation risk, and learnings feed back into the opportunity backlog."
            ),
        },
        {
            "title": "Cyber Resilience Incident Drill",
            "category": "practice",
            "content": (
                "Semi-annual red team exercise covering credential stuffing, supply-chain injection, and backup restores. "
                "Post-mortems log mean-time-to-detect and board-level action items. Exercises conclude with tabletop simulations "
                "for executive stakeholders to refine communications protocols."
            ),
        },
        {
            "title": "Partner Ecosystem Charter",
            "category": "strategy",
            "content": (
                "Channel alliances segmented by solution depth with tiered MDF programs. Certifications demand co-marketing assets, "
                "shared telemetry, and renewal influence targets for strategic partners. Charter introduces quarterly partner councils "
                "to co-design roadmaps and align incentive structures."
            ),
        },
        {
            "title": "Ethical AI Council Mandate",
            "category": "policy",
            "content": (
                "Council meets monthly to review bias testing, user consent flows, and regulatory updates. Decisions require quorum of legal, "
                "data science, and customer advocacy leads with published minutes. Charter mandates scenario planning for upcoming legislation "
                "and tracks remediation commitments through an open dashboard."
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

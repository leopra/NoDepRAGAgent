"""Tool definitions for function calling with OpenAI-compatible clients."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

import weaviate  # type: ignore[import-untyped]
from weaviate import AuthApiKey  # type: ignore[import-untyped]
from weaviate.exceptions import WeaviateBaseError

from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params import FunctionDefinition

from .db import create_postgres_engine

WEAVIATE_CLASS = "ProductInsight"


@dataclass(frozen=True)
class Tool:
    """Combine an executable callable with its OpenAI function definition."""

    name: str
    description: str
    parameters: Dict[str, Any]
    callback: Callable[..., Dict[str, Any]]

    def to_openai_tool(self) -> ChatCompletionFunctionToolParam:
        """Return the ChatCompletions tool specification for this tool."""

        return ChatCompletionFunctionToolParam(
            type="function",
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=self.parameters,
            ),
        )


def _semantic_score(additional: Dict[str, Any]) -> Optional[float]:
    """Translate Weaviate `_additional` metadata into a relevance score."""

    certainty = additional.get("certainty")
    if certainty is not None:
        try:
            return float(certainty)
        except (TypeError, ValueError):
            return None

    distance = additional.get("distance")
    if distance is not None:
        try:
            return 1.0 - float(distance)
        except (TypeError, ValueError):
            return None

    return None


def sum_two_numbers(*, a: float, b: float) -> Dict[str, Any]:
    """Compute the sum of two numbers in a tool-call-friendly format."""

    return {"total": a + b}

def _postgres_engine() -> Engine:
    """Create a SQLAlchemy engine for Postgres using environment variables."""

    return create_postgres_engine()


def query_postgres(*, sql: str, limit: int = 50) -> Dict[str, Any]:
    """Run a SQL statement and return a JSON-serializable payload."""
    # TODO query sanification

    sql = sql.strip()
    if not sql:
        return {"rows": [], "error": "SQL statement must not be empty."}

    capped_limit = max(1, min(limit, 200))

    try:
        engine = _postgres_engine()
    except ModuleNotFoundError as exc:  # pragma: no cover - driver missing in runtime env
        return {"rows": [], "error": f"Missing Postgres driver: {exc}"}
    except Exception as exc:  # pragma: no cover - unexpected configuration problems
        return {"rows": [], "error": str(exc)}

    try:
        with engine.begin() as connection:
            statement = text(sql)
            result = connection.execute(statement)
            if not result.returns_rows:
                return {"rows": [], "rowcount": result.rowcount}

            rows = []
            truncated = False
            for idx, row in enumerate(result):
                if idx >= capped_limit:
                    truncated = True
                    break
                rows.append(dict(row._mapping))
            payload: Dict[str, Any] = {"rows": rows}
            if truncated:
                payload["truncated"] = True
            return payload
    except SQLAlchemyError as exc:
        return {"rows": [], "error": str(exc)}
    except Exception as exc:  # pragma: no cover - unexpected errors
        return {"rows": [], "error": str(exc)}


def _weaviate_client() -> weaviate.Client:
    """Construct a Weaviate client from environment variables."""

    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    api_key = os.getenv("WEAVIATE_API_KEY")
    auth = AuthApiKey(api_key) if api_key else None
    return weaviate.Client(url=url, auth_client_secret=auth)


def query_weaviate(*, query: str, limit: int = 3, category: Optional[str] = None) -> Dict[str, Any]:
    """Query Weaviate for documents related to the supplied query string."""

    normalized_limit = max(1, min(limit, 10))
    query = query.strip()
    if not query:
        return {"results": [], "error": "Query string must not be empty."}

    client = _weaviate_client()
    where_filter: Optional[Dict[str, Any]] = None
    if category:
        normalized_category = category.strip()
        if normalized_category:
            where_filter = {
                "path": ["category"],
                "operator": "Equal",
                "valueText": normalized_category,
            }

    try:
        weaviate_query = (
            client.query.get(WEAVIATE_CLASS, ["title", "category", "content"])
            .with_near_text({"concepts": [query]})
            .with_limit(normalized_limit)
            .with_additional(["distance", "certainty"])
        )

        if where_filter is not None:
            weaviate_query = weaviate_query.with_where(where_filter)

        response = weaviate_query.do()
    except WeaviateBaseError as exc:  # pragma: no cover - network errors
        return {"results": [], "error": str(exc)}
    except Exception as exc:  # pragma: no cover - unexpected errors
        return {"results": [], "error": str(exc)}

    hits: List[Dict[str, Any]] = (
        response.get("data", {})
        .get("Get", {})
        .get(WEAVIATE_CLASS, [])
    )

    documents = []
    for hit in hits:
        additional = hit.get("_additional", {})
        documents.append(
            {
                "title": hit.get("title"),
                "category": hit.get("category"),
                "content": hit.get("content"),
                "score": _semantic_score(additional),
                "distance": additional.get("distance"),
                "certainty": additional.get("certainty"),
            }
        )

    return {"results": documents}


SUM_TWO_NUMBERS_TOOL = Tool(
    name="sum_two_numbers",
    description="Add two numeric values and return their total.",
    parameters={
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "First addend.",
            },
            "b": {
                "type": "number",
                "description": "Second addend.",
            },
        },
        "required": ["a", "b"],
    },
    callback=sum_two_numbers,
)

QUERY_POSTGRES_TOOL = Tool(
    name="query_postgres",
    description="Execute a SQL query against the Postgres operational database.",
    parameters={
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "SQL statement to run; prefer read-only SELECT queries.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of rows to return (default 50).",
                "minimum": 1,
                "maximum": 200,
            },
        },
        "required": ["sql"],
    },
    callback=query_postgres,
)

QUERY_WEAVIATE_TOOL = Tool(
    name="query_weaviate",
    description=(
        "Retrieve the most relevant product or company insight documents from the Weaviate"
        " vector database."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query describing the needed information.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of documents to return (default 3).",
                "minimum": 1,
                "maximum": 10,
            },
            "category": {
                "type": "string",
                "description": (
                    "Optional category filter; only documents with this category will be returned."
                ),
            },
        },
        "required": ["query"],
    },
    callback=query_weaviate,
)

ALL_TOOLS: tuple[Tool, ...] = (
    SUM_TWO_NUMBERS_TOOL,
    QUERY_POSTGRES_TOOL,
    QUERY_WEAVIATE_TOOL,
)

TOOLS: dict[str, Callable[..., Dict[str, Any]]] = {
    tool.name: tool.callback for tool in ALL_TOOLS
}

OPENAI_CHAT_TOOLS: tuple[ChatCompletionFunctionToolParam, ...] = tuple(
    tool.to_openai_tool() for tool in ALL_TOOLS
)

"""Tool definitions for function calling with OpenAI-compatible clients."""

from __future__ import annotations

import os
from typing import Any, Dict, List

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


def sum_two_numbers_definition() -> FunctionDefinition:
    """Return the JSON schema definition for a two-number summation tool."""

    return {
        "name": "sum_two_numbers",
        "description": "Add two numeric values and return their total.",
        "parameters": {
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
    }


def sum_two_numbers(*, a: float, b: float) -> Dict[str, Any]:
    """Compute the sum of two numbers in a tool-call-friendly format."""

    return {"total": a + b}


def sum_two_numbers_tool() -> ChatCompletionFunctionToolParam:
    """Return a chat-completions tool specification for the sum helper."""

    return {
        "type": "function",
        "function": sum_two_numbers_definition(),
    }


def postgres_query_definition() -> FunctionDefinition:
    """JSON schema definition for the Postgres SQL execution tool."""

    return {
        "name": "query_postgres",
        "description": "Execute a SQL query against the Postgres operational database.",
        "parameters": {
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
    }


def _postgres_engine() -> Engine:
    """Create a SQLAlchemy engine for Postgres using environment variables."""

    return create_postgres_engine()


def query_postgres(*, sql: str, limit: int = 50) -> Dict[str, Any]:
    """Run a SQL statement and return a JSON-serializable payload."""
    #TODO query sanification

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


def query_postgres_tool() -> ChatCompletionFunctionToolParam:
    """Tool specification for running SQL queries against Postgres."""

    return {
        "type": "function",
        "function": postgres_query_definition(),
    }


def weaviate_query_definition() -> FunctionDefinition:
    """JSON schema definition for the Weaviate document retrieval tool."""

    return {
        "name": "query_weaviate",
        "description": (
            "Retrieve the most relevant product or company insight documents from the Weaviate"
            " vector database."
        ),
        "parameters": {
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
            },
            "required": ["query"],
        },
    }


def _weaviate_client() -> weaviate.Client:
    """Construct a Weaviate client from environment variables."""

    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    api_key = os.getenv("WEAVIATE_API_KEY")
    auth = AuthApiKey(api_key) if api_key else None
    return weaviate.Client(url=url, auth_client_secret=auth)


def query_weaviate(*, query: str, limit: int = 3) -> Dict[str, Any]:
    """Query Weaviate for documents related to the supplied query string."""

    normalized_limit = max(1, min(limit, 10))
    query = query.strip()
    if not query:
        return {"results": [], "error": "Query string must not be empty."}

    client = _weaviate_client()
    try:
        response = (
            client.query.get(WEAVIATE_CLASS, ["title", "category", "content"])
            .with_bm25(query=query, properties=["title", "content"])
            .with_limit(normalized_limit)
            .with_additional(["score"])
            .do()
        )
    except WeaviateBaseError as exc:  # pragma: no cover - network errors
        return {"results": [], "error": str(exc)}
    except Exception as exc:  # pragma: no cover - unexpected errors
        return {"results": [], "error": str(exc)}

    hits: List[Dict[str, Any]] = (
        response.get("data", {})
        .get("Get", {})
        .get(WEAVIATE_CLASS, [])
    )

    documents = [
        {
            "title": hit.get("title"),
            "category": hit.get("category"),
            "content": hit.get("content"),
            "score": hit.get("_additional", {}).get("score"),
        }
        for hit in hits
    ]

    return {"results": documents}


def query_weaviate_tool() -> ChatCompletionFunctionToolParam:
    """Chat-completions tool specification for the Weaviate retrieval helper."""

    return {
        "type": "function",
        "function": weaviate_query_definition(),
    }

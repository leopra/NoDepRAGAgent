"""Tool definitions for function calling with OpenAI-compatible clients."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Awaitable, Dict, List, Optional, Protocol, Union

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from weaviate.connect import ConnectionParams  # type: ignore[import-not-found]

import weaviate  # type: ignore[import-untyped]
from weaviate.collections.classes.filters import Filter  # type: ignore[attr-defined]
from weaviate.collections.classes.grpc import MetadataQuery  # type: ignore[attr-defined]
from weaviate.exceptions import WeaviateBaseError

from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params import FunctionDefinition
from .db import create_postgres_engine
from .embeddings import embed_contents

WEAVIATE_CLASS = "ProductInsight"


ToolResponse = Dict[str, Any]


class ToolCallback(Protocol):
    """Callable protocol that supports synchronous or asynchronous execution."""

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Union[ToolResponse, Awaitable[ToolResponse]]: ...


@dataclass(frozen=True)
class Tool:
    """Combine an executable callable with its OpenAI function definition."""

    name: str
    description: str
    parameters: Dict[str, Any]
    callback: ToolCallback

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


def final_answer(*, answer: str, sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return the final answer payload that should be surfaced to the user."""

    payload: Dict[str, Any] = {"answer": answer}
    if sources is not None:
        payload["sources"] = sources
    return payload


def request_user_input(*, prompt: str, allow_empty: bool = False) -> Dict[str, Any]:
    """Prompt the end user for additional input and return their response."""

    question = prompt.strip() or "Please provide additional information:"

    if not sys.stdin.isatty():
        return {
            "response": None,
            "error": "Interactive input is unavailable in the current execution context.",
        }

    try:
        answer = input(f"{question}\n> ")
    except EOFError:
        return {
            "response": None,
            "error": "Input stream closed while waiting for user response.",
        }
    except KeyboardInterrupt:
        return {
            "response": None,
            "error": "User cancelled the input request.",
        }

    if not allow_empty and not answer.strip():
        return {
            "response": answer,
            "error": "User response was empty.",
        }

    return {"response": answer}


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
    finally:
        engine.dispose()


def _weaviate_client() -> weaviate.WeaviateAsyncClient:
    """Construct a Weaviate client from environment variables, REST-only (no gRPC)."""

    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    connection_params = ConnectionParams.from_url(url, grpc_port=50051)

    return weaviate.WeaviateAsyncClient(
        connection_params=connection_params,
        auth_client_secret=None,
        skip_init_checks=True,
    )


async def query_weaviate(
    *, query: str, limit: int = 3, category: Optional[str] = None
) -> Dict[str, Any]:
    """Query Weaviate for documents related to the supplied query string."""

    normalized_limit = max(1, min(limit, 10))
    query = query.strip()
    if not query:
        return {"results": [], "error": "Query string must not be empty."}

    client = _weaviate_client()

    try:
        query_vector = (await embed_contents([query]))[0]
    except Exception as exc:
        return {"results": [], "error": f"Failed to embed query: {exc}"}

    try:
        await client.connect()
    except Exception as exc:  # pragma: no cover - connection issues
        return {"results": [], "error": f"Failed to connect to Weaviate: {exc}"}

    filters = None
    # TODO disabled category for now
    # if category:
    #     normalized_category = category.strip()
    #     if normalized_category:
    #         filters = Filter.by_property("category").equal(normalized_category)

    try:
        collection = client.collections.get(WEAVIATE_CLASS)
        query_result = await collection.query.near_vector(  # type: ignore[attr-defined]
            near_vector=query_vector,
            limit=normalized_limit,
            filters=filters,
            return_properties=["title", "category", "content"],
            return_metadata=MetadataQuery(distance=True, certainty=True),
        )
    except WeaviateBaseError as exc:  # pragma: no cover - network errors
        return {"results": [], "error": str(exc)}
    except Exception as exc:  # pragma: no cover - unexpected errors
        return {"results": [], "error": str(exc)}
    finally:
        await client.close()

    documents = []
    for obj in getattr(query_result, "objects", []):
        properties = getattr(obj, "properties", {}) or {}
        metadata = getattr(obj, "metadata", None)
        additional = {
            "certainty": getattr(metadata, "certainty", None),
        }
        certainty = additional["certainty"]
        if certainty is not None:
            try:
                certainty = round(float(certainty), 2)
            except (TypeError, ValueError):
                certainty = additional["certainty"]
        documents.append(
            {
                "title": properties.get("title"),
                "category": properties.get("category"),
                "content": properties.get("content"),
                "certainty": certainty,
            }
        )

    return {"results": documents}



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

FINAL_ANSWER_TOOL_NAME = "final_answer"

FINAL_ANSWER_TOOL = Tool(
    name=FINAL_ANSWER_TOOL_NAME,
    description="Return the final answer to the user and stop further tool usage.",
    parameters={
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Final response that should be relayed to the user.",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of references or citations supporting the answer.",
            },
        },
        "required": ["answer"],
    },
    callback=final_answer,
)

REQUEST_USER_INPUT_TOOL = Tool(
    name="request_user_input",
    description=(
        "Ask the end user for additional information when the current context is insufficient "
        "to continue."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Question that should be shown directly to the user.",
            },
            "allow_empty": {
                "type": "boolean",
                "description": "Allow an empty response to be treated as valid input.",
                "default": False,
            },
        },
        "required": ["prompt"],
    },
    callback=request_user_input,
)

ALL_TOOLS: tuple[Tool, ...] = (
    QUERY_POSTGRES_TOOL,
    QUERY_WEAVIATE_TOOL,
    FINAL_ANSWER_TOOL,
    REQUEST_USER_INPUT_TOOL,
)

TOOLS: dict[str, ToolCallback] = {tool.name: tool.callback for tool in ALL_TOOLS}

OPENAI_CHAT_TOOLS: tuple[ChatCompletionFunctionToolParam, ...] = tuple(
    tool.to_openai_tool() for tool in ALL_TOOLS
)

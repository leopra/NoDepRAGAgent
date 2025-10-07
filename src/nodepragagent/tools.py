"""Tool definitions for function calling with OpenAI-compatible clients."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Awaitable, Dict, List, Optional, Protocol, Union

from pydantic import BaseModel, Field, ConfigDict

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from weaviate.connect import ConnectionParams

import weaviate 
from weaviate.collections.classes.filters import Filter 
from weaviate.collections.classes.grpc import MetadataQuery 
from weaviate.exceptions import WeaviateBaseError

from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params import FunctionDefinition
from .db import Base, create_postgres_engine
from .embeddings import embed_contents
from .errors import LLMError
from .utils import serialize_schema

WEAVIATE_CLASS = "ProductInsight"
POSTGRES_SCHEMA = serialize_schema(Base())

ToolResponse = Union[Dict[str, Any], BaseModel]


class QueryPostgresResult(BaseModel):
    """Structured payload returned from the SQL tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rows: List[Dict[str, Any]] = Field(default_factory=list)
    rowcount: Optional[int] = None
    truncated: Optional[bool] = None
    error: Optional[LLMError] = None


class WeaviateDocument(BaseModel):
    """Single document returned from Weaviate."""

    title: Optional[str] = None
    category: Optional[str] = None
    content: Optional[str] = None
    certainty: Optional[float] = None


class QueryWeaviateResult(BaseModel):
    """Structured payload returned from the Weaviate tool."""

    results: List[WeaviateDocument] = Field(default_factory=list)
    error: Optional[LLMError] = None


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
    args_model: type[BaseModel]
    callback: ToolCallback

    @property
    def parameters(self) -> Dict[str, Any]:
        """Return the OpenAPI-compatible JSON schema for the tool parameters."""

        return self.args_model.model_json_schema()

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
            "error": LLMError(
                reason="input_unavailable",
                message="Interactive input is unavailable in the current execution context.",
            ).as_dict(),
        }

    try:
        answer = input(f"{question}\n> ")
    except EOFError:
        return {
            "response": None,
            "error": LLMError(
                reason="input_stream_closed",
                message="Input stream closed while waiting for user response.",
            ).as_dict(),
        }
    except KeyboardInterrupt:
        return {
            "response": None,
            "error": LLMError(
                reason="input_cancelled",
                message="User cancelled the input request.",
            ).as_dict(),
        }

    if not allow_empty and not answer.strip():
        return {
            "response": answer,
            "error": LLMError(
                reason="empty_response",
                message="User response was empty.",
            ).as_dict(),
        }

    return {"response": answer}


def _postgres_engine() -> Engine:
    """Create a SQLAlchemy engine for Postgres using environment variables."""

    return create_postgres_engine()


def query_postgres(*, sql: str, limit: int = 50) -> QueryPostgresResult:
    """Run a SQL statement and return a JSON-serializable payload."""
    # TODO query sanification

    sql = sql.strip()
    if not sql:
        return QueryPostgresResult(
            rows=[],
            error=LLMError(
                reason="invalid_sql",
                message="SQL statement must not be empty.",
            ),
        )

    capped_limit = max(1, min(limit, 200))

    engine: Engine | None = None
    try:
        engine = _postgres_engine()
    except Exception as exc:
        return QueryPostgresResult(
            rows=[],
            error=LLMError(
                reason="engine_initialization_failed",
                message=str(exc),
            ),
        )

    try:
        with engine.begin() as connection:
            statement = text(sql)
            result = connection.execute(statement)
            if not result.returns_rows:
                return QueryPostgresResult(rows=[], rowcount=result.rowcount)

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
            return QueryPostgresResult(**payload)
    except SQLAlchemyError as exc:
        return QueryPostgresResult(
            rows=[],
            error=LLMError(
                reason="sql_error",
                message=str(exc),
            ),
        )
    except Exception as exc:
        return QueryPostgresResult(
            rows=[],
            error=LLMError(
                reason="unexpected_error",
                message=str(exc),
            ),
        )
    finally:
        if engine is not None:
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
) -> QueryWeaviateResult:
    """Query Weaviate for documents related to the supplied query string."""

    normalized_limit = max(1, min(limit, 10))
    query = query.strip()
    if not query:
        return QueryWeaviateResult(
            results=[],
            error=LLMError(
                reason="invalid_query",
                message="Query string must not be empty.",
            ),
        )

    client = _weaviate_client()

    try:
        query_vector = (await embed_contents([query]))[0]
    except Exception as exc:
        return QueryWeaviateResult(
            results=[],
            error=LLMError(
                reason="embedding_failure",
                message="Failed to embed query.",
                details=str(exc),
            ),
        )

    try:
        await client.connect()
    except Exception as exc:  # pragma: no cover - connection issues
        return QueryWeaviateResult(
            results=[],
            error=LLMError(
                reason="connection_failure",
                message="Failed to connect to Weaviate.",
                details=str(exc),
            ),
        )

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
    except WeaviateBaseError as exc:
        return QueryWeaviateResult(
            results=[],
            error=LLMError(
                reason="weaviate_error",
                message=str(exc),
            ),
        )
    except Exception as exc:
        return QueryWeaviateResult(
            results=[],
            error=LLMError(
                reason="unexpected_error",
                message=str(exc),
            ),
        )
    finally:
        await client.close()

    documents = []
    try:
        objects = query_result.objects
    except AttributeError:
        objects = []

    for obj in objects or []:
        try:
            properties = obj.properties
        except AttributeError:
            properties = {}
        properties = properties or {}

        try:
            metadata = obj.metadata
        except AttributeError:
            metadata = None

        if isinstance(metadata, dict):
            certainty_value = metadata.get("certainty")
        elif metadata is None:
            certainty_value = None
        else:
            try:
                certainty_value = metadata.certainty
            except AttributeError:
                certainty_value = None

        additional = {"certainty": certainty_value}
        certainty = additional["certainty"]
        if certainty is not None:
            try:
                certainty = round(float(certainty), 2)
            except (TypeError, ValueError):
                certainty = additional["certainty"]
        documents.append(
            WeaviateDocument(
                title=properties.get("title"),
                category=properties.get("category"),
                content=properties.get("content"),
                certainty=certainty,
            )
        )

    return QueryWeaviateResult(results=documents)



class QueryPostgresArgs(BaseModel):
    sql: str = Field(
        ...,
        min_length=1,
        description="SQL statement to run; prefer read-only SELECT queries.",
    )
    limit: int = Field(
        50,
        ge=1,
        le=200,
        description="Maximum number of rows to return (default 50).",
    )


QUERY_POSTGRES_TOOL = Tool(
    name="query_postgres",
    description="Execute a SQL query against the Postgres operational database.",
    args_model=QueryPostgresArgs,
    callback=query_postgres,
)


class QueryWeaviateArgs(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Natural language query describing the needed information.",
    )
    limit: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum number of documents to return (default 3).",
    )
    category: Optional[str] = Field(
        None,
        description="Optional category filter; only documents with this category will be returned.",
    )


QUERY_WEAVIATE_TOOL = Tool(
    name="query_weaviate",
    description=(
        "Retrieve the most relevant product or company insight documents from the Weaviate"
        " vector database."
    ),
    args_model=QueryWeaviateArgs,
    callback=query_weaviate,
)



class FinalAnswerArgs(BaseModel):
    answer: str = Field(
        ...,
        description="Final response that should be relayed to the user.",
    )
    sources: Optional[List[Union[str, Dict[str, Any]]]] = Field(
        default=None,
        description="Optional list of references or citations supporting the answer.",
    )


FINAL_ANSWER_TOOL_NAME = "final_answer"
FINAL_ANSWER_TOOL = Tool(
    name=FINAL_ANSWER_TOOL_NAME,
    description="Return the final answer to the user and stop further tool usage.",
    args_model=FinalAnswerArgs,
    callback=final_answer,
)


class RequestUserInputArgs(BaseModel):
    prompt: str = Field(
        ...,
        description="Question that should be shown directly to the user.",
    )
    allow_empty: bool = Field(
        False,
        description="Allow an empty response to be treated as valid input.",
    )


REQUEST_USER_INPUT_TOOL = Tool(
    name="request_user_input",
    description=(
        "Ask the end user for additional information when the current context is insufficient "
        "to continue."
    ),
    args_model=RequestUserInputArgs,
    callback=request_user_input,
)

ALL_TOOLS: tuple[Tool, ...] = (
    QUERY_POSTGRES_TOOL,
    QUERY_WEAVIATE_TOOL,
    REQUEST_USER_INPUT_TOOL,
)

TOOLS: dict[str, Tool] = {tool.name: tool for tool in ALL_TOOLS}

OPENAI_CHAT_TOOLS: list[ChatCompletionFunctionToolParam] = [
    tool.to_openai_tool() for tool in ALL_TOOLS]

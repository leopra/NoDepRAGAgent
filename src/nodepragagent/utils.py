import json
from enum import Enum
from typing import Any, Dict

from sqlalchemy.inspection import inspect
from sqlalchemy.orm import DeclarativeBase
from openai.types.chat import ChatCompletionFunctionToolParam


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ReporterEvent(str, Enum):
    USER_MESSAGE = "user_message"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"

def get_tool_name(tool: ChatCompletionFunctionToolParam) -> str:
    if hasattr(tool, "function") and hasattr(tool.function, "name"):
        return tool.function.name
    elif hasattr(tool, "name"):
        return tool.name
    else:
        return ""

def _format_payload(payload: object) -> str:
    """Render tool arguments or responses for CLI output."""

    if isinstance(payload, str):
        return payload

    try:
        return json.dumps(payload, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return str(payload)


def cli_event_printer(event: ReporterEvent, payload: dict[str, Any]) -> None:
    """Print VLLM client events in a human-friendly format."""

    iteration = payload.get("iteration")
    prefix = f"[iter {iteration}] " if iteration is not None else ""

    if event is ReporterEvent.USER_MESSAGE:
        return
    if event is ReporterEvent.MODEL_REQUEST:
        print(f"{prefix}-> calling model")
    elif event is ReporterEvent.MODEL_RESPONSE:
        content = payload.get("content", "")
        print(f"{prefix}Model> {content}")
    elif event is ReporterEvent.TOOL_CALL:
        tool_name = payload.get("tool_name", "unknown")
        args = _format_payload(payload.get("arguments"))
        tool_id = payload.get("tool_call_id")
        suffix = f" (id: {tool_id})" if tool_id else ""
        print(f"{prefix}Tool> {tool_name}{suffix}\n{args}")
    elif event is ReporterEvent.TOOL_RESULT:
        tool_name = payload.get("tool_name", "unknown")
        if tool_name == "final_answer":
            return
        result = _format_payload(payload.get("response"))
        print(f"{prefix}Tool< {tool_name}\n{result}")
    elif event is ReporterEvent.REASONING:
        reasoning = payload.get("response_reasoning")
        if reasoning:
            formatted_reasoning = _format_payload(reasoning)
            print(f"{prefix}<-- model reasoning\n{formatted_reasoning}")


def serialize_schema(base_model: DeclarativeBase) -> Dict[str, Dict[str, Any]]:
    """Convert SQLAlchemy ORM metadata into a JSON-friendly schema."""

    schema: Dict[str, Dict[str, Any]] = {}
    for mapper in base_model.registry.mappers:
        model = mapper.class_
        table_info: Dict[str, Any] = {
            "table_name": model.__tablename__,
            "columns": [],
            "relationships": [],
        }
        inspection = inspect(model)

        for column in inspection.columns:
            table_info["columns"].append(
                {
                    "name": column.key,
                    "type": str(column.type),
                    "primary_key": column.primary_key,
                    "nullable": column.nullable,
                    "default": str(column.default.arg) if column.default else None,
                }
            )

        for relationship in inspection.relationships:
            table_info["relationships"].append(
                {
                    "name": relationship.key,
                    "target": relationship.mapper.class_.__name__,
                    "uselist": relationship.uselist,
                    "direction": str(relationship.direction),
                    "back_populates": relationship.back_populates,
                }
            )

        schema[model.__name__] = table_info

    return schema

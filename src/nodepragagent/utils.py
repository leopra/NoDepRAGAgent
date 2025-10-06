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

def get_tool_name(tool: ChatCompletionFunctionToolParam) -> str:
    return getattr(tool, "function", tool).name if hasattr(tool, "function") else getattr(tool, "name", "")

def _format_payload(payload: object) -> str:
    """Render tool arguments or responses for CLI output."""

    if isinstance(payload, str):
        return payload

    try:
        return json.dumps(payload, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return str(payload)


def cli_event_printer(event: str, payload: dict[str, Any]) -> None:
    """Print VLLM client events in a human-friendly format."""

    iteration = payload.get("iteration")
    prefix = f"[iter {iteration}] " if iteration is not None else ""

    if event == "user_message":
        return
    if event == "model_request":
        tools = payload.get("tools") or []
        print(f"{prefix}-> calling model")
    elif event == "model_response":
        content = payload.get("content", "")
        print(f"{prefix}Model> {content}")
    elif event == "tool_call":
        tool_name = payload.get("tool_name", "unknown")
        args = _format_payload(payload.get("arguments"))
        tool_id = payload.get("tool_call_id")
        suffix = f" (id: {tool_id})" if tool_id else ""
        print(f"{prefix}Tool> {tool_name}{suffix}\n{args}")
    elif event == "tool_result":
        tool_name = payload.get("tool_name", "unknown")
        if tool_name == "final_answer":
            return
        result = _format_payload(payload.get("response"))
        print(f"{prefix}Tool< {tool_name}\n{result}")
    elif event == "model_response_received":
        response_id = payload.get("response_id")
        if response_id:
            print(f"{prefix}<-- model response id: {response_id}")
        reasoning = payload.get("response_reasoning")
        if reasoning:
            formatted_reasoning = _format_payload(reasoning)
            print(f"{prefix}<-- model reasoning\n{formatted_reasoning}")
    elif event == "max_iterations_reached":
        print("[warn] Maximum iterations reached without final answer")


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

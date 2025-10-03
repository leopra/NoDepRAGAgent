import json
from enum import Enum
from typing import Any

from openai.types.chat import ChatCompletionFunctionToolParam


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


def _tool_name(tool: ChatCompletionFunctionToolParam) -> str:
    """Best-effort extraction of a tool name from the OpenAI tool descriptor."""

    function = getattr(tool, "function", None)
    if function is not None and hasattr(function, "name"):
        name = getattr(function, "name", "")
        if isinstance(name, str):
            return name

    if isinstance(tool, dict):
        function_dict = tool.get("function")
        if isinstance(function_dict, dict):
            name = function_dict.get("name")
            if isinstance(name, str):
                return name

    return ""


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
        if isinstance(tools, (list, tuple)):
            tool_list = ", ".join(str(tool) for tool in tools)
        else:
            tool_list = str(tools)
        print(f"{prefix}-> calling model with tools: {tool_list}")
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

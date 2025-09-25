"""Tool definitions for function calling with OpenAI-compatible clients."""

from __future__ import annotations

from typing import Any, Dict

from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params import FunctionDefinition


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
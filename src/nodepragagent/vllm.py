"""Utilities for calling a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Sequence

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
)

from .db import schema_summary
from .memory import ToolCall
from .tools import FINAL_ANSWER_TOOL, OPENAI_CHAT_TOOLS, TOOLS
from .utils import MessageRole

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3

FINAL_ANSWER_TOOL_NAME = "final_answer"


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


@dataclass(frozen=True)
class VLLMConfig:
    """Configuration for connecting to a local vLLM server."""

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "EMPTY"
    model: str = "gpt-oss:20b"


@dataclass(frozen=True)
class UserMessage:
    content: str
    role: MessageRole = MessageRole.USER

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True)
class AssistantMessage:
    content: str
    role: MessageRole = MessageRole.ASSISTANT

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


class VLLMClient:
    """Thin wrapper around the OpenAI client so we can mock responses in tests."""

    def __init__(self, config: VLLMConfig | None = None, client: AsyncOpenAI | None = None) -> None:
        self.config = config or VLLMConfig()
        self._client = client or AsyncOpenAI(
            base_url=self.config.base_url, api_key=self.config.api_key
        )
        schema_message = (
            "Database Schema\n"
            "Choose between the `query_postgres` SQL tool and the `query_weaviate` vector search tool, or call both if needed to fully answer the request.\n"
            "When SQL is appropriate, call `query_postgres` with a well-formed query against the following schema:\n"
            f"{schema_summary()}"
        )
        self.history: List[Any] = [  # TODO fix typing
            {"role": "system", "content": schema_message}
        ]

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        tools: Sequence[ChatCompletionFunctionToolParam] | None = None,
    ) -> str:
        """Generate a chat completion using the vLLM-backed OpenAI API."""

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

        res = await self.generate_from_messages(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )

        return res

    async def generate_from_messages(
        self,
        message: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        tools: Sequence[ChatCompletionFunctionToolParam] | None = None,
    ) -> str:
        """Generate a completion using an explicit message history."""

        logger.info("Received user message", extra={"user_message": message})
        self.history.append(UserMessage(message).as_dict())

        tool_spec = list(tools) if tools is not None else list(OPENAI_CHAT_TOOLS)

        if not any(_tool_name(tool) == FINAL_ANSWER_TOOL_NAME for tool in tool_spec):
            tool_spec.append(FINAL_ANSWER_TOOL.to_openai_tool())

        it = 0
        while it < MAX_ITERATIONS:
            logger.info(
                "Dispatching chat completion request",
                extra={
                    "iteration": it + 1,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tools": [
                        _tool_name(t) for t in tool_spec
                    ],  # TODO cleanup
                },
            )
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=self.history,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tool_spec,
                tool_choice="auto",
            )

            for choice in response.choices:
                msg = choice.message
                if msg.content:
                    logger.info(
                        "Assistant response received",
                        extra={"response": msg.content, "iteration": it + 1},
                    )
                    self.history.append(AssistantMessage(msg.content).as_dict())
                elif msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
                        raw_arguments = tool_call.function.arguments or "{}"
                        try:
                            arguments = json.loads(raw_arguments)
                        except json.JSONDecodeError:
                            arguments = raw_arguments

                        tool_name = tool_call.function.name
                        self.history.append(
                            ToolCall(
                                name=tool_name,
                                arguments=tool_call.function.arguments,
                                id=tool_call.id,
                            ).as_dict()
                        )
                        logger.info(
                            "Tool call received",
                            extra={
                                "tool_name": tool_name,
                                "tool_call_id": tool_call.id,
                                "arguments": arguments,
                                "iteration": it + 1,
                            },
                        )
                        callback = TOOLS.get(tool_name)

                        if callback is None:
                            tool_response = {"error": f"Unknown tool: {tool_name}"}
                        elif not isinstance(arguments, dict):
                            tool_response = {"error": "Tool arguments must be a JSON object."}
                        else:
                            tool_response = callback(**arguments)

                        logger.info(
                            "Tool response ready",
                            extra={
                                "tool_name": tool_name,
                                "tool_call_id": tool_call.id,
                                "response": tool_response,
                                "iteration": it + 1,
                            },
                        )
                        self.history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(tool_response),
                            }
                        )

                        if tool_name == FINAL_ANSWER_TOOL_NAME:
                            logger.info(
                                "Final answer tool triggered completion",
                                extra={
                                    "tool_call_id": tool_call.id,
                                    "response": tool_response,
                                    "iteration": it + 1,
                                },
                            )
                            return json.dumps(tool_response)

            it += 1

        logger.warning(
            "Maximum iterations reached without final answer",
            extra={"iterations": MAX_ITERATIONS},
        )
        return self.history[-1]

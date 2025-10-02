"""Utilities for calling a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, List, Sequence, cast

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionToolParam,
)

from .db import schema_summary
from .logging_utils import log_operation
from .memory import SerializableMessage, ToolCall, ToolMessage
from .tools import FINAL_ANSWER_TOOL, OPENAI_CHAT_TOOLS, TOOLS
from .utils import MessageRole, _tool_name

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3

FINAL_ANSWER_TOOL_NAME = "final_answer"


@dataclass(frozen=True)
class VLLMConfig:
    """Configuration for connecting to a local vLLM server."""

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "EMPTY"
    model: str = "gpt-oss:20b"


@dataclass(frozen=True)
class ChatMessage:
    content: str
    role: MessageRole

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls(content=content, role=MessageRole.USER)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        return cls(content=content, role=MessageRole.ASSISTANT)

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(content=content, role=MessageRole.SYSTEM)
    

class VLLMClient:
    """Thin wrapper around the OpenAI client so we can mock responses in tests."""

    def __init__(self, config: VLLMConfig | None = None, client: AsyncOpenAI | None = None) -> None:
        self.config = config or VLLMConfig()
        self._client = client or AsyncOpenAI(
            base_url=self.config.base_url, api_key=self.config.api_key
        )
        schema_message = (
            "Choose between the `query_postgres` SQL tool and the `query_weaviate` vector search tool, or call both if needed to fully answer the request.\n"
            "When SQL is appropriate, call `query_postgres` with a well-formed query against the following schema:\n"
            f"{schema_summary()}"
            "If you encounter an error analyze it and retry."
        )
        self.history: List[SerializableMessage] = [ChatMessage.system(schema_message)]

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
            self.history.append(ChatMessage.system(system_prompt))

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
        self.history.append(ChatMessage.user(message))

        tool_spec = list(tools) if tools is not None else list(OPENAI_CHAT_TOOLS)

        if not any(_tool_name(tool) == FINAL_ANSWER_TOOL_NAME for tool in tool_spec):
            tool_spec.append(FINAL_ANSWER_TOOL.to_openai_tool())

        it = 0
        while it < MAX_ITERATIONS:
            tool_names = [_tool_name(t) for t in tool_spec]
            with log_operation(
                logger=logger,
                start_message="Dispatching chat completion request",
                success_message="Chat completion response received",
                failure_message="Chat completion request failed",
                base_extra={
                    "iteration": it + 1,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tools": tool_names,
                },
                success_extra_fn=lambda resp: (
                    {"response_id": getattr(resp, "id")}
                    if getattr(resp, "id", None) is not None
                    else {}
                ),
            ) as completion_log:
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[m.as_dict() for m in self.history],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tool_spec,
                    tool_choice="auto",
                )
                response = completion_log.record_response(response)

            for choice in response.choices:
                msg = choice.message
                if msg.content:
                    logger.info(
                        "Assistant response received",
                        extra={"response": msg.content, "iteration": it + 1},
                    )
                    self.history.append(ChatMessage.assistant(msg.content))
                elif msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
                        raw_arguments = tool_call.function.arguments or "{}"
                        try:
                            arguments = json.loads(raw_arguments)
                        except json.JSONDecodeError:
                            arguments = raw_arguments

                        tool_name = tool_call.function.name
                        with log_operation(
                            logger=logger,
                            start_message="Tool call received",
                            success_message="Tool response ready",
                            failure_message="Tool execution failed",
                            base_extra={
                                "tool_name": tool_name,
                                "tool_call_id": tool_call.id,
                                "iteration": it + 1,
                                "arguments": arguments,
                            },
                            success_extra_fn=lambda response: {"response": response},
                        ) as tool_log:
                            self.history.append(ToolCall.from_openai_tool_call(tool_call))
                            callback = TOOLS.get(tool_name)

                            if callback is None:
                                tool_response = {"error": f"Unknown tool: {tool_name}"}
                            elif not isinstance(arguments, dict):
                                tool_response = {"error": "Tool arguments must be a JSON object."}
                            else:
                                tool_response = callback(**arguments)

                            tool_response = tool_log.record_response(tool_response)
                            self.history.append(
                                ToolMessage(
                                    tool_call_id=tool_call.id,
                                    content=tool_response,
                                )
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

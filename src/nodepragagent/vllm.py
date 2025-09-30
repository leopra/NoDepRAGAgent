"""Utilities for calling a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
from typing import List, Sequence
from .tools import OPENAI_CHAT_TOOLS, TOOLS
from .db import schema_summary
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam, ChatCompletionMessageFunctionToolCall
import json
from .memory import ToolCall

@dataclass(frozen=True)
class VLLMConfig:
    """Configuration for connecting to a local vLLM server."""

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "EMPTY"
    model: str = "gpt-oss:20b"


class VLLMClient:
    """Thin wrapper around the OpenAI client so we can mock responses in tests."""

    def __init__(self, config: VLLMConfig | None = None, client: AsyncOpenAI | None = None) -> None:
        self.config = config or VLLMConfig()
        self._client = client or AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        schema_message = (
            "Database Schema\n"
            "Choose between the `query_postgres` SQL tool and the `query_weaviate` vector search tool, or call both if needed to fully answer the request.\n"
            "When SQL is appropriate, call `query_postgres` with a well-formed query against the following schema:\n"
            f"{schema_summary()}"
        )
        self.history: List[Any] = [ #TODO fix typing
            {"role": "system", "content": schema_message}
        ]


    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        tools: Sequence[ChatCompletionFunctionToolParam] | None = None
    ) -> str:
        """Generate a chat completion using the vLLM-backed OpenAI API."""

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})
        self.history.append({"role": "user", "content": prompt})

        res = await self.generate_from_messages(
            self.history,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
    
        self.history.append({"role": "assistant", "content": res})
        return res
    

    async def generate_from_messages(
        self,
        message: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        tools: Sequence[ChatCompletionFunctionToolParam] | None = None
    ) -> str:
        """Generate a completion using an explicit message history."""

        self.history.append({ 'role': 'user', 'content': message}) #TODO use enum and dataclass for user messages


        tool_spec = list(tools) if tools is not None else list(OPENAI_CHAT_TOOLS)

        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=self.history,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tool_spec,
            tool_choice="auto"
        )

        for choice in response.choices:
            msg = choice.message
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
                    arguments = json.loads(tool_call.function.arguments) #TODO raise errors so agent can fix errors
                    res = TOOLS[tool_call.function.name](**arguments)
                    self.history.append(ToolCall(name=tool_call.function.name, arguments=tool_call.function.arguments, id=tool_call.id))
                    self.history.append({
                        "role":"tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(res)
                    })

            if message and msg.content:
                return msg.content
        return ""
    

    async def call_tool_and_follow_up(self):
        followup = self._client.chat.completions.create(
        model=self.config.model,
        messages=self.history)
        return followup.message

"""Utilities for calling a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
from .tools import (
    sum_two_numbers_tool,
    sum_two_numbers,
    query_weaviate_tool,
    query_weaviate,
    query_postgres_tool,
    query_postgres,
)
from .db import schema_summary
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageFunctionToolCall
import json

TOOLS = {
    sum_two_numbers.__name__: sum_two_numbers,
    query_weaviate.__name__: query_weaviate,
    query_postgres.__name__: query_postgres,
}


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
            "You can call the `query_postgres` tool to run SQL. The database schema is as follows:\n"
            f"{schema_summary()}"
        )
        self.history: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": schema_message}
        ]


    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Generate a chat completion using the vLLM-backed OpenAI API."""

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})
        self.history.append({"role": "user", "content": prompt})

        res = await self.generate_from_messages(
            self.history,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
        self.history.append({"role": "assistant", "content": res})
        return res
    

    async def generate_from_messages(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Generate a completion using an explicit message history."""

        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=[sum_two_numbers_tool(), query_weaviate_tool(), query_postgres_tool()],
            tool_choice="auto"
        )

        for choice in response.choices:
            message = choice.message
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
                    res = TOOLS[tool_call.function.name](**json.loads(tool_call.function.arguments))
                    self.history.append(tool_call)
                    self.history.append({
                        "role":"tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(res)
                    })

            if message and message.content:
                return message.content
        return ""
    

    async def call_tool_and_follow_up(self):
        followup = self._client.chat.completions.create(
        model=self.config.model,
        messages=self.history)
        return followup.message

"""Utilities for calling a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations
import asyncio
import functools
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageCustomToolCall
)
from .utils import serialize_schema

from .memory import (
    ToolCall,
    ToolMessage,
    assistant_message,
    system_message,
    user_message,
    make_json_serializable,
)
from pydantic import ValidationError

from .tools import FINAL_ANSWER_TOOL, OPENAI_CHAT_TOOLS, TOOLS, FINAL_ANSWER_TOOL_NAME
from .config import VLLMConfig, ServiceConfig, DeepSeekConfig
from .db import Base
from .utils import get_tool_name

EventPayload = Dict[str, Any]
EventReporter = Callable[[str, EventPayload], None]

MAX_ITERATIONS = 10

SCHEMA_JSON = serialize_schema(Base())

if __name__ == "__main__":
    print(json.dumps(SCHEMA_JSON, indent=2))

class SearchAgent:
    """Thin wrapper around the OpenAI client so we can mock responses in tests."""

    def __init__(
        self,
        config: ServiceConfig | None = None,
        reporter: EventReporter | None = None,
    ) -> None:
        self.config = config or DeepSeekConfig()
        self._client = AsyncOpenAI(
            base_url=self.config.base_url, api_key=self.config.api_key
        )
        self._reporter = reporter
        schema_message = (
            "Choose between the `query_postgres` SQL tool and the `query_weaviate` vector search tool, or call both if needed to fully answer the request.\n"
            "When SQL is appropriate, call `query_postgres` with a well-formed query against the following schema:\n"
            f"{json.dumps(SCHEMA_JSON, indent=2)}\n"
            "If you encounter an error analyze it and retry."
            "Don't make up any information, only use information you retrieved from SQL or the Vector Database to answer the question.\n"
            "Once you have the final answer call the final_answer tool, do not answer in any other way.\n"
            "Make sure that the tool inputs are always json parsable, do not forget double quotes or parentesys.\n"
            "Before asking questions to the user search for answers to the question and then reason if you need more information to answer.\n"
        )
        self.history: List[ChatCompletionMessageParam] = [system_message(schema_message)]
        self.tool_call_records: List[ToolCall] = []
        self.is_final_answer = False
        self.final_answer_payload: Any | None = None
        self.final_answer_tool: ChatCompletionFunctionToolParam = FINAL_ANSWER_TOOL.to_openai_tool()
        self.tool_spec: List[ChatCompletionFunctionToolParam] = list(OPENAI_CHAT_TOOLS)
        if not any(get_tool_name(tool) == FINAL_ANSWER_TOOL_NAME for tool in self.tool_spec):
            self.tool_spec.append(self.final_answer_tool)
        print(self.config)
        
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
            self.history = [system_message(system_prompt)]

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
        max_tokens: int = 2048,
        tools: Sequence[ChatCompletionFunctionToolParam] | None = None,
    ) -> str:
        """Generate a completion using an explicit message history."""
        self.is_final_answer = False
        self.final_answer_payload = None
        self._log_event("user_message", message=message)
        self.history.append(user_message(message))

        it = 0
        while it < MAX_ITERATIONS:
            self._log_event(
                "model_request",
                temperature=temperature,
                max_tokens=max_tokens,
            )

            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=self.history,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=self.tool_spec,
                tool_choice="auto",
            )

            self._log_event(
                "model_response_received",
                response_reasoning=response.model_extra.get("reasoning", None) if response.model_extra is not None else None,
            )

            for choice in response.choices:
                msg = choice.message
                if msg.content:
                    self._log_event("model_response", content=msg.content)
                    self.history.append(assistant_message(msg.content))
                elif msg.tool_calls:
                    await self.handle_tools(msg.tool_calls)
            if self.is_final_answer:
                return json.dumps(self.history[-1])

            it += 1

        self._log_event("max_iterations_reached", iterations=MAX_ITERATIONS)

        failure_message = {
            "type": "error",
            "reason": "max_iterations_reached",
            "message": "LLM cannot find the answer to the user question.",
        }
        self.history.append(assistant_message(failure_message["message"]))
        return json.dumps(failure_message)

    async def handle_tools(self, tool_calls: List[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageCustomToolCall]) -> str | None:
        for tool_call in tool_calls:
            assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
            raw_arguments = tool_call.function.arguments or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = raw_arguments

            tool_name = tool_call.function.name
            tool_call_record = ToolCall.from_openai_tool_call(tool_call)
            self.tool_call_records.append(tool_call_record)
            self.history.append(tool_call_record.as_message_param())

            self._log_event(
                "tool_call",
                tool_name=tool_name,
                tool_call_id=tool_call.id,
                arguments=arguments,
            )

            tool = TOOLS.get(tool_name)

            if tool is None:
                tool_response = {"error": f"Unknown tool: {tool_name}"}
            elif not isinstance(arguments, dict):
                tool_response = {"error": "Tool arguments must be a JSON object."}
            else:
                try:
                    validated_args = tool.args_model(**arguments)
                except ValidationError as exc:
                    tool_response = {
                        "error": "Invalid tool arguments.",
                        "details": json.dumps(exc.errors()),
                    }
                else:
                    tool_response = await run(
                        tool.callback, **validated_args.model_dump(exclude_none=True)
                    )

            tool_message = ToolMessage(
                tool_call_id=tool_call.id,
                content=tool_response,
            )
            self.history.append(tool_message.as_message_param())

            self._log_event(
                "tool_result",
                tool_name=tool_name,
                tool_call_id=tool_call.id,
                response=tool_response,
            )

            if tool_name == FINAL_ANSWER_TOOL_NAME:
                self._log_event(
                    "final_answer",
                    tool_call_id=tool_call.id,
                    response=tool_response,
                )
        self.is_final_answer = True
        return None

    def save_history(self, file_path: str | Path) -> None:
        """Persist the collected interaction history to a JSON file."""

        path = Path(file_path)
        payload = {
            "history": make_json_serializable(self.history),
            "tool_calls": make_json_serializable(self.tool_call_records),
            "is_final_answer": self.is_final_answer,
            "final_answer_payload": make_json_serializable(self.final_answer_payload),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    def _log_event(self, event: str, **payload: Any) -> None:
        self._report(event, payload)

    def _report(self, event: str, payload: EventPayload) -> None:
        if self._reporter is not None:
            self._reporter(event, payload)

# --- wrapper: run sync in executor, await async directly ---
async def run(func, *args, **kwargs):
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )

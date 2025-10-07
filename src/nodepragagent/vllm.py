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
from .utils import ReporterEvent, get_tool_name, serialize_schema

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
from .errors import LLMError

EventPayload = Dict[str, Any]
EventReporter = Callable[[ReporterEvent, EventPayload], None]

MAX_ITERATIONS = 10
SCHEMA_JSON = serialize_schema(Base())

class SearchAgent:
    """Thin wrapper around the OpenAI client so we can mock responses in tests."""

    def __init__(
        self,
        config: ServiceConfig | None = None,
        reporter: EventReporter | None = None,
        tools: Sequence[ChatCompletionFunctionToolParam] | None = None,
    ) -> None:
        self.config = config or DeepSeekConfig()
        self._client = AsyncOpenAI(
            base_url=self.config.base_url, api_key=self.config.api_key
        )
        self._reporter = reporter
        schema_message = (
            "Choose between the `query_postgres` SQL tool and the `query_weaviate` vector search tool, or call both if needed to fully answer the request.\n"
            "When SQL is appropriate, call `query_postgres` with a well-formed query against the following schema:\n"
            f"{json.dumps(SCHEMA_JSON, indent=0)}\n"
            "If you encounter an error analyze it and retry."
            "Don't make up any information, only use information you retrieved from SQL or the Vector Database to answer the question.\n"
            "Once you have the  required information, call the final_answer tool with the response.\n"
            "Make sure that the tool inputs are always json parsable, do not forget double quotes or parentesys.\n"
            "Before asking questions to the user search for answers to the question and then reason if you need more information to answer.\n"
            "If you are unable to find the answer, call the final_answer tool with a truthful explanation of why you could not find it.\n"
        )
        self.history: List[ChatCompletionMessageParam] = [system_message(schema_message)]
        self.tool_call_records: List[ToolCall] = []
        self.is_final_answer = False
        self.final_answer_payload: str | None = None
        self.final_answer_tool: ChatCompletionFunctionToolParam = FINAL_ANSWER_TOOL.to_openai_tool()
        self.tool_spec: List[ChatCompletionFunctionToolParam] = list(tools) if tools else []
        if not any(get_tool_name(tool) == FINAL_ANSWER_TOOL_NAME for tool in self.tool_spec):
            self.tool_spec.append(self.final_answer_tool)
        

    async def generate_from_messages(
        self,
        message: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a completion using an explicit message history."""
        self.is_final_answer = False
        self.final_answer_payload = None
        self._log_event(ReporterEvent.USER_MESSAGE, message=message)
        self.history.append(user_message(message))

        it = 0
        while it < MAX_ITERATIONS:
            self._log_event(
                ReporterEvent.MODEL_REQUEST,
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

            for choice in response.choices:
                msg = choice.message
                if msg.content:
                    self._log_event(
                        ReporterEvent.REASONING,
                        response_reasoning=msg.model_extra.get("reasoning", None) if msg.model_extra is not None else None,
                    )
                    self._log_event(ReporterEvent.MODEL_RESPONSE, content=msg.content)
                    self.history.append(assistant_message(msg.content))
                elif msg.tool_calls:
                    await self.handle_tools(msg.tool_calls)
            if self.is_final_answer:
                content = self.history[-1].get("content")
                if isinstance(content, str) or content is None:
                    self.final_answer_payload = content
                else:
                    self.final_answer_payload = json.dumps(
                        make_json_serializable(content)
                    )
                assert self.final_answer_payload is not None
                return self.final_answer_payload

            it += 1

        self._log_event(ReporterEvent.MAX_ITERATIONS_REACHED, iterations=MAX_ITERATIONS)

        failure_error = self._build_failure_error()
        failure_payload = failure_error.as_dict()
        self.history.append(assistant_message(failure_payload["message"]))
        return failure_error.as_json()

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
                ReporterEvent.TOOL_CALL,
                tool_name=tool_name,
                tool_call_id=tool_call.id,
                arguments=arguments,
            )

            tool = TOOLS.get(tool_name)

            if tool is None:
                tool_response = LLMError(
                    reason="unknown_tool",
                    message=f"Unknown tool: {tool_name}",
                    details={"tool_name": tool_name},
                ).as_dict()
            elif not isinstance(arguments, dict):
                tool_response = LLMError(
                    reason="invalid_arguments",
                    message="Tool arguments must be a JSON object.",
                ).as_dict()
            else:
                try:
                    validated_args = tool.args_model(**arguments)
                except ValidationError as exc:
                    tool_response = LLMError(
                        reason="invalid_tool_arguments",
                        message="Invalid tool arguments.",
                        details=exc.errors(),
                    ).as_dict()
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
                ReporterEvent.TOOL_RESULT,
                tool_name=tool_name,
                tool_call_id=tool_call.id,
                response=tool_response,
            )

            if tool_name == FINAL_ANSWER_TOOL_NAME:
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

    def _log_event(self, event: ReporterEvent, **payload: Any) -> None:
        self._report(event, payload)

    def _report(self, event: ReporterEvent, payload: EventPayload) -> None:
        if self._reporter is not None:
            self._reporter(event, payload)

    def _build_failure_error(self) -> LLMError:
        return LLMError(
            reason="max_iterations_reached",
            message="LLM cannot find the answer to the user question.",
        )

# --- wrapper: run sync in executor, await async directly ---
async def run(func, *args, **kwargs):
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )

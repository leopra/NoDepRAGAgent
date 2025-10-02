from typing import Any
from dataclasses import dataclass
import json
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)


def make_json_serializable(obj: Any) -> Any:
    """Recursive function to make objects JSON serializable"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        # Try to parse string as JSON if it looks like a JSON object/array
        if isinstance(obj, str):
            try:
                if (obj.startswith("{") and obj.endswith("}")) or (
                    obj.startswith("[") and obj.endswith("]")
                ):
                    parsed = json.loads(obj)
                    return make_json_serializable(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        # For custom objects, convert their __dict__ to a serializable format
        return {
            "_type": obj.__class__.__name__,
            **{k: make_json_serializable(v) for k, v in obj.__dict__.items()},
        }
    else:
        # For any other type, convert to string
        return str(obj)


def system_message(content: str) -> ChatCompletionSystemMessageParam:
    return ChatCompletionSystemMessageParam(role="system", content=content)


def user_message(content: str) -> ChatCompletionUserMessageParam:
    return ChatCompletionUserMessageParam(role="user", content=content)


def assistant_message(content: str) -> ChatCompletionAssistantMessageParam:
    return ChatCompletionAssistantMessageParam(role="assistant", content=content)


@dataclass(kw_only=True)
class ToolCall:
    name: str
    arguments: Any
    id: str

    def as_message_param(self) -> ChatCompletionAssistantMessageParam:
        serialized_arguments = (
            self.arguments
            if isinstance(self.arguments, str)
            else json.dumps(make_json_serializable(self.arguments))
        )

        return ChatCompletionAssistantMessageParam(
            role="assistant",
            tool_calls=[
                {
                    "id": self.id,
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "arguments": serialized_arguments,
                    },
                }
            ],
        )

    @classmethod
    def from_openai_tool_call(cls, tool_call: ChatCompletionMessageFunctionToolCall) -> "ToolCall":
        """Convert OpenAI's tool call structure into our serialized form."""

        return cls(
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
            id=tool_call.id,
        )


@dataclass(kw_only=True)
class ToolMessage:
    tool_call_id: str
    content: Any

    def as_message_param(self) -> ChatCompletionToolMessageParam:
        serialized_content = (
            self.content
            if isinstance(self.content, str)
            else json.dumps(make_json_serializable(self.content))
        )

        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=self.tool_call_id,
            content=serialized_content,
        )

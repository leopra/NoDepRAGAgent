from typing import Any, Protocol, runtime_checkable
from dataclasses import dataclass
import json
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from .utils import MessageRole


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

@dataclass
class ChatMessage:
    content: str = ""
    role: MessageRole

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}

@dataclass
class AssistantMessage(ChatMessage):
    role = MessageRole.ASSISTANT

@dataclass
class SystemMessage(ChatMessage):
    role = MessageRole.SYSTEM

@dataclass
class UserMessage(ChatMessage):
    role = MessageRole.USER
    
@dataclass
class ToolCall(ChatMessage):
    name: str
    arguments: Any
    id: str
    role: MessageRole = MessageRole.ASSISTANT

    def as_dict(self):
        serialized_arguments = (
            self.arguments
            if isinstance(self.arguments, str)
            else json.dumps(make_json_serializable(self.arguments))
        )

        return {
            "role": self.role.value,
            "tool_calls": [
                {
                    "id": self.id,
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "arguments": serialized_arguments,
                    },
                }
            ],
        }

    @classmethod
    def from_openai_tool_call(
        cls, tool_call: ChatCompletionMessageFunctionToolCall
    ) -> "ToolCall":
        """Convert OpenAI's tool call structure into our serialized form."""

        return cls(
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
            id=tool_call.id,
        )


@dataclass
class ToolMessage(ChatMessage):
    tool_call_id: str
    content: Any
    role: MessageRole = MessageRole.TOOL

    def as_dict(self) -> dict[str, Any]:
        serialized_content = (
            self.content
            if isinstance(self.content, str)
            else json.dumps(make_json_serializable(self.content))
        )

        return {
            "role": self.role.value,
            "tool_call_id": self.tool_call_id,
            "content": serialized_content,
        }

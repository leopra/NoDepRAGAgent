from enum import Enum
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

"""Top-level package for the NoDepRAGAgent project."""

from .vllm import VLLMClient, VLLMConfig
from .tools import SUM_TWO_NUMBERS_TOOL, Tool, sum_two_numbers, sum_two_numbers_definition

__all__ = [
    "__version__",
    "VLLMClient",
    "VLLMConfig",
    "sum_two_numbers",
    "sum_two_numbers_definition",
    "SUM_TWO_NUMBERS_TOOL",
    "Tool",
]

__version__ = "0.1.0"

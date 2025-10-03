from dataclasses import dataclass


@dataclass(frozen=True)
class VLLMConfig:
    """Configuration for connecting to a local vLLM server."""

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "EMPTY"
    model: str = "gpt-oss:20b"

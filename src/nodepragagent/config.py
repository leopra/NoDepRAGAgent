import os
from typing import ClassVar, Any

from pydantic import BaseModel, Field, field_validator



def _env_field(env_var: str, default: str) -> Any:
    return Field(default_factory=lambda: os.getenv(env_var, default))

class ServiceConfig(BaseModel):
    """Base configuration for OpenAI-compatible services."""

    require_api_key: ClassVar[bool] = False
    api_key_env_var: ClassVar[str] = "API_KEY"

    base_url: str
    api_key: str
    model: str

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, value: str) -> str:
        if cls.require_api_key and not value:
            raise ValueError(f"{cls.api_key_env_var} environment variable must be set.")
        return value

class VLLMConfig(ServiceConfig):
    """Configuration for connecting to a local vLLM server."""

    api_key_env_var: ClassVar[str] = "VLLM_API_KEY"

    base_url: str = _env_field("VLLM_BASE_URL", "http://localhost:11434/v1")
    api_key: str = _env_field("VLLM_API_KEY", "EMPTY")
    model: str = _env_field("VLLM_MODEL", "gpt-oss:20b")


class DeepSeekConfig(ServiceConfig):
    """Configuration for the DeepSeek API endpoint loaded from environment variables."""

    require_api_key: ClassVar[bool] = True
    api_key_env_var: ClassVar[str] = "DEEPSEEK_API_KEY"

    base_url: str = _env_field("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    api_key: str = _env_field("DEEPSEEK_API_KEY", "")
    model: str = _env_field("DEEPSEEK_MODEL", "deepseek-reasoner")

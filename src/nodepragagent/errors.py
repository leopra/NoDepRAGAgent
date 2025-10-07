"""Shared error payloads for LLM-serializable responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class LLMError(BaseModel):
    """Standard error payload that can be emitted back to the LLM."""

    type: Literal["error"] = "error"
    reason: str
    message: str
    details: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return self.model_dump(exclude_none=True)

    def as_json(self) -> str:
        """Return the payload encoded as JSON."""

        return self.model_dump_json(exclude_none=True)


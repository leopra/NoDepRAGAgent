"""Embedding utilities backed by the local OpenAI-compatible endpoint."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import List, Sequence

from openai import AsyncOpenAI

from .config import VLLMConfig

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


def _embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL") or os.getenv("OLLAMA_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


async def _embed_async(texts: List[str], *, config: VLLMConfig, model: str) -> List[Sequence[float]]:
    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
    try:
        response = await client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]
    finally:
        await client.close()
        

async def embed_contents(contents: Iterable[str], *, config: VLLMConfig | None = None, model: str | None = None) -> List[Sequence[float]]:
    texts = [text for text in contents]
    if not texts:
        return []

    resolved_config = config or VLLMConfig()
    resolved_model = model or _embedding_model()

    return await _embed_async(texts, config=resolved_config, model=resolved_model)

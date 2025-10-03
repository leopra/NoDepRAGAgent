from __future__ import annotations

from pathlib import Path

from nodepragagent.vllm import VLLMClient

ASSET_PATH = Path(__file__).parent / "assets" / "system_prompt.txt"


def _runtime_system_prompt() -> str:
    client = VLLMClient(client=object())
    first_message = client.history[0]

    content = None
    try:
        content = first_message["content"]  # type: ignore[index]
    except (TypeError, KeyError):
        content = getattr(first_message, "content", None)

    if content is None:
        raise AssertionError("System prompt content unavailable")

    return str(content)


def test_system_prompt_asset_matches_runtime(update_assets: bool) -> None:
    runtime_prompt = _runtime_system_prompt()
    asset_content = ASSET_PATH.read_text(encoding="utf-8")

    if update_assets and asset_content != runtime_prompt:
        ASSET_PATH.write_text(runtime_prompt, encoding="utf-8")
        asset_content = runtime_prompt

    assert asset_content == runtime_prompt

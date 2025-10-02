#!/usr/bin/env python3
"""Prompt the local LLM to exercise every available tool once."""

from __future__ import annotations

import asyncio

from nodepragagent.tools import OPENAI_CHAT_TOOLS
from nodepragagent.vllm import VLLMClient

_TOOL_NAMES = {"sum_two_numbers", "query_postgres", "query_weaviate"}


async def _exercise_tools() -> None:
    client = VLLMClient()
    request = (
        "For a diagnostic run, call each available tool exactly once before crafting your final "
        "answer. First add 13 and 29 with the sum_two_numbers tool. Then run the query_postgres "
        "tool with a simple read-only statement like `SELECT 1 AS demo_column;`. Finally, invoke "
        "query_weaviate to search for learning resources about retrieval augmented generation. "
        "After you have all tool outputs, summarize the results succinctly."
        "Call all tools in one request."
    )

    try:
        response = await client.generate_from_messages(
            request,
            temperature=0.0,
            max_tokens=1024,
            tools=list(OPENAI_CHAT_TOOLS),
        )
    except Exception as exc:  # pragma: no cover - network issues are environment specific
        print(f"Model call failed: {exc}")
        return

    tool_calls = client.tool_call_records
    observed = {call.name for call in tool_calls}

    print("Final assistant response:\n")
    print(response or "(no response body)")
    print("\nTool calls observed:")
    if not tool_calls:
        print("  (none)")
    else:
        for call in tool_calls:
            print(f"  - {call.name} args={call.arguments}")

    missing = _TOOL_NAMES - observed
    if missing:
        print(f"\nTools not exercised: {', '.join(sorted(missing))}")
    else:
        print("\nAll tools exercised successfully.")


def main() -> None:
    asyncio.run(_exercise_tools())


if __name__ == "__main__":  # pragma: no cover
    main()

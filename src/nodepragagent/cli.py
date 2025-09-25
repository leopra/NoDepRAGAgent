"""Command line entry point for interacting with the RAG agent."""

import asyncio
import sys
from typing import Iterable, List

from openai.types.chat import ChatCompletionMessageParam

from .vllm import VLLMClient, VLLMConfig

def _prompt_lines() -> Iterable[str]:
    """Yield successive prompts entered by the user."""

    while True:
        try:
            prompt = input("You> ").strip()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            return

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return

        yield prompt


async def main() -> None:
    """Run the RAG agent CLI loop."""

    default_config = VLLMConfig()
    client = VLLMClient(config=default_config)

    print("NoDepRAGAgent CLI. Type 'exit' or 'quit' to leave. Press Ctrl+C to abort.")

    for prompt in _prompt_lines():
        try:
            response = await client.generate_from_messages(
                prompt,
                temperature=1,
                max_tokens=164,
            )
        except Exception as exc:  # pragma: no cover - network errors not deterministic in tests
            print(f"[ERROR] Failed to call model: {exc}", file=sys.stderr)
            history.pop()
            continue

        print(f"Agent> {response}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())

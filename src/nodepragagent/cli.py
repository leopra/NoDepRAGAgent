"""Command line entry point for interacting with the RAG agent."""

import asyncio
import sys
from typing import Iterable, Sequence

from .tools import OPENAI_CHAT_TOOLS
from .vllm import VLLMClient, VLLMConfig
from .utils import cli_event_printer


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


async def main(argv: Sequence[str] | None = None) -> None:
    """Run the RAG agent CLI loop."""

    prompt_arg: str | None = None
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        prompt_arg = " ".join(str(part) for part in argv).strip() or None

    default_config = VLLMConfig()
    client = VLLMClient(config=default_config, reporter=cli_event_printer)

    prompt_source: Iterable[str]
    if prompt_arg is not None:
        prompt_source = (prompt_arg,)
    else:
        print("NoDepRAGAgent CLI. Type 'exit' or 'quit' to leave. Press Ctrl+C to abort.")
        prompt_source = _prompt_lines()

    for prompt in prompt_source:
        try:
            response = await client.generate_from_messages(
                prompt,
                temperature=1,
                max_tokens=2048,
                tools=list(OPENAI_CHAT_TOOLS),
            )
        except Exception as exc:  # pragma: no cover - network errors not deterministic in tests
            print(f"[ERROR] Failed to call model: {exc}", file=sys.stderr)
            continue

        print(f"Agent> {response}")

        if prompt_arg is not None:
            return


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())

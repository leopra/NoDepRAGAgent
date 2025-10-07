"""Command line entry point for interacting with the RAG agent."""

import asyncio
import argparse
import sys
from typing import Iterable, Sequence

from .tools import OPENAI_CHAT_TOOLS
from .vllm import SearchAgent, VLLMConfig, DeepSeekConfig
from .utils import cli_event_printer
from dotenv import load_dotenv
load_dotenv()

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

    parser = argparse.ArgumentParser(description="Interact with the NoDep RAG agent from the CLI")
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional single turn prompt to run without starting the interactive loop",
    )
    parser.add_argument(
        "--save-history",
        dest="save_history",
        metavar="PATH",
        help="File path where the full conversation history will be stored as JSON",
    )

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    prompt_arg = " ".join(args.prompt).strip() if args.prompt else None

    default_config = VLLMConfig()
    client = SearchAgent(config=default_config, reporter=cli_event_printer, tools=OPENAI_CHAT_TOOLS)

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
            )
        except Exception as exc:  # pragma: no cover - network errors not deterministic in tests
            print(f"[ERROR] Failed to call model: {exc}", file=sys.stderr)
            continue

        print(f"Final Answer> {response}")

        if args.save_history:
            try:
                client.save_history(args.save_history)
            except Exception as exc:
                print(f"[ERROR] Failed to save history: {exc}", file=sys.stderr)

        if prompt_arg is not None:
            return

    if args.save_history:
        try:
            client.save_history(args.save_history)
        except Exception as exc:
            print(f"[ERROR] Failed to save history: {exc}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())

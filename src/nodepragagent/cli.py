"""Command line entry point for interacting with the RAG agent."""

import asyncio
import json
import logging
import sys
from typing import Iterable, Sequence

from .tools import OPENAI_CHAT_TOOLS
from .vllm import VLLMClient, VLLMConfig


class _ExtraFormatter(logging.Formatter):
    """Formatter that renders arbitrary `extra` fields as JSON."""

    _STANDARD_ATTRS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "extras",
    }

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            key: value for key, value in vars(record).items() if key not in self._STANDARD_ATTRS
        }
        record.extras = json.dumps(extras, default=str) if extras else "{}"
        return super().format(record)


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

    handler = logging.StreamHandler()
    handler.setFormatter(
        _ExtraFormatter("%(asctime)s %(levelname)s %(name)s %(message)s extras=%(extras)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    prompt_arg: str | None = None
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        prompt_arg = " ".join(str(part) for part in argv).strip() or None

    default_config = VLLMConfig()
    client = VLLMClient(config=default_config)

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
                max_tokens=164,
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

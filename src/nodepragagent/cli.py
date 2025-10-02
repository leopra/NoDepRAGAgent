"""Command line entry point for interacting with the RAG agent."""

import asyncio
import json
import sys
from typing import Iterable, Sequence

from .tools import OPENAI_CHAT_TOOLS
from .vllm import VLLMClient, VLLMConfig


def _format_payload(payload: object) -> str:
    """Render tool arguments or responses for CLI output."""

    if isinstance(payload, str):
        return payload

    try:
        return json.dumps(payload, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return str(payload)


def _cli_event_printer(event: str, payload: dict[str, object]) -> None:
    """Print VLLM client events in a human-friendly format."""

    iteration = payload.get("iteration")
    prefix = f"[iter {iteration}] " if iteration is not None else ""

    if event == "user_message":
        return
    if event == "model_request":
        tools = payload.get("tools") or []
        if isinstance(tools, (list, tuple)):
            tool_list = ", ".join(str(tool) for tool in tools)
        else:
            tool_list = str(tools)
        print(f"{prefix}-> calling model with tools: {tool_list}")
    elif event == "model_response":
        content = payload.get("content", "")
        print(f"{prefix}Model> {content}")
    elif event == "tool_call":
        tool_name = payload.get("tool_name", "unknown")
        args = _format_payload(payload.get("arguments"))
        tool_id = payload.get("tool_call_id")
        suffix = f" (id: {tool_id})" if tool_id else ""
        print(f"{prefix}Tool> {tool_name}{suffix}\n{args}")
    elif event == "tool_result":
        tool_name = payload.get("tool_name", "unknown")
        result = _format_payload(payload.get("response"))
        print(f"{prefix}Tool< {tool_name}\n{result}")
    elif event == "final_answer":
        result = _format_payload(payload.get("response"))
        print(f"{prefix}Final> {result}")
    elif event == "model_response_received":
        response_id = payload.get("response_id")
        if response_id:
            print(f"{prefix}<-- model response id: {response_id}")
    elif event == "max_iterations_reached":
        print("[warn] Maximum iterations reached without final answer")


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
    client = VLLMClient(config=default_config, reporter=_cli_event_printer)

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

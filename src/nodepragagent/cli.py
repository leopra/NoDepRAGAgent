"""Command line entry point for interacting with the RAG agent."""

import asyncio
import argparse
import sys
from typing import Sequence
import json
from openai import OpenAIError

from .tools import OPENAI_CHAT_TOOLS, POSTGRES_SCHEMA
from .vllm import SearchAgent, VLLMConfig, DeepSeekConfig
from .utils import cli_event_printer
from dotenv import load_dotenv
load_dotenv()

POSTGRES_SCHEMA_JSON = json.dumps(POSTGRES_SCHEMA, indent=0)

SYSTEM_PROMPT = (
    "You are a Hybrid Search Agent that must respond truthfully to the user's questions.\n"
    "Choose between the `query_postgres` SQL tool and the `query_weaviate` vector search tool, or call both if needed to fully answer the request.\n"
    "When SQL is appropriate, call `query_postgres` with a well-formed query against the following schema:\n"
    f"{POSTGRES_SCHEMA_JSON}\n"
    "If you encounter an error analyze it and retry. If you don't find the answer in the SQL results, use the `query_weaviate` tool to search for relevant documents.\n"
    "Don't make up any information, only use information you retrieved from SQL or the Vector Database to answer the question.\n"
    "Make sure that the tool inputs are always json parsable, do not forget double quotes or parentesys.\n"
    "Before asking questions to the user search for answers to the question and then reason if you need more information to answer.\n"
    "Once you have the required information, respond directly to the user with the answer. If you are unable to find the answer, provide a truthful explanation.\n"
    "When you have the final answer, respond in a well written manner, citing the sources you used to construct your answer. Do not answer in json format.\n"
)

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
    client = SearchAgent(
        config=default_config,
        reporter=cli_event_printer,
        tools=OPENAI_CHAT_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    if prompt_arg:
        prompt = prompt_arg
    else:
        try:
            prompt = input("You> ").strip()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            return

        if not prompt or prompt.lower() in {"exit", "quit"}:
            return

    try:
        response = await client.generate_from_messages(
            prompt,
            temperature=1,
            max_tokens=2048,
        )
    except OpenAIError as exc:
        print(f"[ERROR] Failed to call model: {exc}", file=sys.stderr)

    if args.save_history:
        try:
            client.save_history(args.save_history)
        except OSError as exc:
            print(f"[ERROR] Failed to save history: {exc}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())

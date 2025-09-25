# NoDepRAGAgent

Interview Task: Building a Hybrid Query System (No Frameworks)

## Setup

Use [uv](https://github.com/astral-sh/uv) for environment management and dependency installation:

```sh
make install
```

This runs `uv sync --extra dev`, creating `.venv` and installing the project with development dependencies.

## Objective

We'd like you to design and implement a system that can answer user questions by intelligently querying both a structured database (SQL) and an unstructured text corpus (via vector search). The key challenge is to do this without relying on existing LLM frameworks like Langchain, LlamaIndex, etc. You will need to implement the core logic for query understanding, decomposition, data retrieval, and answer synthesis.

## System Requirements

1. Input: The system should accept a natural language question from the user.
2. Outputs: This final step should synthesize the information and generate a coherent, natural language answer to the user's original question.

## Implementation Notes

- No Frameworks: You must not use libraries like Langchain, Haystack, LlamaIndex, or any other comprehensive RAG/LLM agent frameworks. Standard libraries for database interaction (e.g., sqlite3), numerical operations (e.g., numpy for vector math if needed), and basic text processing are acceptable.
- Frontend: A command-line interface (CLI) for interaction is perfectly acceptable.
- Focus: The emphasis is on your understanding of the components of such a system, the logic flow, and how you would approach building it from more fundamental pieces.

## Demonstration & Discussion

During our meeting, we'd like you to:

1. Demonstrate your system with a few example questions.
2. Talk us through your implementation: Explain your design choices, the logic for query decomposition, data retrieval, result combination, and answer synthesis.
3. Discuss the challenges you encountered while building this system from scratch, particularly in the absence of high-level frameworks.

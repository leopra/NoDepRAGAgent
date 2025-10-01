from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from nodepragagent.vllm import VLLMClient, VLLMConfig


def _chat_message(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(content=content)


def _chat_choice(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(message=_chat_message(content))


def _chat_response(choices: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(choices=choices)


def test_generate_returns_mocked_content() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _chat_response(
        [
            _chat_choice("mocked response"),
        ]
    )
    client = VLLMClient(config=VLLMConfig(model="mock-model"), client=mock_client)

    result = client.generate("hello", temperature=0.5, max_tokens=64)

    assert result == "mocked response"
    mock_client.chat.completions.create.assert_called_once()
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "mock-model"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 64
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]


def test_generate_includes_system_prompt_when_provided() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _chat_response(
        [
            _chat_choice("another response"),
        ]
    )
    client = VLLMClient(client=mock_client)

    client.generate("question", system_prompt="be concise")

    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["messages"] == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "question"},
    ]


def test_generate_returns_empty_string_when_no_content() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _chat_response([])
    client = VLLMClient(client=mock_client)

    result = client.generate("hello")

    assert result == ""


def test_generate_from_messages_sends_history() -> None:
    history = [
        {"role": "system", "content": "stay helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
        {"role": "user", "content": "how are you?"},
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _chat_response(
        [
            _chat_choice("doing well"),
        ]
    )
    client = VLLMClient(client=mock_client)

    result = client.generate_from_messages(history, temperature=0.7, max_tokens=256)

    assert result == "doing well"
    mock_client.chat.completions.create.assert_called_once()
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["messages"] == history
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 256

"""Logging utilities shared across NoDepRAGAgent modules."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator


class _LogContext:
    """Helper that stores state for a logging context manager."""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        base_extra: dict[str, Any],
        success_extra_fn: Callable[[Any], dict[str, Any]] | None,
        success_message: str,
    ) -> None:
        self._logger = logger
        self._base_extra = base_extra
        self._success_extra_fn = success_extra_fn
        self._success_message = success_message
        self._response: Any = None

    def record_response(self, response: Any) -> Any:
        self._response = response
        return response

    def log_success(self) -> None:
        extra = dict(self._base_extra)
        if self._success_extra_fn is not None:
            extra_update = self._success_extra_fn(self._response)
            if isinstance(extra_update, dict):
                extra.update(extra_update)
        self._logger.info(self._success_message, extra=extra)


@contextmanager
def log_operation(
    *,
    logger: logging.Logger,
    start_message: str,
    success_message: str,
    failure_message: str,
    base_extra: dict[str, Any],
    success_extra_fn: Callable[[Any], dict[str, Any]] | None = None,
) -> Iterator[_LogContext]:
    """Generic logging context manager to avoid repeated boilerplate."""

    logger.info(start_message, extra=base_extra)
    context = _LogContext(
        logger=logger,
        base_extra=base_extra,
        success_extra_fn=success_extra_fn,
        success_message=success_message,
    )
    try:
        yield context
    except Exception:
        logger.exception(failure_message, extra=base_extra)
        raise
    else:
        context.log_success()


__all__ = ["log_operation"]

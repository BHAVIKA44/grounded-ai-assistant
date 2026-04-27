"""Retry helpers."""

import asyncio
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


async def retry_async(
    func: Callable[[], Awaitable[T]],
    retries: int = 3,
    delay_seconds: float = 0.5,
    backoff: float = 2.0,
) -> T:
    """Retry an async callable with exponential backoff."""
    attempt = 0
    wait = delay_seconds
    while True:
        try:
            return await func()
        except Exception:
            attempt += 1
            if attempt > retries:
                raise
            await asyncio.sleep(wait)
            wait *= backoff

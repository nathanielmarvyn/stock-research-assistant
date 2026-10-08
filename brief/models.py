"""Shared result types.

Every section of the brief is produced as a ``SectionResult``. A failure in one
section is captured as data (``error``) instead of an exception, so the UI can
render an error card for that section and keep loading the rest.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Generic, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


class DataUnavailableError(Exception):
    """An expected, user-facing failure (no data, provider down, rate-limited).

    ``safe_section`` shows its message as-is and skips the traceback, unlike
    unexpected exceptions, which are logged in full as bugs.
    """


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SectionResult(Generic[T]):
    """The outcome of building one section: its data or an error, plus provenance."""

    data: T | None
    source: str
    as_of: datetime = field(default_factory=utc_now)
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the section has data to show."""
        return self.error is None and self.data is not None

    @classmethod
    def success(
        cls, data: T, source: str, as_of: datetime | None = None
    ) -> SectionResult[T]:
        """Wrap successfully fetched data."""
        return cls(data=data, source=source, as_of=as_of or utc_now())

    @classmethod
    def failure(cls, error: str, source: str) -> SectionResult[T]:
        """Record a user-readable error for a section that could not be built."""
        return cls(data=None, source=source, error=error)


def safe_section(
    source: str,
) -> Callable[[Callable[P, SectionResult[T]]], Callable[P, SectionResult[T]]]:
    """Decorator: turn any unexpected exception into a failed SectionResult."""

    def decorator(func: Callable[P, SectionResult[T]]) -> Callable[P, SectionResult[T]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> SectionResult[T]:
            try:
                return func(*args, **kwargs)
            except DataUnavailableError as exc:
                logger.warning("Section %s unavailable: %s", func.__name__, exc)
                return SectionResult.failure(str(exc), source=source)
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                logger.exception("Section %s failed", func.__name__)
                return SectionResult.failure(
                    f"{type(exc).__name__}: {exc}", source=source
                )

        return wrapper

    return decorator

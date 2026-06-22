from __future__ import annotations

import datetime
import enum

from typing_extensions import Any

__all__ = ["value_phrase"]


def value_phrase(value: Any) -> str:
    """
    Render a Python value as a human-readable string — the single value-lexicalisation point.

    * ``None`` → ``"nothing"`` (a genuine value-slot absence; a top-level ``== None`` comparison
      is rendered as an absence predicate, not via this function).
    * A bare ``type`` → its ``__name__`` (``Apple`` → ``"Apple"``).
    * A tuple of types → ``"A or B or C"``.
    * An ``enum`` member → its ``name`` (``OPTION_A`` rather than ``<TestEnum.OPTION_A: …>``).
    * A ``datetime`` with no time → ``"May 23, 2026"``; with a time → ``"May 23, 2026 at 14:30"``.
    * Anything else → ``repr(value)``.

    :param value: Python value from a literal node.
    :return: Human-readable string representation.
    """
    if value is None:
        return "nothing"
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, tuple) and all(
        isinstance(variable, type) for variable in value
    ):
        return " or ".join(variable.__name__ for variable in value)
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, datetime.datetime):
        if value.time() == datetime.time.min:
            return value.strftime("%B %-d, %Y")
        return value.strftime("%B %-d, %Y at %H:%M")
    return repr(value)

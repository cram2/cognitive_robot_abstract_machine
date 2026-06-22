from __future__ import annotations

import contextlib
from dataclasses import dataclass

from typing_extensions import Iterator


@dataclass
class RenderConfiguration:
    """
    Mutable render-mode flags for the current verbalization pass — the small set that switch how
    a clause is realised, independent of *what* is said:

    * ``query_depth`` selects the imperative *"Find …"* form for a top-level query versus a
      nested noun phrase for a sub-query used as a value.
    * ``compact_predicates`` drops the copula in post-nominal / HAVING contexts (*"greater than
      10"* instead of *"is greater than 10"*).
    """

    query_depth: int = 0
    """Number of enclosing query/noun renderings on the stack.  ``0`` ⇒ the next
    Entity is the top-level request (imperative *"Find … such that …"*); ``> 0`` ⇒
    a nested Entity rendered as a noun phrase."""

    compact_predicates: bool = False
    """When ``True``, comparators omit the copula *"is"* (*"greater than"* rather
    than *"is greater than"*)."""

    @contextlib.contextmanager
    def query_depth_scope(self) -> Iterator[None]:
        """Increment ``query_depth`` for the duration of a ``with`` block, restoring it on exit."""
        self.query_depth += 1
        try:
            yield
        finally:
            self.query_depth -= 1

    @contextlib.contextmanager
    def compact_predicates_scope(self) -> Iterator[None]:
        """Set ``compact_predicates`` ``True`` for a ``with`` block, restoring it on exit (even on error)."""
        previous = self.compact_predicates
        self.compact_predicates = True
        try:
            yield
        finally:
            self.compact_predicates = previous

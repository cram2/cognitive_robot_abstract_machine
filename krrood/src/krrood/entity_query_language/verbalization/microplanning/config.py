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
    """

    query_depth: int = 0
    """Number of enclosing query/noun renderings on the stack.  ``0`` ⇒ the next
    Entity is the top-level request (imperative *"Find … such that …"*); ``> 0`` ⇒
    a nested Entity rendered as a noun phrase."""

    possessive_aggregate: bool = False
    """When ``True``, an aggregate value noun drops its determiner so it reads as a possession
    (*"sum of salaries"* rather than *"the sum of salaries"*) — used by a fronted *"whose <aggregate>
    is …"* group filter."""

    @contextlib.contextmanager
    def query_depth_scope(self) -> Iterator[None]:
        """Increment ``query_depth`` for the duration of a ``with`` block, restoring it on exit."""
        self.query_depth += 1
        try:
            yield
        finally:
            self.query_depth -= 1

    @contextlib.contextmanager
    def possessive_aggregate_scope(self) -> Iterator[None]:
        """Set ``possessive_aggregate`` ``True`` for a ``with`` block, restoring it on exit."""
        previous = self.possessive_aggregate
        self.possessive_aggregate = True
        try:
            yield
        finally:
            self.possessive_aggregate = previous

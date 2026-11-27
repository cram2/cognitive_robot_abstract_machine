"""
Mimics of what an evaluation reads ground terms with, such as a planning state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Iterable, Iterator

from krrood.entity_query_language.ground_term import (
    GroundTerm,
    LiveTermReader,
    TermReader,
    TermValue,
)
from krrood.entity_query_language.utils import T

# %% readers


@dataclass(kw_only=True)
class AssignedValueReader(TermReader):
    """
    Reads the value assigned to a ground term, and the live value of every other.
    """

    assigned_values: dict[GroundTerm, Any] = field(default_factory=dict)
    """
    The value read for each assigned ground term.
    """

    def value_of_term(self, *, term: GroundTerm) -> TermValue | None:
        if term in self.assigned_values:
            return TermValue(value=self.assigned_values[term])
        return LiveTermReader().value_of_term(term=term)

    def existing_values(self, *, values: Iterable[T]) -> Iterator[T]:
        return iter(values)

    def added_instances_of(self, *, type_: type[T]) -> Iterator[T]:
        return iter(())

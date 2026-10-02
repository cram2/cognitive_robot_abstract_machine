"""
The unit a query source offers the EQL runner: one named, typed set of objects.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from typing_extensions import Any, Optional, Type


@dataclass(frozen=True)
class QueryDomain:
    """
    One ready-made EQL variable: a name to write queries with, and what it ranges over.

    A source declares its domains rather than building a namespace itself, so it cannot
    shadow the EQL factories a query is written in.
    """

    name: str
    """
    Name the variable is bound to in a query.
    """

    entity_type: Type[Any]
    """
    Type of the objects the variable ranges over; also in scope under its class name.
    """

    objects: Optional[Iterable[Any]] = None
    """
    The objects themselves, or None when the answer does not come from this process.

    A re-iterable collection supplies current values on each query. With no collection,
    :class:`~cramera.knowledge.queryable_knowledge.QueryEvaluation` supplies the answer.
    """

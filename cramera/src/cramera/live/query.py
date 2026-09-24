"""Registered query data and failures when live knowledge is unavailable."""

from collections.abc import Callable

from typing_extensions import TypeAlias

from cramera.knowledge.presets import Preset
from cramera.knowledge.queryable_knowledge import QueryableKnowledge

QueryKnowledge: TypeAlias = QueryableKnowledge | list[QueryableKnowledge]
"""
One registered query scope or a collection of scopes.
"""

QueryKnowledgeSource: TypeAlias = QueryKnowledge | Callable[[], QueryKnowledge]
"""
Registered knowledge, provided directly or read fresh for each query operation.
"""

QueryPresets: TypeAlias = list[Preset] | Callable[[], list[Preset]]
"""
Registered presets, provided directly or collected when requested.
"""


class NoQuerySourceRegistered(Exception):
    """
    Raised when a query has neither an explicit source nor an attached world source.
    """

    def __init__(self) -> None:
        """
        Explain that this session has no registered source of queryable state.
        """
        super().__init__("no query source is registered on this bridge")

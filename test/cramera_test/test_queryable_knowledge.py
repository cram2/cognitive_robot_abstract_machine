"""
Tests for the bodies of knowledge a question can be asked of.

A demo knows two different things: what is true of it right now, and what its finished
runs left behind. They are asked in the same language but answered in different places,
which is what a scope names.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from typing_extensions import Any, List  # noqa: E402

from cramera.knowledge.query_domain import QueryDomain  # noqa: E402
from cramera.knowledge.queryable_knowledge import (  # noqa: E402
    InMemoryEvaluation,
    QueryableKnowledge,
    QueryEvaluation,
    QueryScope,
    UnknownQueryScope,
)

from .dataset.queryable_records import NamedRecord  # noqa: E402


@dataclass(frozen=True)
class RecordingEvaluation(QueryEvaluation):
    """
    An evaluation that reports being asked instead of answering.
    """

    answer: Any = "answered elsewhere"

    def evaluate(self, expression: Any) -> Any:
        return self.answer


class TestQueryScope:
    def test_every_scope_has_a_heading_to_show_it_under(self):
        assert QueryScope.CURRENT_STATE.label == "Current State Queries"
        assert QueryScope.EPISODIC_MEMORY.label == "Episodic Memory Queries"

    def test_a_scope_is_read_back_from_the_name_it_travels_as(self):
        assert QueryScope.of_name("episodic_memory") is QueryScope.EPISODIC_MEMORY

    def test_a_name_no_scope_carries_is_refused(self):
        with pytest.raises(UnknownQueryScope):
            QueryScope.of_name("yesterday")


class TestQueryableKnowledge:
    def test_knowledge_is_answered_from_memory_unless_it_says_otherwise(self):
        knowledge = QueryableKnowledge(
            scope=QueryScope.CURRENT_STATE,
            domains=[QueryDomain("record", NamedRecord, [])],
        )
        assert isinstance(knowledge.evaluation, InMemoryEvaluation)

    def test_declared_domains_are_the_ones_offered(self):
        domains: List[QueryDomain] = [QueryDomain("record", NamedRecord, [])]
        knowledge = QueryableKnowledge(
            scope=QueryScope.EPISODIC_MEMORY,
            domains=domains,
            evaluation=RecordingEvaluation(),
        )
        assert knowledge.domains == domains

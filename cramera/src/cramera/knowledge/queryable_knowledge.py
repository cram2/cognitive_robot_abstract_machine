"""
The bodies of knowledge a demo offers to be questioned about.

A running demo knows several different things: what is true of it right now, what its
detectors saw it do, and what its finished runs left behind. All are asked in the same
language, but each is answered from somewhere else, so each declares how it answers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from krrood.entity_query_language.evaluable import Evaluable
from krrood.exceptions import DataclassException
from typing_extensions import Any, Dict, List

from cramera.knowledge.query_domain import QueryDomain


class QueryScope(StrEnum):
    """
    Which of a demo's bodies of knowledge a question is about.
    """

    CURRENT_STATE = "current_state"
    """
    What is true of the run in progress, read from the objects it keeps as it goes.
    """

    DETECTED_EVENTS = "detected_events"
    """
    The moments of the run its detectors saw, each answered with the window of the
    recording worth replaying around it.
    """

    EPISODIC_MEMORY = "episodic_memory"
    """
    What the runs that already finished recorded of themselves.
    """

    @property
    def label(self) -> str:
        """
        The heading questions of this scope are offered under.
        """
        return "%s Queries" % self.value.replace("_", " ").title()

    @classmethod
    def of_name(cls, name: str) -> QueryScope:
        """
        The scope a name stands for.

        :param name: The scope's name, as it travels to and from the viewer.
        :raises UnknownQueryScope: When no scope carries that name.
        """
        if name not in cls._value2member_map_:
            raise UnknownQueryScope(name=name)
        return cls(name)


@dataclass
class UnknownQueryScope(DataclassException):
    """
    Raised when a question is asked of a body of knowledge that is not on offer.
    """

    name: str
    """
    The scope name that was asked for.
    """

    def error_message(self) -> str:
        """
        Identify the unavailable query scope.
        """
        return "No queryable knowledge named '%s'" % self.name

    def suggest_correction(self) -> str:
        """
        List the query scopes the viewer supports.
        """
        return "Ask one of: %s." % ", ".join(scope.value for scope in QueryScope)


class QueryEvaluation(ABC):
    """
    Where the answer to a query is worked out.
    """

    @abstractmethod
    def evaluate(self, expression: Evaluable) -> Any:
        """
        Answer one query expression.

        :param expression: The query to answer.
        """


@dataclass(frozen=True)
class InMemoryEvaluation(QueryEvaluation):
    """
    Answers from the objects the query's domains already hold.
    """

    def evaluate(self, expression: Evaluable) -> Any:
        """
        Answer by evaluating the expression where it stands.

        :param expression: The query to answer.
        """
        return expression.evaluate()


@dataclass
class QueryableKnowledge:
    """
    One body of knowledge questions can be asked of, and how they are answered.
    """

    scope: QueryScope
    """
    Which of a demo's bodies of knowledge this is.
    """

    domains: List[QueryDomain]
    """
    The ready-made variables a question about it may range over.
    """

    evaluation: QueryEvaluation = field(default_factory=InMemoryEvaluation)
    """
    Where a question about it is worked out.
    """

    extra_names: Dict[str, Any] = field(default_factory=dict)
    """
    Further names a question about it may use, such as the vocabulary its values are
    recorded in.
    """

from __future__ import annotations

import importlib.util
from enum import Enum, StrEnum, auto


class InferMode(Enum):
    """
    The infer mode of a predicate, whether to infer new relations or retrieve current relations.
    """

    Auto = auto()
    """
    Inference is done automatically depending on the world state.
    """
    Always = auto()
    """
    Inference is always performed.
    """
    Never = auto()
    """
    Inference is never performed.
    """


class EQLMode(Enum):
    """
    The modes of an entity query.
    """

    Rule = auto()
    """
    Means this is a Rule that infers new relations/instances.
    """
    Query = auto()
    """
    Means this is a Query that searches for matches
    """


class DomainSource(Enum):
    """
    The domain source of a variable.
    """

    EXPLICIT = auto()
    """
    Explicitly provided domain.
    """
    DEDUCTION = auto()
    """
    Inferred using deductive reasoning.
    """
    GROUPING = auto()
    """
    Derived from grouping operations.
    """


class ProbabilisticLibrary(StrEnum):
    """
    The top-level packages of the libraries only probabilistic queries need.
    """

    RANDOM_EVENTS = "random_events"
    PROBABILISTIC_MODEL = "probabilistic_model"

    @classmethod
    def missing(cls) -> tuple[ProbabilisticLibrary, ...]:
        """
        :return: The members whose library is not installed.
        """
        return tuple(
            library for library in cls if importlib.util.find_spec(library) is None
        )

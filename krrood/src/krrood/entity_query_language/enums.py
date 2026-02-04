from enum import Enum, auto


class RDREdge(Enum):
    Refinement = "except if"
    """
    Refinement edge, the edge that represents the refinement of an incorrectly fired rule.
    """
    Alternative = "else if"
    """
    Alternative edge, the edge that represents the alternative to the rule that has not fired.
    """
    Next = "also if"
    """
    Next edge, the edge that represents the next rule to be evaluated.
    """
    Then = "then"
    """
    Then edge, the edge that represents the connection to the conclusion.
    """


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


class PredicateType(Enum):
    """
    The type of a predicate.
    """

    SubClassOfPredicate = auto()
    """
    The predicate is an instance of Predicate class.
    """
    DecoratedMethod = auto()
    """
    The predicate is a method decorated with @predicate decorator.
    """

from __future__ import annotations

from enum import StrEnum


class SemanticRole(StrEnum):
    """
    Semantic category of a fragment, determining its colour markup.
    """

    KEYWORD = "keyword"
    """
    EQL structure words — *If*, *Then*, *Find*, *Where*, *Such that*.
    """

    VARIABLE = "variable"
    """
    Type and instance names — *Robot*, *Employee 1*.
    """

    AGGREGATION = "aggregation"
    """
    Aggregation phrases — *sum of*, *number of*, *average of*.
    """

    OPERATOR = "operator"
    """
    Comparator phrases — *is greater than*, *equals*.
    """

    VERB = "verb"
    """
    Lexical verbs of a predicate clause, given as a lemma — *work*, *contain*, *love* —
    realised present-tense (*works*) and negated with do-support (*does not work*) by
    the morphology pass.
    """

    LOGICAL = "logical"
    """
    Logical connectives — *and*, *or*, *not*, *for all*, *there exists*.
    """

    LITERAL = "literal"
    """
    Literal values — ``42``, ``"hello"``, ``True``.
    """

    ATTRIBUTE = "attribute"
    """
    Attribute and field names — *battery*, *tasks*, *name*.
    """

    PLAIN = "plain"
    """
    Neutral connecting text with no special colour.
    """

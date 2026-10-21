"""
Tests that a query evaluated while another query is being evaluated is an evaluation of
its own, giving the same answer it gives when evaluated alone.
"""

from krrood.entity_query_language.factories import (
    add,
    alternative,
    an,
    deduced_variable,
    entity,
    inference,
    refinement,
    variable,
)
from krrood.entity_query_language.query.query import Query

from ...dataset.query_evaluated_during_another_evaluation import (
    KindConcludedOnIteration,
    NarrowOpeningKind,
    Opening,
    OpeningKind,
    WideOpeningKind,
)

# %% rule tree


def build_opening_kind_rule_tree(opening_to_classify: Opening) -> Query:
    """
    :param opening_to_classify: The case the rule tree classifies.
    :return: A rule tree concluding the kind of *opening_to_classify* by its width.
    """
    opening = variable(Opening, domain=[opening_to_classify])
    kind = deduced_variable(OpeningKind)
    rule_tree = an(entity(kind).where(opening.width > 0))
    with rule_tree:
        add(kind, inference(OpeningKind)(opening=opening))
        with refinement(opening.width < 1):
            add(kind, inference(NarrowOpeningKind)(opening=opening))
        with alternative(opening.width > 10):
            add(kind, inference(WideOpeningKind)(opening=opening))
    return rule_tree


# %% evaluation inside another evaluation


def test_rule_tree_evaluated_alone_concludes_the_kind():
    """
    The rule tree evaluated on its own concludes the refined kind of its case.
    """
    opening = Opening(width=0.5)

    assert build_opening_kind_rule_tree(opening).tolist() == [
        NarrowOpeningKind(opening=opening)
    ]


def test_rule_tree_evaluated_while_another_query_is_evaluated_concludes_the_kind():
    """
    The rule tree evaluated while another query iterates the domain that runs it
    concludes the same kind it concludes when evaluated on its own.
    """
    opening = Opening(width=0.5)
    kind = variable(
        OpeningKind,
        domain=KindConcludedOnIteration(build_opening_kind_rule_tree(opening)),
    )

    assert an(entity(kind)).tolist() == [NarrowOpeningKind(opening=opening)]

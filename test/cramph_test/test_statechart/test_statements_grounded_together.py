from __future__ import annotations

from cramph.composites import StatementNode, StatementsGroundedTogether
from cramph.data_types import ObservationStateValues
from cramph.nodes_for_testing import NodeSucceedingOnObservingTrue
from krrood.entity_query_language.backends import (
    EntityQueryLanguageGenerativeBackend,
)
from krrood.entity_query_language.factories import a, variable
from krrood.entity_query_language.query.match import Match

# %% helpers


def _statement_of_names(names: list[str]) -> Match[NodeSucceedingOnObservingTrue]:
    """
    :return: A statement with one candidate per name in `names`, ranked in that order.
    """
    return a(NodeSucceedingOnObservingTrue)(
        name=variable(str, domain=names), observation=ObservationStateValues.TRUE
    )


def _grounded_together(*names_per_statement: list[str]) -> StatementsGroundedTogether:
    """
    :return: One statement per list of names, grounded together.
    """
    return StatementsGroundedTogether(
        statements=[
            StatementNode(statement=_statement_of_names(names))
            for names in names_per_statement
        ],
        query_backend=EntityQueryLanguageGenerativeBackend(),
        default_candidate_limit=10,
    )


def _names_of(
    combination: dict[StatementNode, NodeSucceedingOnObservingTrue],
) -> list[str]:
    """
    :return: The names of the candidates of `combination`, in the order of its
        statements.
    """
    return [candidate.name for candidate in combination.values()]


# %% the combinations of the candidates


def test_the_first_combination_takes_the_candidate_each_statement_ranks_first() -> None:
    grounding = _grounded_together(["a1", "a2"], ["b1", "b2"])

    combination = next(iter(grounding))

    assert list(combination) == grounding.statements
    assert _names_of(combination) == ["a1", "b1"]


def test_the_combinations_end_once_a_statement_has_no_candidate_left() -> None:
    grounding = _grounded_together(["a1", "a2"], ["b1"])

    combinations = list(grounding)

    assert [_names_of(combination) for combination in combinations] == [
        ["a1", "b1"],
        ["a2", "b1"],
    ]


# %% a statement going on after the combination it runs


def test_a_statement_goes_on_with_the_candidates_read_before_its_own() -> None:
    """
    The statement goes on with the candidates read for earlier combinations, then with
    those beyond, and never with the one it runs.
    """
    grounding = _grounded_together(["a1", "a2", "a3"], ["b1"])
    combinations = iter(grounding)
    next(combinations)
    combination = next(combinations)
    [statement, _] = grounding.statements
    backend = grounding.query_backend

    grounding.take(combination)

    assert _names_of(combination) == ["a2", "b1"]
    assert [
        statement.next_candidate(backend, 10).name,
        statement.next_candidate(backend, 10).name,
        statement.next_candidate(backend, 10),
    ] == ["a1", "a3", None]


def test_a_statement_going_on_counts_only_its_own_candidate_towards_its_limit() -> None:
    grounding = StatementsGroundedTogether(
        statements=[
            StatementNode(statement=_statement_of_names(["a1", "a2", "a3"])),
            StatementNode(statement=_statement_of_names(["b1"])),
        ],
        query_backend=EntityQueryLanguageGenerativeBackend(),
        default_candidate_limit=2,
    )
    combinations = iter(grounding)
    next(combinations)
    combination = next(combinations)
    [statement, _] = grounding.statements

    grounding.take(combination)

    assert statement.next_candidate(grounding.query_backend, 2).name == "a1"
    assert statement.reached_candidate_limit

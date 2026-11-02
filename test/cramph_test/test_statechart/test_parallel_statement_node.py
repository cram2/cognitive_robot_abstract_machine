from __future__ import annotations

from cramph.composites import (
    Parallel,
    StatementNode,
    ParallelStatementNode,
)
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


def _parallel_statement_node(
    *names_per_statement: list[str],
) -> ParallelStatementNode:
    """
    :return: A node grounding together one statement per list of names.
    """
    statements = [
        StatementNode(statement=_statement_of_names(names))
        for names in names_per_statement
    ]
    return ParallelStatementNode(
        parallel=Parallel(nodes=list(statements)), statements=statements
    )


def _names_of(
    combination: dict[StatementNode, NodeSucceedingOnObservingTrue],
) -> list[str]:
    """
    :return: The names of the candidates of `combination`, in the order of its
        statements.
    """
    return [candidate.name for candidate in combination.values()]


# %% grounding the statements of a parallel together


def test_the_first_combination_takes_the_candidate_each_statement_ranks_first() -> None:
    node = _parallel_statement_node(["a1", "a2"], ["b1", "b2"])

    combination = node.next_combination(EntityQueryLanguageGenerativeBackend(), 10)

    assert list(combination) == node.statements
    assert _names_of(combination) == ["a1", "b1"]


def test_the_combinations_go_on_where_the_last_one_stopped() -> None:
    node = _parallel_statement_node(["a1", "a2"], ["b1"])
    backend = EntityQueryLanguageGenerativeBackend()

    combinations = [
        node.next_combination(backend, 10),
        node.next_combination(backend, 10),
        node.next_combination(backend, 10),
    ]

    assert [_names_of(combination) for combination in combinations[:2]] == [
        ["a1", "b1"],
        ["a2", "b1"],
    ]
    assert combinations[2] is None


def test_stopping_the_grounding_starts_the_combinations_anew() -> None:
    node = _parallel_statement_node(["a1", "a2"], ["b1"])
    backend = EntityQueryLanguageGenerativeBackend()
    node.next_combination(backend, 10)

    node.stop_grounding()

    assert _names_of(node.next_combination(backend, 10)) == ["a1", "b1"]


# %% a statement going on from the candidate it was handed


def test_a_statement_handed_a_candidate_goes_on_with_the_others_read_before() -> None:
    """
    The statement goes on with the candidates read for earlier combinations, then with
    those beyond, and never with the one it was handed.
    """
    node = _parallel_statement_node(["a1", "a2", "a3"], ["b1"])
    backend = EntityQueryLanguageGenerativeBackend()
    node.next_combination(backend, 10)
    combination = node.next_combination(backend, 10)
    [statement, _] = node.statements
    continuation = StatementNode(statement=statement.statement)

    continuation.ground_on_from(
        statement, node.candidates_read_for(statement), combination[statement]
    )

    assert _names_of(combination) == ["a2", "b1"]
    assert [
        continuation.next_candidate(backend, 10).name,
        continuation.next_candidate(backend, 10).name,
        continuation.next_candidate(backend, 10),
    ] == ["a1", "a3", None]

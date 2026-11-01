from __future__ import annotations

from cramph.composites import (
    CandidateCombinations,
    Parallel,
    RankedCandidates,
    StatementNode,
    UnderspecifiedParallel,
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


def _ranked_candidates_of_names(names: list[str]) -> RankedCandidates:
    """
    :return: The candidates of a statement with one candidate per name in `names`.
    """
    return RankedCandidates(
        statement=StatementNode(statement=_statement_of_names(names)),
        query_backend=EntityQueryLanguageGenerativeBackend(),
        default_candidate_limit=10,
    )


def _underspecified_parallel(
    *names_per_statement: list[str],
) -> UnderspecifiedParallel:
    """
    :return: A node grounding together one statement per list of names.
    """
    statements = [
        StatementNode(statement=_statement_of_names(names))
        for names in names_per_statement
    ]
    return UnderspecifiedParallel(
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


# %% combining the candidates of several statements


def test_combinations_take_the_candidates_ranked_first_before_later_ones() -> None:
    """
    Every combination comes up once, and none takes a later candidate of a statement
    before the combinations of the candidates ranked before it.
    """
    combinations = CandidateCombinations(
        rankings=[
            _ranked_candidates_of_names(["a1", "a2"]),
            _ranked_candidates_of_names(["b1", "b2", "b3"]),
        ]
    )

    assert [
        tuple(candidate.name for candidate in combination)
        for combination in combinations
    ] == [
        ("a1", "b1"),
        ("a1", "b2"),
        ("a2", "b1"),
        ("a1", "b3"),
        ("a2", "b2"),
        ("a2", "b3"),
    ]


def test_a_statement_without_candidates_leaves_no_combination() -> None:
    combinations = CandidateCombinations(
        rankings=[
            _ranked_candidates_of_names(["a1"]),
            _ranked_candidates_of_names([]),
        ]
    )

    assert list(combinations) == []


# %% grounding the statements of a parallel together


def test_the_first_combination_takes_the_candidate_each_statement_ranks_first() -> None:
    node = _underspecified_parallel(["a1", "a2"], ["b1", "b2"])

    combination = node.next_combination(EntityQueryLanguageGenerativeBackend(), 10)

    assert list(combination) == node.statements
    assert _names_of(combination) == ["a1", "b1"]


def test_the_combinations_go_on_where_the_last_one_stopped() -> None:
    node = _underspecified_parallel(["a1", "a2"], ["b1"])
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
    node = _underspecified_parallel(["a1", "a2"], ["b1"])
    backend = EntityQueryLanguageGenerativeBackend()
    node.next_combination(backend, 10)

    node.stop_grounding()

    assert _names_of(node.next_combination(backend, 10)) == ["a1", "b1"]


# %% a statement going on from the candidate it was handed


def test_a_statement_handed_a_candidate_goes_on_with_the_others_of_its_ranking() -> None:
    """
    The statement goes on with the candidates its ranking pulled for earlier
    combinations, then with those beyond, and never with the one it was handed.
    """
    node = _underspecified_parallel(["a1", "a2", "a3"], ["b1"])
    backend = EntityQueryLanguageGenerativeBackend()
    node.next_combination(backend, 10)
    combination = node.next_combination(backend, 10)
    [statement, _] = node.statements
    continuation = StatementNode(statement=statement.statement)

    continuation.ground_on_from(node.ranking_of(statement), combination[statement])

    assert _names_of(combination) == ["a2", "b1"]
    assert [
        continuation.next_candidate(backend, 10).name,
        continuation.next_candidate(backend, 10).name,
        continuation.next_candidate(backend, 10),
    ] == ["a1", "a3", None]

"""
Tests grounding several statements as one: the combinations of their candidates, the
variables they share, what a combination has to meet and how many are tried.
"""

from __future__ import annotations

from dataclasses import dataclass

from cramph.composites import (
    CombinationRequirement,
    StatementCombination,
    StatementNode,
    StatementsGroundedTogether,
)
from cramph.data_types import ObservationStateValue
from cramph.nodes_for_testing import NodeSucceedingOnObservingTrue
from krrood.entity_query_language.backends import (
    Answer,
    EntityQueryLanguageGenerativeBackend,
)
from krrood.entity_query_language.factories import a, variable
from krrood.entity_query_language.query.match import Match

# %% helpers


def _statement_of_names(
    names: list[str] | NodeSucceedingOnObservingTrue,
) -> Match[NodeSucceedingOnObservingTrue]:
    """
    :param names: The names of the candidates, ranked in their order, or a variable
        ranging over them.
    :return: A statement with one candidate per name.
    """
    name = variable(str, domain=names) if isinstance(names, list) else names
    return a(NodeSucceedingOnObservingTrue)(
        name=name, observation=ObservationStateValue.TRUE
    )


def _grounded_together(
    *names_per_statement: list[str],
    default_candidate_limit: int = 10,
    requirements: tuple[CombinationRequirement, ...] = (),
) -> StatementsGroundedTogether:
    """
    :return: One statement per list of names, grounded together.
    """
    return StatementsGroundedTogether(
        statements=[
            StatementNode(statement=_statement_of_names(names))
            for names in names_per_statement
        ],
        query_backend=EntityQueryLanguageGenerativeBackend(),
        default_candidate_limit=default_candidate_limit,
        requirements=requirements,
    )


def _names_of(combination: dict[StatementNode, Answer]) -> list[str]:
    """
    :return: The names of the candidates of `combination`, in the order of its
        statements.
    """
    return [answer.value.name for answer in combination.values()]


@dataclass(kw_only=True)
class NamesDiffer(CombinationRequirement):
    """
    Requires the candidates of a combination of two statements to have different names.
    """

    def condition_on(self, *, combination: StatementCombination) -> object:
        return combination.candidates[0].name != combination.candidates[1].name


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


def test_statements_sharing_a_variable_take_one_value_of_it() -> None:
    name = variable(str, domain=["n1", "n2"])
    grounding = StatementsGroundedTogether(
        statements=[
            StatementNode(statement=_statement_of_names(name)),
            StatementNode(statement=_statement_of_names(name)),
        ],
        query_backend=EntityQueryLanguageGenerativeBackend(),
        default_candidate_limit=10,
    )

    combinations = list(grounding)

    assert [_names_of(combination) for combination in combinations] == [
        ["n1", "n1"],
        ["n2", "n2"],
    ]


def test_a_combination_reports_the_values_of_the_shared_variable() -> None:
    name = variable(str, domain=["n1"])
    statement = StatementNode(statement=_statement_of_names(name))
    grounding = StatementsGroundedTogether(
        statements=[statement],
        query_backend=EntityQueryLanguageGenerativeBackend(),
        default_candidate_limit=10,
    )

    [combination] = list(grounding)

    assert combination[statement].variable_values[name._id_] == "n1"


# %% what a combination has to meet


def test_only_the_combinations_meeting_the_requirements_are_yielded() -> None:
    grounding = _grounded_together(
        ["n1", "n2"], ["n1", "n2"], requirements=(NamesDiffer(),)
    )

    combinations = list(grounding)

    assert [_names_of(combination) for combination in combinations] == [
        ["n1", "n2"],
        ["n2", "n1"],
    ]


def test_a_combination_failing_the_requirements_counts_towards_the_limit() -> None:
    grounding = _grounded_together(
        ["n1", "n2"],
        ["n1", "n2"],
        default_candidate_limit=1,
        requirements=(NamesDiffer(),),
    )

    combinations = list(grounding)

    assert combinations == []
    assert grounding.reached_candidate_limit


# %% a statement grounded on its own again


def test_a_statement_running_anew_leaves_out_the_candidate_handed_to_it() -> None:
    """
    The statement goes on with the candidates other than the one a combination handed
    it, in their order.
    """
    grounding = _grounded_together(["a1", "a2", "a3"], ["b1"])
    combinations = iter(grounding)
    next(combinations)
    combination = next(combinations)
    [statement, _] = grounding.statements
    statement.take(answer=combination[statement])
    backend = grounding.query_backend

    names = [
        statement.next_candidate(
            query_backend=backend, default_candidate_limit=10
        ).name,
        statement.next_candidate(
            query_backend=backend, default_candidate_limit=10
        ).name,
        statement.next_candidate(query_backend=backend, default_candidate_limit=10),
    ]

    assert _names_of(combination) == ["a2", "b1"]
    assert names == ["a1", "a3", None]

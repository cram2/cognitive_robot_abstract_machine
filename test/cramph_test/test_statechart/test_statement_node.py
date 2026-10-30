from __future__ import annotations

from dataclasses import dataclass


from cramph.composites import (
    ChildChooser,
    ChildChooserAccess,
    Parallel,
    Sequence,
    StatementNode,
    TryAll,
    TryInOrder,
)
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.node import EndStatechart, StatechartNode
from cramph.nodes_for_testing import NodeSucceedingOnObservingTrue
from cramph.statechart import Statechart
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.entity_query_language.backends import (
    EntityQueryLanguageGenerativeBackend,
)
from krrood.entity_query_language.factories import a, variable
from krrood.entity_query_language.query.match import Match

# %% mimics


@dataclass
class ChooserRunningEveryCandidate(ChildChooser[StatementNode]):
    """
    Runs the next candidate of a statement node's statement as its child.
    """

    default_candidate_limit: int = 10
    """
    How many candidates a statement without a limit of its own yields.
    """

    def choose_child(self, node: StatementNode) -> StatechartNode | None:
        return node.next_candidate(
            EntityQueryLanguageGenerativeBackend(), self.default_candidate_limit
        )


# %% helpers


def _succeeding_statement(name: str) -> Match[NodeSucceedingOnObservingTrue]:
    """
    :return: A statement of a node that succeeds as soon as it observes.
    """
    return a(NodeSucceedingOnObservingTrue)(
        name=name, observation=ObservationStateValues.TRUE
    )


def _statement_of_names(names: list[str]) -> Match[NodeSucceedingOnObservingTrue]:
    """
    :return: A statement with one candidate per name in `names`.
    """
    return a(NodeSucceedingOnObservingTrue)(
        name=variable(str, domain=names), observation=ObservationStateValues.TRUE
    )


def _run(
    executor: StatechartExecutor, chooser: ChildChooser, goal: StatechartNode
) -> Statechart:
    """
    Run `goal` with `chooser` until it succeeded.

    :return: The statechart that ran.
    """
    executor.context.add_extension(ChildChooserAccess(choosers=[chooser]))
    statechart = Statechart(context=executor.context)
    statechart.add_node(goal)
    statechart.add_node(EndStatechart.when_true(goal))
    executor.compile(statechart)
    executor.tick_until_end(timeout=100)
    return statechart


# %% a statement among the children of a control structure node


def test_a_statement_handed_to_a_language_node_becomes_a_statement_node() -> None:
    statement = _succeeding_statement("step")

    for language_node in (
        Sequence(nodes=[statement]),
        TryInOrder(nodes=[statement]),
        TryAll(nodes=[statement]),
        Parallel(nodes=[statement]),
    ):
        [step] = language_node.nodes
        assert isinstance(step, StatementNode)
        assert step.statement is statement


def test_a_statement_added_to_a_sequence_becomes_a_statement_node() -> None:
    statement = _succeeding_statement("step")
    sequence = Sequence()

    sequence.add_node(statement)

    [step] = sequence.nodes
    assert isinstance(step, StatementNode)
    assert step.statement is statement


def test_a_statement_inserted_into_a_joined_sequence_runs_as_a_statement_node(
    statechart_executor: StatechartExecutor,
) -> None:
    first = NodeSucceedingOnObservingTrue(
        name="first", observation=ObservationStateValues.TRUE
    )
    statement = _succeeding_statement("inserted")
    sequence = Sequence(nodes=[first])
    statechart_executor.context.add_extension(
        ChildChooserAccess(choosers=[ChooserRunningEveryCandidate()])
    )
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(sequence)

    sequence.insert_after(first, statement)

    inserted = sequence.nodes[1]
    assert isinstance(inserted, StatementNode)
    assert inserted.statement is statement
    assert inserted.parent_node is sequence


def test_a_statement_replacing_a_step_runs_as_a_statement_node(
    statechart_executor: StatechartExecutor,
) -> None:
    replaced = NodeSucceedingOnObservingTrue(
        name="replaced", observation=ObservationStateValues.TRUE
    )
    statement = _succeeding_statement("replacement")
    sequence = Sequence(nodes=[replaced])
    statechart_executor.context.add_extension(
        ChildChooserAccess(choosers=[ChooserRunningEveryCandidate()])
    )
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(sequence)

    statechart.replace(replaced, statement)

    [replacement] = sequence.nodes
    assert isinstance(replacement, StatementNode)
    assert replacement.statement is statement


def test_a_sequence_runs_the_node_its_statement_grounds_to(
    statechart_executor: StatechartExecutor,
) -> None:
    sequence = Sequence(nodes=[_succeeding_statement("grounded")])

    _run(statechart_executor, ChooserRunningEveryCandidate(), sequence)

    [step] = sequence.nodes
    [grounded] = step.grounded_nodes
    assert isinstance(grounded, NodeSucceedingOnObservingTrue)
    assert grounded.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert sequence.life_cycle_state == LifeCycleValues.SUCCEEDED


# %% the candidates of a statement


def test_a_statement_yields_no_more_candidates_than_the_default_limit() -> None:
    node = StatementNode(statement=_statement_of_names(["first", "second"]))
    backend = EntityQueryLanguageGenerativeBackend()

    first = node.next_candidate(backend, default_candidate_limit=1)

    assert first.name == "first"
    assert node.next_candidate(backend, default_candidate_limit=1) is None
    assert node.reached_candidate_limit


def test_the_limit_of_a_statement_outranks_the_default_limit() -> None:
    statement = _statement_of_names(["first", "second"]).limit(2)
    node = StatementNode(statement=statement)

    node.next_candidate(
        EntityQueryLanguageGenerativeBackend(), default_candidate_limit=1
    )

    assert node.candidate_limit == statement._limit_


def test_a_statement_that_ran_out_of_candidates_did_not_reach_its_limit() -> None:
    node = StatementNode(statement=_statement_of_names(["only"]))
    backend = EntityQueryLanguageGenerativeBackend()

    node.next_candidate(backend, default_candidate_limit=2)

    assert node.next_candidate(backend, default_candidate_limit=2) is None
    assert not node.reached_candidate_limit


def test_grounding_again_after_stopping_starts_from_the_first_candidate() -> None:
    node = StatementNode(statement=_statement_of_names(["first", "second"]))
    backend = EntityQueryLanguageGenerativeBackend()
    node.next_candidate(backend, default_candidate_limit=2)

    node.stop_grounding()

    assert node.next_candidate(backend, default_candidate_limit=2).name == "first"


# %% introspecting the node


def test_the_class_diagram_resolves_the_type_hints_of_a_statement_node() -> None:
    """
    The ORM is generated from the class diagram, which has to resolve every field.
    """
    ClassDiagram(classes=[StatementNode])

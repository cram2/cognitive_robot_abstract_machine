"""
Running the node an underspecified statement grounds to, the one node a definite
statement describes, and the control structures statements state the nodes of.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cramph.composites import (
    ChildChooser,
    ChildChooserAccess,
    DefiniteStatementNode,
    Parallel,
    Sequence,
    StatementBindings,
    StatementNode,
    TryAll,
    TryInOrder,
)
from cramph.data_types import LifeCycleValue, ObservationStateValue
from cramph.exceptions import ControlStructureStatementLeavesAChoiceError
from cramph.executor import StatechartExecutor
from cramph.node import (
    EndStatechart,
    FailsOnObservingFalse,
    NodeArtifacts,
    StatechartNode,
    SucceedsOnObservingTrue,
)
from cramph.nodes_for_testing import NodeSucceedingOnObservingTrue
from cramph.statechart import Statechart
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.entity_query_language.backends import (
    EntityQueryLanguageGenerativeBackend,
)
from krrood.entity_query_language.exceptions import MultipleSolutionFound
from krrood.entity_query_language.factories import a, the, variable
from krrood.entity_query_language.query.match import Match
from krrood.symbolic_math.symbolic_math import Scalar

# %% mimics


@dataclass(kw_only=True)
class ChooserRunningEveryCandidate(ChildChooser[StatementNode]):
    """
    Runs the next candidate of a statement node's statement as its child.
    """

    default_candidate_limit: int = 10
    """
    How many candidates a statement without a limit of its own yields.
    """

    def choose_child(self, *, node: StatementNode) -> StatechartNode | None:
        return node.next_candidate(
            query_backend=EntityQueryLanguageGenerativeBackend(),
            default_candidate_limit=self.default_candidate_limit,
        )


@dataclass(kw_only=True)
class ChooserRunningTheDescribedNode(ChildChooser[DefiniteStatementNode]):
    """
    Runs the one node a definite statement describes under the values the other
    statements bound, once.
    """

    def choose_child(self, *, node: DefiniteStatementNode) -> StatechartNode | None:
        if node.latest_child is not None:
            return None
        return node.sole_candidate(
            query_backend=EntityQueryLanguageGenerativeBackend(),
            bound_values=node.statement_bindings.values_bound_by_others(
                statement_id=node._node_id
            ),
        )


@dataclass(eq=False, repr=False, kw_only=True)
class SequenceOfItsOwnSteps(Sequence):
    """
    A sequence that creates its steps itself instead of being handed them.
    """

    step: str
    """
    The name of its one step.
    """

    nodes: list[StatechartNode] = field(default_factory=list, init=False)
    """
    The one step, created with the sequence.
    """

    def __post_init__(self) -> None:
        self.nodes = [
            NodeSucceedingOnObservingTrue(
                name=self.step, observation=ObservationStateValue.TRUE
            )
        ]
        super().__post_init__()


@dataclass(eq=False, repr=False, kw_only=True)
class StepWithATool(SucceedsOnObservingTrue, FailsOnObservingFalse, StatechartNode):
    """
    A step that succeeds with a working tool and fails with a broken one, named after
    what it does and the tool it uses.
    """

    step: str
    """
    What the step does.
    """

    tool: str
    """
    The tool the step uses.
    """

    broken_tools: list[str] = field(default_factory=list)
    """
    The tools the step fails with.
    """

    def __post_init__(self) -> None:
        self.name = f"{self.step} with {self.tool}"
        super().__post_init__()

    def build_artifacts(self) -> NodeArtifacts:
        """
        Observes False with a tool of :attr:`broken_tools`, True with any other.
        """
        working = (
            ObservationStateValue.FALSE
            if self.tool in self.broken_tools
            else (ObservationStateValue.TRUE)
        )
        return NodeArtifacts(observation=Scalar(float(working)))


# %% helpers


def _succeeding_statement(name: str) -> Match[NodeSucceedingOnObservingTrue]:
    """
    :return: A statement of a node that succeeds as soon as it observes.
    """
    return a(NodeSucceedingOnObservingTrue)(
        name=name, observation=ObservationStateValue.TRUE
    )


def _statement_of_names(names: list[str]) -> Match[NodeSucceedingOnObservingTrue]:
    """
    :return: A statement with one candidate per name in `names`.
    """
    return a(NodeSucceedingOnObservingTrue)(
        name=variable(str, domain=names), observation=ObservationStateValue.TRUE
    )


def _run(
    executor: StatechartExecutor, choosers: list[ChildChooser], goal: StatechartNode
) -> Statechart:
    """
    Run `goal` with `choosers` until it succeeded.

    :return: The statechart that ran.
    """
    executor.context.add_extension(extension=ChildChooserAccess(choosers=choosers))
    statechart = Statechart(context=executor.context)
    statechart.add_node(node=goal)
    statechart.add_node(node=EndStatechart.when_true(node=goal))
    executor.compile(statechart=statechart)
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

    sequence.add_node(node=statement)

    [step] = sequence.nodes
    assert isinstance(step, StatementNode)
    assert step.statement is statement


def test_a_statement_inserted_into_a_joined_sequence_runs_as_a_statement_node(
    statechart_executor: StatechartExecutor,
) -> None:
    first = NodeSucceedingOnObservingTrue(
        name="first", observation=ObservationStateValue.TRUE
    )
    statement = _succeeding_statement("inserted")
    sequence = Sequence(nodes=[first])
    statechart_executor.context.add_extension(
        extension=ChildChooserAccess(choosers=[ChooserRunningEveryCandidate()])
    )
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=sequence)

    sequence.insert_after(reference=first, node=statement)

    inserted = sequence.nodes[1]
    assert isinstance(inserted, StatementNode)
    assert inserted.statement is statement
    assert inserted.parent_node is sequence


def test_a_statement_replacing_a_step_runs_as_a_statement_node(
    statechart_executor: StatechartExecutor,
) -> None:
    replaced = NodeSucceedingOnObservingTrue(
        name="replaced", observation=ObservationStateValue.TRUE
    )
    statement = _succeeding_statement("replacement")
    sequence = Sequence(nodes=[replaced])
    statechart_executor.context.add_extension(
        extension=ChildChooserAccess(choosers=[ChooserRunningEveryCandidate()])
    )
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=sequence)

    statechart.replace(node=replaced, replacement=statement)

    [replacement] = sequence.nodes
    assert isinstance(replacement, StatementNode)
    assert replacement.statement is statement


def test_a_sequence_runs_the_node_its_statement_grounds_to(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A statement among the nodes of a sequence runs the node it grounds to.
    """
    sequence = Sequence(nodes=[_succeeding_statement("grounded")])

    _run(statechart_executor, [ChooserRunningEveryCandidate()], sequence)

    [step] = sequence.nodes
    [grounded] = step.grounded_nodes
    assert isinstance(grounded, NodeSucceedingOnObservingTrue)
    assert grounded.life_cycle_state == LifeCycleValue.SUCCEEDED
    assert sequence.life_cycle_state == LifeCycleValue.SUCCEEDED


# %% the candidates of a statement


def test_a_statement_yields_no_more_candidates_than_the_default_limit() -> None:
    node = StatementNode(statement=_statement_of_names(["first", "second"]))
    backend = EntityQueryLanguageGenerativeBackend()

    first = node.next_candidate(query_backend=backend, default_candidate_limit=1)

    assert first.name == "first"
    assert node.next_candidate(query_backend=backend, default_candidate_limit=1) is None
    assert node.reached_candidate_limit


def test_the_limit_of_a_statement_outranks_the_default_limit() -> None:
    statement = _statement_of_names(["first", "second"]).limit(n=2)
    node = StatementNode(statement=statement)

    node.next_candidate(
        query_backend=EntityQueryLanguageGenerativeBackend(), default_candidate_limit=1
    )

    assert node.candidate_limit == statement._limit_


def test_a_statement_that_ran_out_of_candidates_did_not_reach_its_limit() -> None:
    node = StatementNode(statement=_statement_of_names(["only"]))
    backend = EntityQueryLanguageGenerativeBackend()

    node.next_candidate(query_backend=backend, default_candidate_limit=2)

    assert node.next_candidate(query_backend=backend, default_candidate_limit=2) is None
    assert not node.reached_candidate_limit


def test_a_statement_that_never_grounded_did_not_reach_its_limit() -> None:
    """
    A statement handed its candidate rather than grounding one has pulled none.
    """
    node = StatementNode(statement=_statement_of_names(["only"]))

    assert not node.reached_candidate_limit


def test_grounding_again_after_stopping_starts_from_the_first_candidate() -> None:
    node = StatementNode(statement=_statement_of_names(["first", "second"]))
    backend = EntityQueryLanguageGenerativeBackend()
    node.next_candidate(query_backend=backend, default_candidate_limit=2)

    node.stop_grounding()

    assert (
        node.next_candidate(query_backend=backend, default_candidate_limit=2).name
        == "first"
    )


# %% statements sharing a variable


def test_a_statement_reusing_a_variable_takes_the_value_an_earlier_statement_chose(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    The taking step fails with the first tool and goes on to the second, so the putting
    step, which could use either, has to take the second as well.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    sequence = Sequence(
        nodes=[
            a(StepWithATool)(step="take", tool=tool, broken_tools=["hammer"]),
            a(StepWithATool)(step="put", tool=tool),
        ]
    )

    _run(statechart_executor, [ChooserRunningEveryCandidate()], sequence)

    taking, putting = sequence.nodes
    [*failed, took] = taking.grounded_nodes
    [put] = putting.grounded_nodes
    assert [each.life_cycle_state for each in failed] == [LifeCycleValue.FAILED]
    assert put.tool == took.tool
    assert put.tool != failed[0].tool


def test_a_statement_that_is_reset_releases_the_variables_it_bound(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    Once a statement is reset, the values its chosen candidate gave its variables no
    longer bind any other statement.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    sequence = Sequence(nodes=[a(StepWithATool)(step="take", tool=tool)])
    _run(statechart_executor, [ChooserRunningEveryCandidate()], sequence)
    [taking] = sequence.nodes
    bindings = statechart_executor.context.require_extension(
        extension_type=StatementBindings
    )
    assert bindings.values_bound_by_others(statement_id="another statement") == {
        tool._id_: taking.grounded_nodes[-1].tool
    }

    taking.on_reset()

    assert bindings.values_bound_by_others(statement_id="another statement") == {}


def test_a_statement_keeps_a_value_another_statement_bound() -> None:
    """
    A statement cannot take over a variable another statement bound first.
    """
    bindings = StatementBindings()
    tool = variable(str, domain=["hammer", "wrench"])
    bindings.bind(statement_id="take", variable_values={tool._id_: "wrench"})

    bindings.bind(statement_id="put", variable_values={tool._id_: "hammer"})

    assert bindings.values_bound_by_others(statement_id="put") == {tool._id_: "wrench"}
    assert bindings.values_bound_by_others(statement_id="take") == {}


def test_a_statement_choosing_again_replaces_what_it_bound() -> None:
    """
    A statement choosing another candidate binds its variables to that candidate's
    values instead.
    """
    bindings = StatementBindings()
    tool = variable(str, domain=["hammer", "wrench"])
    bindings.bind(statement_id="take", variable_values={tool._id_: "hammer"})

    bindings.bind(statement_id="take", variable_values={tool._id_: "wrench"})

    assert bindings.values_bound_by_others(statement_id="put") == {tool._id_: "wrench"}


# %% definite statements


def test_a_definite_statement_handed_to_a_language_node_becomes_a_definite_statement_node() -> (
    None
):
    """
    A `the(...)` statement among the nodes of a control structure is run by a node that
    describes its one node instead of choosing among candidates.
    """
    statement = the(StepWithATool)(step="put", tool="hammer")

    [step] = Sequence(nodes=[statement]).nodes

    assert type(step) is DefiniteStatementNode
    assert step.statement is statement


def test_a_definite_statement_describes_the_one_node_the_bound_values_leave() -> None:
    """
    Under the values other statements bound, a definite statement describes one node.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    node = DefiniteStatementNode(statement=the(StepWithATool)(step="put", tool=tool))

    described = node.sole_candidate(
        query_backend=EntityQueryLanguageGenerativeBackend(),
        bound_values={tool._id_: "wrench"},
    )

    assert (described.step, described.tool) == ("put", "wrench")


def test_a_definite_statement_leaving_a_choice_raises() -> None:
    """
    A definite statement whose variables are not all bound describes several nodes.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    node = DefiniteStatementNode(statement=the(StepWithATool)(step="put", tool=tool))

    with pytest.raises(MultipleSolutionFound):
        node.sole_candidate(
            query_backend=EntityQueryLanguageGenerativeBackend(), bound_values={}
        )


def test_a_definite_statement_runs_with_the_value_the_statement_before_it_chose(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    The taking step fails with the first tool and goes on to the second, which leaves
    the putting step one tool to describe.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    sequence = Sequence(
        nodes=[
            a(StepWithATool)(step="take", tool=tool, broken_tools=["hammer"]),
            the(StepWithATool)(step="put", tool=tool),
        ]
    )

    _run(
        statechart_executor,
        [ChooserRunningEveryCandidate(), ChooserRunningTheDescribedNode()],
        sequence,
    )

    taking, putting = sequence.nodes
    [put] = putting.grounded_nodes
    assert put.tool == taking.grounded_nodes[-1].tool


# %% control structures given as statements


def test_a_statement_stating_the_nodes_of_a_control_structure_becomes_that_structure() -> (
    None
):
    """
    An `a(...)` statement of a control structure that states its nodes is that
    structure, whose nodes are the statements it states.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    taking = a(StepWithATool)(step="take", tool=tool)
    putting = the(StepWithATool)(step="put", tool=tool)
    statement = a(Sequence)(nodes=[taking, putting])

    [structure] = Sequence(nodes=[statement]).nodes

    assert type(structure) is Sequence
    assert structure.statement is statement
    assert [(type(node), node.statement) for node in structure.nodes] == [
        (StatementNode, taking),
        (DefiniteStatementNode, putting),
    ]


def test_a_statement_of_a_structure_creating_its_own_nodes_stays_a_statement() -> None:
    """
    A control structure that creates its nodes itself, as an action does, is chosen
    among candidates like any other statement.
    """
    statement = a(SequenceOfItsOwnSteps)(step=variable(str, domain=["first", "second"]))

    [step] = Sequence(nodes=[statement]).nodes

    assert type(step) is StatementNode


def test_a_statement_of_a_control_structure_leaving_a_choice_of_its_own_raises() -> (
    None
):
    """
    A control structure given as a statement is built from what the statement states, so
    it cannot leave anything but its nodes open.
    """
    statement = a(Sequence)(
        name=variable(str, domain=["first", "second"]),
        nodes=[_succeeding_statement("step")],
    )

    with pytest.raises(ControlStructureStatementLeavesAChoiceError):
        Sequence(nodes=[statement])


# %% introspecting the node


def test_the_class_diagram_resolves_the_type_hints_of_a_statement_node() -> None:
    """
    The ORM is generated from the class diagram, which has to resolve every field.
    """
    ClassDiagram(classes=[StatementNode])

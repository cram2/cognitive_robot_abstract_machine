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
    DefiniteDescription,
    DefiniteStatementNode,
    DescribedNodes,
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
        return node.ground_described_node(
            query_backend=EntityQueryLanguageGenerativeBackend()
        )


@dataclass(kw_only=True)
class ChooserRunningANodeOfAnotherType(ChildChooser[StatementNode]):
    """
    Runs, once, a node of another type than the statement describes.
    """

    def choose_child(self, *, node: StatementNode) -> StatechartNode | None:
        if node.latest_child is not None:
            return None
        return NodeSucceedingOnObservingTrue(
            name="another node", observation=ObservationStateValue.TRUE
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


def _start(
    executor: StatechartExecutor, choosers: list[ChildChooser], goal: StatechartNode
) -> Statechart:
    """
    Start running `goal` with `choosers`, without running it to its end.

    :return: The statechart that started.
    """
    executor.context.add_extension(extension=ChildChooserAccess(choosers=choosers))
    statechart = Statechart(context=executor.context)
    statechart.add_node(node=goal)
    executor.compile(statechart=statechart)
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
    _start(statechart_executor, [ChooserRunningEveryCandidate()], sequence)
    [taking] = sequence.nodes
    bindings = statechart_executor.context.require_extension(
        extension_type=StatementBindings
    )
    assert bindings.values_bound_by_others(statement_id="another statement") == {
        tool._id_: taking.grounded_nodes[-1].tool
    }

    taking.on_reset()

    assert bindings.values_bound_by_others(statement_id="another statement") == {}


def test_the_values_statements_bound_are_released_once_the_statechart_ran(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    The choices of a statechart that ran bind nothing in the next one run in the same
    context.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    _run(
        statechart_executor,
        [ChooserRunningEveryCandidate()],
        Sequence(nodes=[a(StepWithATool)(step="take", tool=tool)]),
    )

    bindings = statechart_executor.context.require_extension(
        extension_type=StatementBindings
    )
    assert bindings.bindings == {}


def test_a_statement_grounds_anew_once_another_statement_bound_another_value(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A statement whose candidates were opened under one value of a shared variable does
    not go on with them once another statement bound the variable to another value.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    putting = StatementNode(statement=a(StepWithATool)(step="put", tool=tool))
    never_succeeding = NodeSucceedingOnObservingTrue(
        name="never succeeding", observation=ObservationStateValue.UNKNOWN
    )
    _start(
        statechart_executor,
        [ChooserRunningEveryCandidate()],
        Sequence(nodes=[never_succeeding, putting]),
    )
    bindings = putting.statement_bindings
    backend = EntityQueryLanguageGenerativeBackend()
    bindings.bind(statement_id="take", variable_values={tool._id_: "hammer"})
    putting.next_candidate(query_backend=backend, default_candidate_limit=10)

    bindings.bind(statement_id="take", variable_values={tool._id_: "wrench"})

    assert (
        putting.next_candidate(query_backend=backend, default_candidate_limit=10).tool
        == "wrench"
    )


def test_a_statement_choosing_a_node_of_another_type_binds_nothing(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A child holding no node of the statement's type gives the statement's variables no
    value.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    sequence = Sequence(nodes=[a(StepWithATool)(step="take", tool=tool)])

    _start(statechart_executor, [ChooserRunningANodeOfAnotherType()], sequence)

    [taking] = sequence.nodes
    assert [child.name for child in taking.chosen_children] == ["another node"]
    assert taking.grounded_nodes == []
    assert taking.statement_bindings.bindings == {}


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

    description = node.describe(
        query_backend=EntityQueryLanguageGenerativeBackend(),
        bound_values={tool._id_: "wrench"},
    )

    described = description.sole_node.instance
    assert description.described_nodes == DescribedNodes.ONE
    assert (described.step, described.tool) == ("put", "wrench")


def test_a_definite_statement_leaving_a_choice_describes_several_nodes() -> None:
    """
    A definite statement whose variables are not all bound describes several nodes.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    node = DefiniteStatementNode(statement=the(StepWithATool)(step="put", tool=tool))

    description = node.describe(
        query_backend=EntityQueryLanguageGenerativeBackend(), bound_values={}
    )

    assert description == DefiniteDescription(described_nodes=DescribedNodes.SEVERAL)


def test_a_definite_statement_whose_condition_the_bound_values_break_describes_no_node() -> (
    None
):
    """
    The statement only describes a step with the second tool, so with the first tool
    bound it describes none.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    putting = the(StepWithATool)(step="put", tool=tool)
    node = DefiniteStatementNode(statement=putting.where(putting.tool == "wrench"))

    description = node.describe(
        query_backend=EntityQueryLanguageGenerativeBackend(),
        bound_values={tool._id_: "hammer"},
    )

    assert description == DefiniteDescription(described_nodes=DescribedNodes.NONE)


def test_a_definite_statement_describing_several_nodes_fails_without_running_one(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A definite statement whose tool no statement chose has no single node to run, and
    remembers that it described several.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    sequence = Sequence(nodes=[the(StepWithATool)(step="put", tool=tool)])

    _start(statechart_executor, [ChooserRunningTheDescribedNode()], sequence)

    [putting] = sequence.nodes
    assert putting.chosen_children == []
    assert putting.ran_out_of_children
    assert putting.latest_description == DefiniteDescription(
        described_nodes=DescribedNodes.SEVERAL
    )


def test_a_definite_statement_binds_the_variables_it_shares_with_the_statements_after_it(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    The definite step describes the second tool only, and the step after it, which would
    take the first, takes that tool as well.
    """
    tool = variable(str, domain=["hammer", "wrench"])
    taking = the(StepWithATool)(step="take", tool=tool)
    sequence = Sequence(
        nodes=[
            taking.where(taking.tool == "wrench"),
            a(StepWithATool)(step="put", tool=tool),
        ]
    )

    _run(
        statechart_executor,
        [ChooserRunningEveryCandidate(), ChooserRunningTheDescribedNode()],
        sequence,
    )

    _, putting = sequence.nodes
    assert [step.tool for step in putting.grounded_nodes] == ["wrench"]


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

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field

import rustworkx as rx
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Callable,
    Self,
)

import krrood.symbolic_math.symbolic_math as sm
from cramph.plotters.gantt_chart_plotter import HistoryGanttChartPlotter
from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from krrood.rustworkx_utils.graph_visualizer_base import (
    GraphLayout,
    GraphVisualizerBackend,
    GraphVisualizerBase,
)
from cramph.context import ContextExtension, StatechartContext
from cramph.data_types import (
    StatechartJSONKey,
    TransitionKind,
    ObservationStateValues,
)
from cramph.exceptions import (
    EmptyStatechartError,
    NodesMissingContextExtensionsError,
    PrerequisiteNotExpandedError,
    StatechartAlreadyCompiledError,
    NotHeldByControlStructureNodeError,
    NotInStatechartError,
    RemovedNodeStillReferencedError,
)
from cramph.node import (
    CancelStatechart,
    CompositeNode,
    DerivedConditionVariable,
    DeserializedNodeTracker,
    EndStatechart,
    GenericStatechartNode,
    StatechartNode,
    TransitionCondition,
)
from cramph.composites import CompositeNodeChoosingItsChild, ControlStructureNode
from cramph.compiler import StatechartCompiler
from cramph.plotters.graphviz import StatechartGraphviz
from cramph.plotters.interactive_graph import StatechartGraphVisualizer
from cramph.state import LastObservationState, LifeCycleState, ObservationState
from cramph.state_history import StateHistory, StateHistoryItem
from semantic_digital_twin.world_description.world_entity import (
    WorldEntityReferenceWriter,
)

if TYPE_CHECKING:
    from types import TracebackType

    from krrood.entity_query_language.query.match import Match

logger = logging.getLogger(__name__)


# %% modifying a statechart


@dataclass(eq=False)
class ReplacedNode:
    """
    A node another node was put in the place of, with what it had before, so that a
    modification that fails afterwards can put it back.
    """

    node: StatechartNode
    """
    The node that was replaced.
    """

    replacement: StatechartNode
    """
    The node put in its place.
    """

    former_parent: CompositeNode | None
    """
    The parent the node had, or None for a top-level node.
    """

    former_conditions: dict[TransitionKind, sm.Scalar]
    """
    The transitions the former parent decided for the node, as they were.
    """

    @classmethod
    def before_replacing(
        cls, node: StatechartNode, replacement: StatechartNode
    ) -> ReplacedNode:
        """
        :param node: The node about to be replaced.
        :param replacement: The node about to be put in its place.
        :return: The node as it is now.
        """
        return cls(
            node=node,
            replacement=replacement,
            former_parent=node.parent_node,
            former_conditions={
                transition_kind: node.get_condition(transition_kind)
                for transition_kind in TransitionKind.owner_wired_kinds()
            },
        )

    def put_back(self) -> None:
        """
        Gives the node its former parent, its place among that parent's nodes, and the
        transitions that parent decided for it.
        """
        self.node.parent_node = self.former_parent
        for transition_kind, condition in self.former_conditions.items():
            self.node.set_condition(transition_kind, condition)
        if self.former_parent is not None:
            self.former_parent.list_in_place_of(self.replacement, self.node)


@dataclass
class StatechartModification:
    """
    Batches changes to a statechart, so that a compiled statechart compiles again only
    once, when the outermost of any nested modifications ends.

    If the block raises, the nodes added inside it are dropped again and the statechart
    stays as it was.
    """

    statechart: Statechart
    """
    The statechart being changed.
    """

    _node_count_at_start: int = field(init=False, default=0)
    """
    How many nodes the statechart held when this modification started.
    """

    _replaced_count_at_start: int = field(init=False, default=0)
    """
    How many nodes were replaced in the open modifications when this one started.
    """

    def __enter__(self) -> Self:
        self._node_count_at_start = len(self.statechart.nodes)
        self._replaced_count_at_start = len(self.statechart._replaced_nodes)
        self.statechart._open_modifications += 1
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        statechart = self.statechart
        statechart._open_modifications -= 1
        if exception is not None:
            statechart._put_back_replaced_nodes_from(self._replaced_count_at_start)
            statechart._drop_nodes_from(self._node_count_at_start)
        if statechart._open_modifications == 0:
            statechart._replaced_nodes.clear()
        compiler = statechart.compiler
        if statechart._open_modifications > 0 or not compiler.is_compiled:
            return
        if not compiler.changed_since_compile:
            return
        compiler.compile()


# %% the statechart


@dataclass
class Statechart(SubclassJSONSerializer):
    """
    Represents a statechart.
    A statechart is a directed graph of nodes and edges.
    Nodes have two states: observation state and life cycle state.
    Life cycle states indicate the current state in the life cycle of the node:
        - NOT_STARTED: the node has not started yet.
        - RUNNING: the node is running.
        - PAUSED: the node is paused.
        - SUCCEEDED: the node's success condition ended it.
        - FAILED: the node's fail condition ended it.
        - INTERRUPTED: the node's interrupt condition ended it, or one of its ancestors
                       ended.
    Out of these 6 states, nodes are only "active" if they are in the RUNNING state, and
    the last 3 are terminal: they are the node's outcome and only left by a reset.
    Observation states indicate the current observation of the node:
        - TrinaryFalse: the thing the node is observing is not True.
        - TrinaryUnknown: the node cannot determine the truth value yet, or is not
                          observing at all.
        - TrinaryTrue: the thing the node is observing is True.
    Only a running node observes. A node that has not started or has reached a terminal
    state reports TrinaryUnknown from the next tick on, while a paused node
    keeps its last observation because it resumes and observes again.
    An observation is re-evaluated every tick and may change in both directions, whereas a
    outcome is latched. A condition is two-valued and may read either through a
    predicate: the observation state of a node through :attr:`~cramph.node.StatechartNode.observes_true` or
    :attr:`~cramph.node.StatechartNode.observes_false`, or its life cycle state through e.g. :attr:`~cramph.node.StatechartNode.is_failed`. What a
    node observes is gone once the tick it ended in is over, so a condition that outlives the node it
    reads has to read something that outlasts it: :attr:`~cramph.node.StatechartNode.last_observed_true` keeps whether
    the observation the node took most recently was True, whatever its outcome, and a
    life cycle predicate keeps the outcome. Every tick settles the whole statechart before
    it returns, see :class:`~cramph.compiled_tick.CompiledTick`, so a node waiting on another node's
    outcome starts on the tick that outcome is reached, however deeply either is nested.
    Nodes are connected with edges, or transitions.
    There are 6 types of transitions:
        - start condition: If True, the node transitions from NOT_STARTED to RUNNING,
                           or to PAUSED while its pause condition is True.
        - pause condition: If True, the node transitions from RUNNING to PAUSED.
                           If False, the node transitions from PAUSED to RUNNING.
        - success condition: If True, the node ends from RUNNING or PAUSED as SUCCEEDED.
        - fail condition: If True, the node ends from RUNNING or PAUSED as FAILED.
        - interrupt condition: If True, the node ends from RUNNING or PAUSED as
                               INTERRUPTED.
        - reset condition: If True, the node transitions from any state to NOT_STARTED.
    The condition that ends a node decides its outcome; what the node observes at that
    moment has no say in it. A node ending takes its descendants down with it, and each
    of them is INTERRUPTED, however the node ended.
    If multiple conditions are met, the following order is used:
        1. its own reset condition, or its parent has not started
        2. its own success condition
        3. its own fail condition
        4. its own interrupt condition, or its parent has ended
        5. its own pause condition, or its parent is paused
        6. its own start condition, while its parent is running
    How to use this class:
        1. initialized with the context of the executor that runs it
        2. add nodes.
        3. set the transition conditions of nodes
        4. compile the statechart.
        5. call tick() to update the observation state and life cycle state.
            tick() raises the exception of a CancelStatechart that started, once the
            tick is complete.
        6. call is_ended() to check if the statechart is done.
    """

    context: StatechartContext = field(kw_only=True, repr=False)
    """
    The context the nodes of this statechart are expanded, built and ticked in.
    """

    pass_limit: int = field(default=20, kw_only=True, repr=False)
    """
    The most passes that may change a statechart without nested nodes within one tick,
    so that even at the limit a statechart of a few hundred nodes settles within a 50 Hz
    tick.
    """

    passes_per_nesting_level: int = field(default=2, kw_only=True, repr=False)
    """
    The passes added to :attr:`pass_limit` for every nesting level: one to pass an
    outcome on to the parent, and one for a sibling of that parent to react to it.
    """

    rx_graph: rx.PyDiGraph[StatechartNode] = field(
        default_factory=lambda: rx.PyDAG(multigraph=True), init=False, repr=False
    )
    """
    The underlying graph of the statechart.
    """

    observation_state: ObservationState = field(init=False)
    """
    Combined representation of the observation state of the statechart, to enable an
    efficient tick().
    """

    life_cycle_state: LifeCycleState = field(init=False)
    """
    Combined representation of the life cycle state of the statechart, to enable an
    efficient tick().
    """

    last_observation_state: LastObservationState = field(init=False)
    """
    Combined representation of the observation every node took most recently, to enable
    an efficient tick().
    """

    state_history: StateHistory = field(default_factory=StateHistory, init=False)
    """
    The history of how the state of the statechart changed over time.
    """

    compiler: StatechartCompiler = field(init=False, repr=False, compare=False)
    """
    Compiles this statechart and keeps track of what changed since it compiled last.
    """

    _open_modifications: int = field(default=0, init=False, repr=False)
    """
    How many :class:`StatechartModification` blocks are open on this statechart.
    """

    _replaced_nodes: list[ReplacedNode] = field(
        default_factory=list, init=False, repr=False
    )
    """
    The nodes replaced in the open modifications that are still in this statechart, in
    order.
    """

    _nodes: list[StatechartNode] = field(default_factory=list, init=False, repr=False)
    """
    Cache of all nodes in index order, appended to in :meth:`add_node`.

    Reading this instead of rebuilding the list from `rx_graph` on every access is what
    keeps :meth:`tick` cheap.
    """

    _choosing_nodes: list[CompositeNodeChoosingItsChild] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Cache of all :class:`~cramph.composites.CompositeNodeChoosingItsChild` nodes,
    checked every tick in :meth:`_let_waiting_nodes_choose_their_child`.
    """

    _cancel_nodes: list[CancelStatechart] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Cache of all :class:`CancelStatechart` nodes, checked every tick in
    :meth:`_raise_if_cancelled`.
    """

    _end_nodes: list[EndStatechart] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Cache of all :class:`EndStatechart` nodes, checked every tick in :meth:`is_ended`.
    """

    def __post_init__(self) -> None:
        """
        Creates the (initially empty) life cycle, observation and last observation
        states for this statechart, and its compiler.
        """
        self.life_cycle_state = LifeCycleState(self)
        self.observation_state = ObservationState(self)
        self.last_observation_state = LastObservationState(self)
        self.compiler = StatechartCompiler(statechart=self)

    def create_structure_copy(self) -> Statechart:
        """
        Creates a copy of the statechart, where every node is an instance of the base
        class of its kind, see
        :meth:`~cramph.node.StatechartNode.create_structure_copy`.

        This is useful if only the structure of the statechart is needed, for example,
        for visualization.

        :return: The structural copy.
        """
        statechart_copy = Statechart(context=self.context)
        # copy nodes in order to make sure index is correct
        for node in self.nodes:
            statechart_copy._register_node(node.create_structure_copy())
        # link parent/child
        for node in self.get_nodes_by_type(CompositeNode):
            goal_copy: CompositeNode = statechart_copy.get_node_by_index(node.index)
            for child_node in node.nodes:
                child_node_copy = statechart_copy.get_node_by_index(child_node.index)
                child_node_copy.parent_node_index = node.index
                goal_copy.nodes.append(child_node_copy)
        # copy conditions and plot specs
        for node in self.nodes:
            node_copy = statechart_copy.get_node_by_index(node.index)
            node_copy.plot_specifications = deepcopy(node.plot_specifications)
            for transition_kind in TransitionKind:
                node_copy.set_condition(
                    transition_kind,
                    statechart_copy._copy_condition(
                        node.get_condition(transition_kind)
                    ),
                )
        return statechart_copy

    def _copy_condition(self, condition: sm.Scalar) -> sm.Scalar:
        """
        :param condition: A condition of the chart this chart is a structural copy of.
        :return: The same condition, reading the nodes of this chart with the same index.
        """
        variables: list[DerivedConditionVariable] = condition.free_variables()
        if not variables:
            return condition
        return sm.Scalar(condition).substitute(
            variables,
            [
                variable.for_node(
                    self.get_node_by_index(variable.statechart_node.index)
                )
                for variable in variables
            ],
        )

    @property
    def nodes(self) -> list[StatechartNode]:
        """
        :return: All nodes of the statechart.
        """
        return list(self._nodes)

    @property
    def top_level_nodes(self) -> list[StatechartNode]:
        """
        :return: All nodes that don't belong to a CompositeNode.
        """
        return [node for node in self.nodes if node.parent_node is None]

    @property
    def edges(self) -> list[TransitionCondition]:
        """
        The edges of the underlying graph.

        .. warning:: This may return duplicate edges if a transition uses multiple nodes.

        :return: The edges of the underlying graph.
        """
        return self.rx_graph.edges()

    @property
    def unique_edges(self) -> list[TransitionCondition]:
        """
        :return: The edges of the statechart, without duplicates.
        """
        return list(set(self.edges))

    def modify(self) -> StatechartModification:
        """
        :return: A block batching changes to this statechart. A compiled statechart
            compiles again once the outermost block ends, keeping the state and the
            history of the nodes it already held.
        """
        return StatechartModification(statechart=self)

    def request_rebuild(self, node: StatechartNode) -> None:
        """
        Builds `node` again on the next compile, because an expression it builds from
        changed, such as the children it reads.

        Called outside a :meth:`modify` block, the statechart compiles right away.

        :param node: A node of this statechart.
        """
        with self.modify():
            if node not in self.compiler.nodes_to_rebuild:
                self.compiler.nodes_to_rebuild.append(node)
            self.compiler.changed_since_compile = True

    def add_node(self, node: StatechartNode) -> None:
        """
        Adds a node to the statechart, and expands it right away if it is a
        :class:`CompositeNode`, which adds its children in turn.

        A compiled statechart compiles again once the node joined, or once the outermost
        :meth:`modify` block around the addition ends.

        :param node: The node to add.
        :raises PrerequisiteNotExpandedError: If `node` is a composite node that reads a
            composite node which has not joined this statechart yet.
        """
        with self.modify():
            self._register_node(node)
            if isinstance(node, CompositeNode):
                self._expand(node)

    def _put_back_replaced_nodes_from(self, first_put_back: int) -> None:
        """
        Puts back every node replaced from `first_put_back` on, the latest first.

        :param first_put_back: The position of the first record to undo in
            :attr:`_replaced_nodes`.
        """
        for replaced in reversed(self._replaced_nodes[first_put_back:]):
            replaced.put_back()
        del self._replaced_nodes[first_put_back:]

    def _drop_nodes_from(self, first_dropped_index: int) -> None:
        """
        Drops every node from `first_dropped_index` on, which no compile covered yet.

        :param first_dropped_index: The index of the first node to drop.
        """
        if first_dropped_index == len(self._nodes):
            return
        self.compiler.changed_since_compile = True
        kept_nodes = self._nodes[:first_dropped_index]
        for dropped_node in self._nodes[first_dropped_index:]:
            parent_node = dropped_node.parent_node
            if parent_node in kept_nodes and dropped_node in parent_node.nodes:
                parent_node.nodes.remove(dropped_node)
            dropped_node._statechart = None
            dropped_node.index = None
            dropped_node.parent_node_index = None
        self._renumber(kept_nodes)

    def _expand(self, node: CompositeNode) -> None:
        """
        Expands `node` in :attr:`context`, once every composite node it reads while
        expanding has joined this statechart.

        :param node: The composite node that just joined this statechart.
        :raises NodesMissingContextExtensionsError: If :attr:`context` lacks an
            extension `node` requires to expand.
        """
        for prerequisite in node.prerequisite_nodes:
            if isinstance(prerequisite, CompositeNode) and (
                prerequisite._statechart is not self
            ):
                raise PrerequisiteNotExpandedError(node=node, prerequisite=prerequisite)
        self.check_context_holds(
            [node],
            lambda expanding_node: expanding_node.context_extensions_required_to_expand,
        )
        node.adopt_handed_children()
        node.expand()

    def _register_node(self, node: StatechartNode) -> None:
        """
        Adds a node to the statechart without expanding it, and finalizes its
        initialization.

        :param node: The node to add.
        """
        node.statechart = self
        node.index = self.rx_graph.add_node(node)
        self.life_cycle_state.append_default_value()
        self.observation_state.append_default_value()
        self.last_observation_state.append_default_value()
        self._nodes.append(node)
        self.compiler.changed_since_compile = True
        if isinstance(node, CancelStatechart):
            self._cancel_nodes.append(node)
        if isinstance(node, EndStatechart):
            self._end_nodes.append(node)
        if isinstance(node, CompositeNodeChoosingItsChild):
            self._choosing_nodes.append(node)

    def remove_node(self, node: StatechartNode) -> None:
        """
        Removes a node and everything below it from the statechart, and from the node it
        is a child of.

        The nodes that stay keep their order and are numbered anew without gaps.

        :param node: The node to remove.
        :raises NotInStatechartError: If `node` does not belong to this statechart.
        :raises StatechartAlreadyCompiledError: If the latest compile covers `node`.
        :raises RemovedNodeStillReferencedError: If a node that stays refers to a
            removed node through a condition or as a prerequisite.
        """
        if node._statechart is not self:
            raise NotInStatechartError(name=node.name)
        if self.compiler.is_compiled and self.compiler.latest_compile_covers(node):
            raise StatechartAlreadyCompiledError()
        removed_nodes = [node, *node.descendants]
        kept_nodes = [kept for kept in self._nodes if kept not in removed_nodes]
        self._check_not_referenced(removed_nodes, kept_nodes)
        parent_node = node.parent_node
        if parent_node is not None and node in parent_node.nodes:
            parent_node.nodes.remove(node)
        self._renumber(kept_nodes)
        self.compiler.nodes_to_rebuild = [
            rebuilt
            for rebuilt in self.compiler.nodes_to_rebuild
            if rebuilt in kept_nodes
        ]
        for removed_node in removed_nodes:
            removed_node._statechart = None
            removed_node.index = None
            removed_node.parent_node_index = None

    def replace(
        self, node: StatechartNode, replacement: StatechartNode | Match
    ) -> None:
        """
        Puts `replacement` where `node` is: at its position among the children of the
        control structure node holding it, or on the top level.

        If `replacement` is a composite node with `node` among its children, `node`
        stays in the statechart as that child, at its position there, and forgets the
        transitions its former parent decided for it, so that `replacement` decides
        them. Otherwise `node` leaves the statechart together with everything below it.
        A :meth:`modify` block that raises puts back a `node` that stayed in the
        statechart.

        .. warning:: A :meth:`modify` block that raises after `node` left the statechart
            does not bring it back.

        :param node: A node of this statechart, on the top level or a child of a control
            structure node.
        :param replacement: The node to put in its place, which belongs to no
            statechart yet, or a statement of it.
        :raises NotInStatechartError: If `node` does not belong to this statechart.
        :raises NotHeldByControlStructureNodeError: If a node other than a control structure node
            holds `node`.
        :raises StatechartAlreadyCompiledError: If the latest compile covers `node`.
        """
        if node._statechart is not self:
            raise NotInStatechartError(name=node.name)
        holder = node.parent_node
        if holder is not None and not isinstance(holder, ControlStructureNode):
            raise NotHeldByControlStructureNodeError(node=node, parent=holder)
        if self.compiler.is_compiled and self.compiler.latest_compile_covers(node):
            raise StatechartAlreadyCompiledError()
        replacement = ControlStructureNode._as_node(replacement)
        if holder is not None:
            holder._check_is_not_a_child(replacement)
            holder._check_can_be_a_child(replacement)
        holds_node = isinstance(replacement, CompositeNode) and any(
            child is node for child in replacement.nodes
        )
        record = ReplacedNode.before_replacing(node, replacement)
        with self.modify():
            self._replaced_nodes.append(record)
            self._register_node(replacement)
            replacement.parent_node = holder
            if holder is not None:
                holder.list_in_place_of(node, replacement)
            if holds_node:
                replacement.take_over(node)
            if isinstance(replacement, CompositeNode):
                self._expand(replacement)
            if holder is not None:
                holder._wire_children()
                if holder.parent_node is not None:
                    holder.parent_node._succeed_at_its_goal(holder)
            if not holds_node:
                self.remove_node(node)
                self._replaced_nodes.remove(record)

    @staticmethod
    def _check_not_referenced(
        removed_nodes: list[StatechartNode], kept_nodes: list[StatechartNode]
    ) -> None:
        """
        :raises RemovedNodeStillReferencedError: If a node of `kept_nodes` refers to a
            node of `removed_nodes` through a condition or as a prerequisite.
        """
        for kept_node in kept_nodes:
            referenced_nodes = list(kept_node.prerequisite_nodes) + [
                dependency
                for condition in kept_node.effective_conditions
                for dependency in condition.node_dependencies
            ]
            for referenced_node in referenced_nodes:
                if referenced_node in removed_nodes:
                    raise RemovedNodeStillReferencedError(
                        removed_node=referenced_node, referencing_node=kept_node
                    )

    def _renumber(self, kept_nodes: list[StatechartNode]) -> None:
        """
        Keeps only `kept_nodes`, numbering them anew in their current order.

        The graph loses its edges, which the next compile adds again.

        :param kept_nodes: The nodes that stay, in index order.
        """
        kept_indices = [kept_node.index for kept_node in kept_nodes]
        new_index_of = {old: new for new, old in enumerate(kept_indices)}
        for state in (
            self.life_cycle_state,
            self.observation_state,
            self.last_observation_state,
        ):
            state.data = state.data[kept_indices]
        for kept_node in kept_nodes:
            if kept_node.parent_node_index is not None:
                kept_node.parent_node_index = new_index_of[kept_node.parent_node_index]
        self.rx_graph = rx.PyDAG(multigraph=True)
        for kept_node in kept_nodes:
            kept_node.index = self.rx_graph.add_node(kept_node)
        self._nodes = kept_nodes
        self._cancel_nodes = [
            cancel_node
            for cancel_node in self._cancel_nodes
            if cancel_node in kept_nodes
        ]
        self._end_nodes = [
            end_node for end_node in self._end_nodes if end_node in kept_nodes
        ]
        self._choosing_nodes = [
            choosing_node
            for choosing_node in self._choosing_nodes
            if choosing_node in kept_nodes
        ]

    def add_nodes(self, nodes: list[StatechartNode]) -> None:
        """
        Adds every node in `nodes` to the statechart in one :meth:`modify` block, see
        :meth:`add_node`.

        :param nodes: The nodes to add.
        """
        with self.modify():
            for node in nodes:
                self.add_node(node)

    def get_node_by_index(self, index: int) -> StatechartNode:
        """
        :param index: The :attr:`~StatechartNode.index` of the node to look up.
        :return: The node with the given index.
        """
        return self.rx_graph.get_node_data(index)

    def check_context_holds(
        self,
        nodes: list[StatechartNode],
        required_extensions_of: Callable[
            [StatechartNode], tuple[type[ContextExtension], ...]
        ],
    ) -> None:
        """
        :param nodes: The nodes to check.
        :param required_extensions_of: Gives the extension types a node requires.
        :raises NodesMissingContextExtensionsError: If :attr:`context` lacks an
            extension a node in `nodes` requires.
        """
        nodes_by_missing_extension = {}
        for node in nodes:
            for extension_type in required_extensions_of(node):
                if self.context.get_extension(extension_type) is None:
                    nodes_by_missing_extension.setdefault(extension_type, []).append(
                        node
                    )
        if nodes_by_missing_extension:
            raise NodesMissingContextExtensionsError(
                nodes_by_missing_extension=nodes_by_missing_extension
            )

    def tick(self) -> None:
        """
        Executes a single tick of the statechart in its :attr:`context`.

        Every node is brought to the state it reaches in this tick, see
        :class:`~cramph.compiled_tick.CompiledTick`, then every node waiting for a child
        chooses one, then the life cycle callbacks of every change run and the tick is
        recorded. A :class:`CancelStatechart` that started in this
        tick ends the statechart only after that, even if a
        :class:`~cramph.state_history.StateHistoryObserver` of the record failed.

        If the statechart changed since it compiled last, or the kinematic structure of
        the world did, it compiles again before this tick settles once every
        :class:`~cramph.compiler.RecompileCallback` is at rest, see
        :meth:`~cramph.compiler.StatechartCompiler.compile`; a change made by a
        callback of this tick is picked up by the next one. While a change of the
        statechart waits for that, no node changes its state, so nothing runs on the
        tick compiled before the change.
        """
        self.compiler.compile_if_changed()
        if not self.compiler.changed_since_compile:
            changes = self.compiler.compiled_tick.settle()
            # Before the callbacks, so a child is chosen against the world as it was
            # when the nodes starting beside it in this tick had not acted yet.
            self._let_waiting_nodes_choose_their_child()
            for change in changes:
                change.run_callback()
        try:
            self.state_history.append(
                next_item=StateHistoryItem(
                    tick_count=self.context.tick_count,
                    life_cycle_state=self.life_cycle_state,
                    observation_state=self.observation_state,
                )
            )
        finally:
            self._raise_if_cancelled()

    def _let_waiting_nodes_choose_their_child(self) -> None:
        """
        Lets every node waiting for a child choose one, in one :meth:`modify` block, so
        the statechart compiles at most once for all of them.

        Choosing blocks the tick the way compiling does, so it waits until every
        :class:`~cramph.compiler.RecompileCallback` is at rest, and a choice is made
        against a world that holds still.
        """
        waiting_nodes = [
            node for node in self._choosing_nodes if node.is_waiting_for_a_child
        ]
        if not waiting_nodes:
            return
        self.compiler.when_held_still(lambda: self._choose_children_of(waiting_nodes))

    def _choose_children_of(
        self, waiting_nodes: list[CompositeNodeChoosingItsChild]
    ) -> None:
        """
        Let every node of `waiting_nodes` choose its child, compiling once for all.
        """
        with self.modify():
            for node in waiting_nodes:
                node.choose_child()

    def get_nodes_by_type(
        self, node_type: type[GenericStatechartNode]
    ) -> list[GenericStatechartNode]:
        """
        :param node_type: The node type to filter for.
        :return: All nodes that are an instance of `node_type`.
        """
        return [node for node in self.nodes if isinstance(node, node_type)]

    def get_preceding_node_by_type(
        self, node: StatechartNode, node_type: type[GenericStatechartNode]
    ) -> GenericStatechartNode | None:
        """
        :param node: A node of this statechart.
        :param node_type: The node type to look for.
        :return: The closest node of `node_type` that runs before `node`, see
            :meth:`get_preceding_nodes_by_type`, or None if there is none.
        """
        return next(iter(self.get_preceding_nodes_by_type(node, node_type)), None)

    def get_preceding_nodes_by_type(
        self, node: StatechartNode, node_type: type[GenericStatechartNode]
    ) -> list[GenericStatechartNode]:
        """
        :param node: A node of this statechart.
        :param node_type: The node type to look for.
        :return: Every node of `node_type` that runs before `node`, closest first,
            searching the nodes left of `node` and of each of its ancestors.
        """
        return [
            earlier
            for ancestor_or_self in [node, *node.path]
            for sibling in reversed(ancestor_or_self.left_siblings)
            for earlier in reversed([sibling, *sibling.descendants])
            if isinstance(earlier, node_type)
        ]

    def is_ended(self) -> bool:
        """
        :return: True if the statechart is done, meaning at least one
            :class:`EndStatechart` observes True, False otherwise.
        """
        return any(
            self.observation_state[node] == ObservationStateValues.TRUE
            for node in self._end_nodes
        )

    def _raise_if_cancelled(self) -> None:
        """
        Raises the exception of the first :class:`CancelStatechart` node that started in
        the current tick.
        """
        for node in self._cancel_nodes:
            node.raise_pending_exception()

    def cleanup_nodes(self) -> None:
        """
        Calls :meth:`~StatechartNode.cleanup` on every node, in :attr:`context`.
        """
        for node in self.nodes:
            node.cleanup()

    def draw(self, file_name: str) -> None:
        """
        Uses graphviz to draw the statechart and safe it at `file_name`.

        :param file_name: Where to save the resulting file.
        """
        StatechartGraphviz(self).to_dot_graph_pdf(file_name=file_name)

    @property
    def layers(self) -> list[list[StatechartNode]]:
        """
        :return: The nodes layer by layer, from the top-level nodes down to the most
            deeply nested children, each layer in order.
        """
        layers = []
        layer = self.top_level_nodes
        while layer:
            layers.append(layer)
            layer = [child for node in layer for child in node.children]
        return layers

    def visualize(
        self,
        backend: GraphVisualizerBackend = GraphVisualizerBackend.CYTOSCAPE,
        layout: GraphLayout = GraphLayout.LAYERED,
    ) -> GraphVisualizerBase:
        """
        Open an interactive visualization of the nodes and their children, see
        :class:`~cramph.plotters.interactive_graph.StatechartGraphVisualizer`.

        :param backend: The rendering technology to use.
        :param layout: The algorithm used to place the nodes.
        :return: The running visualizer.
        """
        visualizer = StatechartGraphVisualizer(self).create_visualizer(
            backend=backend, layout=layout
        )
        visualizer.run()
        return visualizer

    def plot_gantt_chart(
        self,
        path: str = "./ganttchart.pdf",
        context: StatechartContext = None,
        second_length_in_cm: float = 2.0,
    ) -> None:
        """
        Renders a Gantt chart of :attr:`state_evolution_history` and saves it at `path`.

        :param path: Where to save the resulting PDF.
        :param context: If given, the x-axis is scaled to seconds using its
            :attr:`~StatechartContext.tick_duration` instead of ticks.
        :param second_length_in_cm: Width in cm of one second on the x-axis.
        """
        HistoryGanttChartPlotter(
            self, second_width_in_cm=second_length_in_cm, context=context
        ).plot_gantt_chart(path)

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        World entities are written as references, because whoever reads a statechart
        resolves them against its own world, which has the same entities.

        :return: The JSON representation of this statechart, including all nodes and the
            transition conditions of every node the document holds.
        """
        return {
            **super().to_json(**kwargs),
            **self.nodes_from_to_json(first_node_index=0, **kwargs),
        }

    def nodes_from_to_json(
        self, first_node_index: int, **kwargs: Any
    ) -> dict[str, Any]:
        """
        World entities are written as references, see :meth:`to_json`.

        :param first_node_index: The index of the first node to write.
        :return: The JSON representation of the nodes from `first_node_index` on and of
            their transition conditions, which :meth:`add_nodes_from_json` adds to a
            statechart holding the nodes before them.
        """
        kwargs = {**kwargs, **WorldEntityReferenceWriter().create_kwargs()}
        written_nodes = self.nodes[first_node_index:]
        return {
            StatechartJSONKey.NODES: [
                to_json(node, **kwargs) for node in written_nodes
            ],
            StatechartJSONKey.CONDITIONS: [
                condition.to_json(**kwargs)
                for node in written_nodes
                for condition in node.conditions
            ],
            StatechartJSONKey.CHILDREN: {
                node.parent_node.index: [
                    child.index for child in node.parent_node.nodes
                ]
                for node in written_nodes
                if node.parent_node is not None
            },
        }

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], *, context: StatechartContext, **kwargs: Any
    ) -> Self:
        """
        Reconstructs a statechart from its JSON representation, as produced by
        :meth:`to_json`: first all nodes, then the transition conditions of every node
        the document holds, then the children of every goal in the order it was written
        with. The goals are not expanded again, because they expanded before they were
        serialized.

        :param data: The JSON dict.
        :param context: The context the deserialized statechart is built and run in.
        :param kwargs: Forwarded to :func:`~krrood.adapters.json_serializer.from_json`
            for every node.
        :return: The deserialized statechart.
        """
        statechart = cls(context=context)
        statechart._register_nodes_from_json(data, **kwargs)
        return statechart

    def add_nodes_from_json(self, data: dict[str, Any], **kwargs: Any) -> None:
        """
        Adds the nodes written by :meth:`nodes_from_to_json` of a statechart holding the
        same nodes as this one before them, in one :meth:`modify` block.

        The nodes are not expanded again, because they expanded before they were
        written, and their conditions may read the nodes this statechart already holds.

        :param data: The JSON dict.
        :param kwargs: Forwarded to :func:`~krrood.adapters.json_serializer.from_json`
            for every node.
        """
        with self.modify():
            self._register_nodes_from_json(
                data,
                **{
                    **DeserializedNodeTracker.from_statechart(self).create_kwargs(),
                    **kwargs,
                },
            )

    def _register_nodes_from_json(self, data: dict[str, Any], **kwargs: Any) -> None:
        """
        Registers the written nodes, then their transition conditions, then hands every
        goal holding one of them its children in the order it was written with.
        """
        DeserializedNodeTracker.from_kwargs(kwargs)
        for json_data in data[StatechartJSONKey.NODES]:
            node = from_json(json_data, **kwargs)
            self._register_node(node)
        for json_data in data[StatechartJSONKey.CONDITIONS]:
            transition = TransitionCondition.from_json(json_data, **kwargs)
            transition.owner._set_transition(transition)
        for parent_index, child_indices in data[StatechartJSONKey.CHILDREN].items():
            # JSON writes the integer keys as strings
            self.get_node_by_index(int(parent_index)).nodes = [
                self.get_node_by_index(child_index) for child_index in child_indices
            ]

    def sanity_check(self) -> None:
        """
        Executes a sanity check on the statechart to ensure that it is valid.
        """
        if len(self.nodes) == 0:
            raise EmptyStatechartError()

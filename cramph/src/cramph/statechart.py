"""
The statechart: a graph of nodes that is compiled and ticked to run them.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum

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
from cramph.compiled_tick import LifeCycleChange
from cramph.data_types import (
    LifeCycleValues,
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


@dataclass(kw_only=True, eq=False)
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
        cls, *, node: StatechartNode, replacement: StatechartNode
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
                transition_kind: node.get_condition(transition_kind=transition_kind)
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
            self.node.set_condition(
                transition_kind=transition_kind, expression=condition
            )
        if self.former_parent is not None:
            self.former_parent.list_in_place_of(
                child=self.replacement, substitute=self.node
            )


@dataclass(kw_only=True)
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
            statechart._put_back_replaced_nodes_from(
                first_put_back=self._replaced_count_at_start
            )
            statechart._drop_nodes_from(first_dropped_index=self._node_count_at_start)
        if statechart._open_modifications == 0:
            statechart._replaced_nodes.clear()
        compiler = statechart.compiler
        if statechart._open_modifications > 0 or not compiler.is_compiled:
            return
        if not compiler.changed_since_compile:
            return
        compiler.compile()


# %% serialization keys


class StatechartJSONKey(StrEnum):
    """
    Keys a serialized statechart carries.
    """

    NODES = "nodes"
    """
    The nodes of the statechart, in the order of their index.
    """

    CONDITIONS = "conditions"
    """
    Every transition condition of every node the document holds, including the children
    of goals that join the statechart only when it is compiled.
    """

    CHILDREN = "children"
    """
    The indices of the children of every node that is the parent of a node the document
    holds, in the order the parent holds them, by the index of the parent.
    """

    CHOSEN_CHILDREN = "chosen_children"
    """
    The indices of the children every node choosing its child that the document holds
    has chosen, in the order it chose them, by the index of the node.
    """


# %% the statechart


@dataclass(kw_only=True)
class Statechart(SubclassJSONSerializer):
    """
    A directed graph of nodes whose life cycle and observation states :meth:`tick`
    brings up to date, until an :class:`EndStatechart` observes True or a
    :class:`CancelStatechart` starts.

    .. seealso:: ``cramph/doc/statecharts.md`` for the states, the transitions between
        them and their precedence, what one tick does and how a statechart is built,
        run and ended.
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
    :meth:`_end_if_cancelled`.
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
        self.life_cycle_state = LifeCycleState(statechart=self)
        self.observation_state = ObservationState(statechart=self)
        self.last_observation_state = LastObservationState(statechart=self)
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
            statechart_copy._register_node(node=node.create_structure_copy())
        # link parent/child
        for node in self.get_nodes_by_type(node_type=CompositeNode):
            goal_copy: CompositeNode = statechart_copy.get_node_by_index(
                index=node.index
            )
            for child_node in node.nodes:
                child_node_copy = statechart_copy.get_node_by_index(
                    index=child_node.index
                )
                child_node_copy.parent_node_index = node.index
                goal_copy.nodes.append(child_node_copy)
        # copy conditions and plot specs
        for node in self.nodes:
            node_copy = statechart_copy.get_node_by_index(index=node.index)
            node_copy.plot_specifications = deepcopy(node.plot_specifications)
            for transition_kind in TransitionKind:
                node_copy.set_condition(
                    transition_kind=transition_kind,
                    expression=DerivedConditionVariable.reading_counterparts(
                        condition=node.get_condition(transition_kind=transition_kind),
                        counterpart_of=lambda read: statechart_copy.get_node_by_index(
                            index=read.index
                        ),
                    ),
                )
        return statechart_copy

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

    def request_rebuild(self, *, node: StatechartNode) -> None:
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

    def add_node(self, *, node: StatechartNode) -> None:
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
            self._register_node(node=node)
            if isinstance(node, CompositeNode):
                self._expand(node=node)

    def _put_back_replaced_nodes_from(self, *, first_put_back: int) -> None:
        """
        Puts back every node replaced from `first_put_back` on, the latest first.

        :param first_put_back: The position of the first record to undo in
            :attr:`_replaced_nodes`.
        """
        for replaced in reversed(self._replaced_nodes[first_put_back:]):
            replaced.put_back()
        del self._replaced_nodes[first_put_back:]

    def _drop_nodes_from(self, *, first_dropped_index: int) -> None:
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
        self._renumber(kept_nodes=kept_nodes)

    def _expand(self, *, node: CompositeNode) -> None:
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
            nodes=[node],
            required_extensions_of=lambda expanding_node: (
                expanding_node.context_extensions_required_to_expand
            ),
        )
        node.adopt_handed_children()
        node.expand()

    def _register_node(self, *, node: StatechartNode) -> None:
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

    def remove_node(self, *, node: StatechartNode) -> None:
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
        if self.compiler.is_compiled and self.compiler.latest_compile_covers(node=node):
            raise StatechartAlreadyCompiledError()
        removed_nodes = [node, *node.descendants]
        kept_nodes = [kept for kept in self._nodes if kept not in removed_nodes]
        self._check_not_referenced(removed_nodes=removed_nodes, kept_nodes=kept_nodes)
        parent_node = node.parent_node
        if parent_node is not None and node in parent_node.nodes:
            parent_node.nodes.remove(node)
        self._renumber(kept_nodes=kept_nodes)
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
        self, *, node: StatechartNode, replacement: StatechartNode | Match
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
        if self.compiler.is_compiled and self.compiler.latest_compile_covers(node=node):
            raise StatechartAlreadyCompiledError()
        replacement = ControlStructureNode._as_node(node=replacement)
        if holder is not None:
            holder._check_is_not_a_child(node=replacement)
            holder._check_can_be_a_child(node=replacement)
        holds_node = isinstance(replacement, CompositeNode) and any(
            child is node for child in replacement.nodes
        )
        record = ReplacedNode.before_replacing(node=node, replacement=replacement)
        with self.modify():
            self._replaced_nodes.append(record)
            self._register_node(node=replacement)
            replacement.parent_node = holder
            if holder is not None:
                holder.list_in_place_of(child=node, substitute=replacement)
            if holds_node:
                replacement.take_over(node=node)
            if isinstance(replacement, CompositeNode):
                self._expand(node=replacement)
            if holder is not None:
                holder._wire_children()
                if holder.parent_node is not None:
                    holder.parent_node._succeed_at_its_goal(child=holder)
            if not holds_node:
                self.remove_node(node=node)
                self._replaced_nodes.remove(record)

    @staticmethod
    def _check_not_referenced(
        *, removed_nodes: list[StatechartNode], kept_nodes: list[StatechartNode]
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

    def _renumber(self, *, kept_nodes: list[StatechartNode]) -> None:
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

    def add_nodes(self, *, nodes: list[StatechartNode]) -> None:
        """
        Adds every node in `nodes` to the statechart in one :meth:`modify` block, see
        :meth:`add_node`.

        :param nodes: The nodes to add.
        """
        with self.modify():
            for node in nodes:
                self.add_node(node=node)

    def get_node_by_index(self, *, index: int) -> StatechartNode:
        """
        :param index: The :attr:`~StatechartNode.index` of the node to look up.
        :return: The node with the given index.
        """
        return self.rx_graph.get_node_data(index)

    def check_context_holds(
        self,
        *,
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
                if self.context.get_extension(extension_type=extension_type) is None:
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
        :class:`~cramph.state_history.StateHistoryObserver` of the record failed, see
        :meth:`_end_if_cancelled`.

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
            self._end_if_cancelled()

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
        self.compiler.when_held_still(
            step=lambda: self._choose_children_of(waiting_nodes=waiting_nodes)
        )

    def _choose_children_of(
        self, *, waiting_nodes: list[CompositeNodeChoosingItsChild]
    ) -> None:
        """
        Let every node of `waiting_nodes` choose its child, compiling once for all.

        A choice may hand another node its child, or let the chooser of another node
        answer, so the nodes still waiting are asked again until a round chooses
        nothing.
        """
        with self.modify():
            while waiting_nodes:
                for node in waiting_nodes:
                    if node.is_waiting_for_a_child:
                        node.choose_child()
                still_waiting = [
                    node for node in waiting_nodes if node.is_waiting_for_a_child
                ]
                if len(still_waiting) == len(waiting_nodes):
                    return
                waiting_nodes = still_waiting

    def get_nodes_by_type(
        self, *, node_type: type[GenericStatechartNode]
    ) -> list[GenericStatechartNode]:
        """
        :param node_type: The node type to filter for.
        :return: All nodes that are an instance of `node_type`.
        """
        return [node for node in self.nodes if isinstance(node, node_type)]

    def get_preceding_node_by_type(
        self, *, node: StatechartNode, node_type: type[GenericStatechartNode]
    ) -> GenericStatechartNode | None:
        """
        :param node: A node of this statechart.
        :param node_type: The node type to look for.
        :return: The closest node of `node_type` that runs before `node`, see
            :meth:`get_preceding_nodes_by_type`, or None if there is none.
        """
        return next(
            iter(self.get_preceding_nodes_by_type(node=node, node_type=node_type)), None
        )

    def get_preceding_nodes_by_type(
        self, *, node: StatechartNode, node_type: type[GenericStatechartNode]
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

    def _end_if_cancelled(self) -> None:
        """
        Ends the statechart if a :class:`CancelStatechart` node started in the current
        tick: every other node still running or paused is interrupted, which is recorded
        as the last snapshot of :attr:`state_history`, naming the first such cancel
        node and its exception, and then that exception is raised.
        """
        cancelling_node = next(
            (node for node in self._cancel_nodes if node.has_pending_exception), None
        )
        if cancelling_node is None:
            return
        for change in self._interrupt_unfinished_nodes(cancelling_node=cancelling_node):
            change.run_callback()
        try:
            self.state_history.append(
                next_item=StateHistoryItem(
                    tick_count=self.context.tick_count,
                    life_cycle_state=self.life_cycle_state,
                    observation_state=self.observation_state,
                    cancelled_by=cancelling_node,
                    cancellation_error=cancelling_node.pending_exception,
                )
            )
        finally:
            cancelling_node.raise_pending_exception()

    def _interrupt_unfinished_nodes(
        self, *, cancelling_node: CancelStatechart
    ) -> list[LifeCycleChange]:
        """
        :param cancelling_node: The node cancelling the statechart, which is left as it
            is.
        :return: The change of every node but `cancelling_node` that was running or
            paused and is interrupted now.
        """
        changes = [
            LifeCycleChange(
                node=node,
                previous_state=node.life_cycle_state,
                new_state=LifeCycleValues.INTERRUPTED,
            )
            for node in self.nodes
            if node is not cancelling_node
            and node.life_cycle_state
            in (LifeCycleValues.RUNNING, LifeCycleValues.PAUSED)
        ]
        for change in changes:
            self.life_cycle_state[change.node] = change.new_state
        return changes

    def cleanup_nodes(self) -> None:
        """
        Calls :meth:`~StatechartNode.cleanup` on every node, in :attr:`context`.
        """
        for node in self.nodes:
            node.cleanup()

    def draw(self, *, file_name: str) -> None:
        """
        Uses graphviz to draw the statechart and safe it at `file_name`.

        :param file_name: Where to save the resulting file.
        """
        StatechartGraphviz(statechart=self).to_dot_graph_pdf(file_name=file_name)

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
        *,
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
        visualizer = StatechartGraphVisualizer(statechart=self).create_visualizer(
            backend=backend, layout=layout
        )
        visualizer.run()
        return visualizer

    def plot_gantt_chart(
        self,
        *,
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
            statechart=self, second_width_in_cm=second_length_in_cm, context=context
        ).plot_gantt_chart(file_name=path)

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
        self,
        *,
        first_node_index: int,
        stop_node_index: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        World entities are written as references, see :meth:`to_json`.

        :param first_node_index: The index of the first node to write.
        :param stop_node_index: The index of the first node not to write, None to write
            every node from `first_node_index` on.
        :return: The JSON representation of the nodes from `first_node_index` up to
            `stop_node_index` and of their transition conditions, which
            :meth:`add_nodes_from_json` adds to a statechart holding the nodes before
            them.
        """
        kwargs = {**kwargs, **WorldEntityReferenceWriter().create_kwargs()}
        written_nodes = self.nodes[first_node_index:stop_node_index]
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
            StatechartJSONKey.CHOSEN_CHILDREN: {
                node.index: [child.index for child in node.chosen_children]
                for node in written_nodes
                if isinstance(node, CompositeNodeChoosingItsChild)
                and node.chosen_children
            },
        }

    @classmethod
    def _from_json(
        cls, *, data: dict[str, Any], context: StatechartContext, **kwargs: Any
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
        statechart._register_nodes_from_json(data=data, **kwargs)
        return statechart

    def add_nodes_from_json(self, *, data: dict[str, Any], **kwargs: Any) -> None:
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
                data=data,
                **{
                    **DeserializedNodeTracker.from_statechart(
                        statechart=self
                    ).create_kwargs(),
                    **kwargs,
                },
            )

    def _register_nodes_from_json(self, *, data: dict[str, Any], **kwargs: Any) -> None:
        """
        Registers the written nodes, then their transition conditions, then hands every
        goal holding one of them its children in the order it was written with.
        """
        DeserializedNodeTracker.from_kwargs(from_json_kwargs=kwargs)
        for json_data in data[StatechartJSONKey.NODES]:
            node = from_json(json_data, **kwargs)
            self._register_node(node=node)
        for json_data in data[StatechartJSONKey.CONDITIONS]:
            transition = TransitionCondition.from_json(data=json_data, **kwargs)
            transition.owner._set_transition(transition=transition)
        for parent_index, child_indices in data[StatechartJSONKey.CHILDREN].items():
            # JSON writes the integer keys as strings
            self.get_node_by_index(index=int(parent_index)).nodes = [
                self.get_node_by_index(index=child_index)
                for child_index in child_indices
            ]
        for node_index, child_indices in data[
            StatechartJSONKey.CHOSEN_CHILDREN
        ].items():
            self.get_node_by_index(index=int(node_index)).restore_chosen_children(
                children=[
                    self.get_node_by_index(index=child_index)
                    for child_index in child_indices
                ]
            )

    def sanity_check(self) -> None:
        """
        Executes a sanity check on the statechart to ensure that it is valid.
        """
        if len(self.nodes) == 0:
            raise EmptyStatechartError()

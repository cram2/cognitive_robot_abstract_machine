from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Callable

from cramph.compiled_tick import CompiledTick
from cramph.exceptions import ConditionScopeError, CyclicNodeDependencyError
from cramph.node import CompositeNode, StatechartNode, TransitionCondition
from cramph.state_history import StateHistoryItem
from semantic_digital_twin.world import ModelRevision

if TYPE_CHECKING:
    from cramph.statechart import Statechart


# %% being told that a statechart compiles again


@dataclass
class RecompileCallback(ABC):
    """
    Something told whenever a statechart that already compiled compiles again, because
    nodes joined it or the kinematic structure of its world changed.
    """

    @abstractmethod
    def before_recompile(self) -> bool:
        """
        Start holding still, because the statechart is about to compile again or a node
        is about to choose its child, either of which blocks its tick until it is done.

        The statechart keeps ticking, asking again every tick, until every callback
        answers that it is at rest; only then does it block its tick.

        :return: Whether what this callback drives is at rest.
        """

    @abstractmethod
    def after_recompile(self) -> None:
        """
        React to the statechart having compiled again.
        """


# %% compiling a statechart


@dataclass(eq=False)
class StatechartCompiler:
    """
    Compiles a statechart: completes and builds its nodes, derives the edges of its
    graph from their conditions and compiles its tick, and keeps track of what changed
    since the latest compile.
    """

    statechart: Statechart = field(repr=False)
    """
    The statechart this compiler compiles.
    """

    compiled_tick: CompiledTick | None = field(default=None, init=False, repr=False)
    """
    Updates every node of :attr:`statechart` once per tick, created by :meth:`compile`.
    """

    changed_since_compile: bool = field(default=False, init=False)
    """
    Whether nodes joined or were dropped since the latest compile.
    """

    nodes_to_rebuild: list[StatechartNode] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Nodes the latest compile covered whose expressions changed since, so the next
    compile builds them again.
    """

    _recompile_callbacks: list[RecompileCallback] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Told whenever :attr:`statechart` compiles again after it had compiled.
    """

    _held_still: bool = field(default=False, init=False, repr=False)
    """
    Whether every :class:`RecompileCallback` answered that it is at rest, for the step
    blocking the tick that runs right now.
    """

    _compiled_node_count: int = field(default=0, init=False, repr=False)
    """
    How many of the nodes, counted in index order, the latest compile covered.
    """

    _compiled_world_revision: ModelRevision | None = field(
        default=None, init=False, repr=False
    )
    """
    The kinematic structure of the world the nodes were last built against.
    """

    @property
    def is_compiled(self) -> bool:
        """
        :return: Whether :meth:`compile` ran.
        """
        return self.compiled_tick is not None

    def add_recompile_callback(self, callback: RecompileCallback) -> None:
        """
        :param callback: Told whenever :attr:`statechart` compiles again after it had
            compiled. A callback already registered is not added twice.
        """
        if any(registered is callback for registered in self._recompile_callbacks):
            return
        self._recompile_callbacks.append(callback)

    def latest_compile_covers(self, node: StatechartNode) -> bool:
        """
        :param node: A node of :attr:`statechart`.
        :return: Whether the latest compile covered `node`, which fixed its wiring.
        """
        return node.index < self._compiled_node_count

    def compile(self) -> None:
        """
        Compiles all components of :attr:`statechart` in its context.

        This method must be called before the statechart ticks.

        Compiling again, once nodes joined, keeps the state and the history of the
        nodes already there: only the joined nodes are completed and built, and the
        tick is compiled over every node. Every :class:`RecompileCallback` is told.
        Compiling again waits until every :class:`RecompileCallback` is at rest; until
        then the statechart goes on ticking as it compiled last, see
        :meth:`~cramph.statechart.Statechart.tick`.
        """
        is_recompile = self.is_compiled
        if is_recompile and not self._hold_still():
            return
        statechart = self.statechart
        statechart.sanity_check()
        self._compile_nodes(statechart.nodes[self._compiled_node_count :])
        if is_recompile:
            for callback in list(self._recompile_callbacks):
                callback.after_recompile()
            return
        statechart.state_history.append(
            next_item=StateHistoryItem(
                tick_count=0,
                life_cycle_state=statechart.life_cycle_state,
                observation_state=statechart.observation_state,
            )
        )

    def compile_if_changed(self) -> None:
        """
        Compiles again once every :class:`RecompileCallback` is at rest, if
        :attr:`statechart` changed since it compiled last, which a compile left waiting
        for them does, or if the kinematic structure of the world changed since the
        nodes were built, because expressions reading the structure, such as forward
        kinematics, describe the old one; the latter builds every node again.

        The state, the history and what each node registered in
        :meth:`~cramph.node.StatechartNode.set_up` are kept, so the statechart goes on
        from where it was.
        """
        if self.changed_since_compile or self._world_structure_changed():
            self.when_held_still(self.compile)

    def when_held_still(self, step: Callable[[], None]) -> None:
        """
        Run `step`, which blocks the tick, once every :class:`RecompileCallback` is at
        rest; until then the statechart goes on ticking and asks again next tick.

        :param step: What blocks the tick.
        """
        if not self._hold_still():
            return
        self._held_still = True
        try:
            step()
        finally:
            self._held_still = False

    def add_transitions(self) -> None:
        """
        Rebuilds the edges of the graph of :attr:`statechart` from the current
        transition conditions of every node.

        .. note:: Must run after goal expansion, because parent relationships are only
            known then.

        :raises ConditionScopeError: If a condition references a node from a different
            scope level, in which case the edges are left as they were.
        """
        conditions_by_owner = [
            (node, condition)
            for node in self.statechart.nodes
            for condition in node.effective_conditions
        ]
        for owner, condition in conditions_by_owner:
            self._validate_condition_scope(owner, condition)
        self.statechart.rx_graph.clear_edges()
        for owner, condition in conditions_by_owner:
            self._create_edge_for_condition(owner, condition)

    def _compile_nodes(self, nodes: list[StatechartNode]) -> None:
        """
        Completes `nodes`, builds them, or every node if the kinematic structure of the
        world changed since the nodes were built, then compiles the tick over every
        node.

        :param nodes: The nodes no compile covered yet.
        """
        goals = [node for node in nodes if isinstance(node, CompositeNode)]
        for goal in goals:
            goal.check_children()
        self.statechart.check_context_holds(
            nodes, lambda compiled_node: compiled_node.required_context_extensions
        )
        if self._world_structure_changed():
            nodes_to_build = self.statechart.nodes
        else:
            nodes_to_build = self.nodes_to_rebuild + [
                node for node in nodes if node not in self.nodes_to_rebuild
            ]
        self._build_nodes(nodes=nodes_to_build)
        self._compile_tick()
        self._compiled_node_count = len(self.statechart.nodes)
        self.changed_since_compile = False
        self.nodes_to_rebuild = []

    def _compile_tick(self) -> None:
        """
        Compiles the tick over every node as currently built, against the current
        kinematic structure of the world.
        """
        self.add_transitions()
        self.compiled_tick = CompiledTick(statechart=self.statechart)
        self.compiled_tick.compile()
        self._compiled_world_revision = self._world_revision()

    def _world_revision(self) -> ModelRevision:
        """
        :return: The current kinematic structure of the world of the context of
            :attr:`statechart`.
        """
        return self.statechart.context.world.get_world_model_manager().revision

    def _world_structure_changed(self) -> bool:
        """
        :return: Whether the kinematic structure of the world changed since the nodes
            were last built.
        """
        return self._world_revision() != self._compiled_world_revision

    def _hold_still(self) -> bool:
        """
        Ask every :class:`RecompileCallback` to hold still, unless they already answered
        that they are at rest for the step running now.

        :return: Whether all of them are at rest.
        """
        if self._held_still:
            return True
        answers = [
            callback.before_recompile() for callback in self._recompile_callbacks
        ]
        return all(answers)

    def _validate_condition_scope(
        self, owner: StatechartNode, condition: TransitionCondition
    ) -> None:
        """
        Checks that `condition` only depends on `owner` itself, siblings of `owner` or
        direct children of `owner`.

        :param owner: The node that owns `condition`.
        :param condition: The condition to validate.
        :raises ConditionScopeError: If `condition` depends on a node from a different
            scope level.
        """
        for variable in condition.variables:
            dependency = variable.statechart_node
            if dependency is owner:
                continue
            if dependency.parent_node is owner.parent_node:
                continue
            if dependency.parent_node is owner:
                continue
            raise ConditionScopeError(
                condition=condition,
                new_expression=condition.expression,
                dependency=dependency,
            )

    def _create_edge_for_condition(
        self, owner: StatechartNode, condition: TransitionCondition
    ) -> None:
        """
        Adds an edge from `owner` to every node `condition` depends on.

        :param owner: The node the edges originate from.
        :param condition: The condition whose node dependencies become edge targets.
        """
        for parent_node in condition.node_dependencies:
            self.statechart.rx_graph.add_edge(owner.index, parent_node.index, condition)

    def _build_nodes(self, nodes: list[StatechartNode]) -> None:
        """
        Builds `nodes` and applies their resulting artifacts, leaving every other node
        of :attr:`statechart` as it was built already.

        :param nodes: The nodes to build.
        """
        built_node_indices = {node.index for node in self.statechart.nodes} - {
            node.index for node in nodes
        }
        for node in nodes:
            self._build_and_apply_artifacts(node, built_node_indices, [])

    def _build_and_apply_artifacts(
        self,
        node: StatechartNode,
        built_node_indices: set[int],
        dependency_chain: list[StatechartNode],
    ) -> None:
        """
        Builds `node`, recursively building the nodes it depends on and, if it is a
        :class:`~cramph.node.CompositeNode`, its children first, then stores the
        resulting :class:`~cramph.node.NodeArtifacts` on the node.

        Already-built nodes (tracked via `built_node_indices`) are skipped.

        :param node: The node to build.
        :param built_node_indices: The indices of nodes already built, updated in place.
        :param dependency_chain: The nodes currently being built, used to detect cycles.
        """
        if node.index in built_node_indices:
            return
        self._check_no_dependency_cycle(node, dependency_chain)
        chain = dependency_chain + [node]
        for dependency in node.prerequisite_nodes:
            self._build_and_apply_artifacts(dependency, built_node_indices, chain)
        if isinstance(node, CompositeNode):
            for child_node in node.nodes:
                self._build_and_apply_artifacts(child_node, built_node_indices, chain)
        built_node_indices.add(node.index)
        node.apply_artifacts(node.build())

    @staticmethod
    def _check_no_dependency_cycle(
        node: StatechartNode,
        dependency_chain: list[StatechartNode],
    ) -> None:
        """
        Raises if `node` already appears in the chain of nodes currently being expanded
        or built, which would otherwise recurse forever.

        :param node: The node to check.
        :param dependency_chain: The nodes currently being expanded or built.
        """
        if node not in dependency_chain:
            return
        cycle_start = dependency_chain.index(node)
        raise CyclicNodeDependencyError(
            node=node, cycle=dependency_chain[cycle_start:] + [node]
        )

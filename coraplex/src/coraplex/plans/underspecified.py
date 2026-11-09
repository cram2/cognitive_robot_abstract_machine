from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field

from typing_extensions import (
    TYPE_CHECKING,
    Any,
)

from coraplex.datastructures.enums import ActionTrialVisualization
from coraplex.plans.context_extensions import StatementGrounding
from coraplex.plans.plan_transformation import PlanRewriting, PlanTransformation
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.plan_transformations import WriteTheOdometryInsteadOfDriving
from coraplex.robot_plans.robot_specific_plan_transformations import (
    KeepStretchInFrontOfTheContainerItCloses,
)
from cramph.visualization import RvizVisualization
from cramph.context import StatechartContext
from cramph.composites import (
    ChildChooser,
    CompositeNodeChoosingItsChild,
    ControlStructureNodeRunningItsChildrenInTurn,
    GenericChoosingNode,
    Parallel,
    Sequence,
    StatementNode,
    StatementsGroundedTogether,
)
from cramph.executor import ExecutorExtension
from cramph.exceptions import ExecutionFailure
from cramph.data_types import TransitionKind
from cramph.node import DerivedConditionVariable, NodeRebuild, StatechartNode
from cramph.statechart import Statechart
from giskardpy.motion_statechart.monitors.progress_monitors import Stalled

if TYPE_CHECKING:
    from coraplex.plans.executors import PlanExecutor
    from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)

# %% preparing a chosen child


@dataclass
class PlanChildChooser(ChildChooser[GenericChoosingNode], ABC):
    """
    Chooses children that are part of a plan: each is rewritten by the plan
    transformations of its statechart, and given up on once it stops approaching its
    goal, so the node that chose it can choose the next one.
    """

    def prepare_chosen_child(
        self,
        node: GenericChoosingNode,
        child: StatechartNode,
    ) -> None:
        """
        Rewrites `child` with the plan transformations of `context`, before the monitor
        watching it for a stall collects what it watches, see
        :meth:`~cramph.composites.ChildChooser.prepare_chosen_child`.
        """
        rewriting = node.context.get_extension(PlanRewriting)
        if rewriting is not None:
            rewriting.rewrite(child)

    def failure_monitors_for(
        self,
        node: GenericChoosingNode,
        child: StatechartNode,
    ) -> list[StatechartNode]:
        """
        Differs from :meth:`~cramph.composites.ChildChooser.failure_monitors_for` in
        giving up on `child` once nothing below it approaches its goal.

        :return: A monitor firing once nothing below `child` approaches its goal.
        """
        return [Stalled(monitored_node=child)]


# %% handing a statement a candidate grounded beforehand


@dataclass
class CandidateHandOver(PlanChildChooser[StatementNode]):
    """
    Hands a statement the candidate it was grounded to beforehand, prepared and watched
    for a stall like a candidate the statement grounds itself.

    The statement counts as grounded then, so the plan transformations rewriting
    statements leave it alone.
    """

    candidate: Action
    """
    The candidate the statement runs.
    """

    def choose_child(self, node: StatementNode) -> StatechartNode | None:
        """
        :return: :attr:`candidate`, in a sequence of its own for the nodes a plan
            transformation puts beside it.
        """
        return Sequence(name=f"{self.candidate.name}/steps", nodes=[self.candidate])

    def prepare_chosen_child(self, node: StatementNode, child: StatechartNode) -> None:
        """
        Differs from :meth:`PlanChildChooser.prepare_chosen_child` in also keeping the
        plan transformations from rewriting `node`.
        """
        super().prepare_chosen_child(node, child)
        rewriting = node.context.get_extension(PlanRewriting)
        if rewriting is not None:
            rewriting.treat_as_offered(node)


# %% rebuilding part of a plan onto a world


@dataclass
class SubtreeRebuild(NodeRebuild):
    """
    Rebuilds a node of a plan and everything below it onto a world to be tried, putting
    candidates in for its statements.

    Every node is built anew from the arguments it is constructed with, see
    :meth:`~cramph.node.StatechartNode.rebuilt_by`, an action from its parameters, see
    :meth:`~coraplex.robot_plans.actions.base.Action.rebound_to`, a statement as the
    candidate it is put in for, and any other node choosing its child as the child it
    runs now.
    """

    root: StatechartNode
    """
    The node rebuilt with everything below it.
    """

    candidates: dict[StatementNode, Action]
    """
    The candidate each statement of :attr:`root` runs once rebuilt; the branch of a
    statement without one ends before it.
    """

    left_out: set[StatechartNode] = field(default_factory=set)
    """
    The nodes below :attr:`root` left out of the rebuilt one, with everything below
    them.
    """

    _world: World | None = field(default=None, init=False, repr=False)
    """
    The world the current :meth:`build_for_trial` rebuilds the nodes onto.
    """

    _rebuilt: dict[StatechartNode, StatechartNode | None] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Every node the current rebuild rebuilt so far, so that a node referred to twice is
    rebuilt once.
    """

    _cut: bool = field(default=False, init=False, repr=False)
    """
    Whether a branch was cut below the node rebuilt last.
    """

    @staticmethod
    def root_around(node: StatechartNode) -> StatechartNode:
        """
        :param node: A node of a running plan.
        :return: The node holding everything that runs beside `node`: the outermost
            parallel above it with no action in between, or `node` itself if there is
            none. A parallel that is an action, and one above an action, are left out,
            because a trial builds an action anew from its parameters, which grounds
            what it runs anew.
        """
        root = node
        for ancestor in node.path:
            if isinstance(ancestor, Action):
                return root
            if isinstance(ancestor, Parallel):
                root = ancestor
        return root

    @classmethod
    def unfinished_nodes(
        cls, root: StatechartNode, proposals: dict[StatementNode, Action]
    ) -> SubtreeRebuild:
        """
        :param root: A running node of a plan.
        :param proposals: The candidate proposed for each statement being grounded,
            below `root` or `root` itself.
        :return: The rebuild of what is left to run of `root`: the nodes that ended are
            left out, since what they did is in the world already, a statement running
            a candidate keeps running it, and each statement of `proposals` runs its
            proposal.
        """
        descendants = root.descendants
        running_candidates = {
            node: node.grounded_nodes[-1]
            for node in descendants
            if isinstance(node, StatementNode)
            and node.latest_child is not None
            and not node.latest_child.life_cycle_state.is_terminal
        }
        return cls(
            root=root,
            candidates={**running_candidates, **proposals},
            left_out={
                node for node in descendants if node.life_cycle_state.is_terminal
            },
        )

    def build_for_trial(self, world: World) -> StatechartNode:
        """
        :param world: The copy the rebuilt nodes are tried in.
        :return: :attr:`root` rebuilt onto `world` to be tried: a statement runs its
            candidate in its place, so a failing candidate fails the trial, and every
            branch ends before its first statement without a candidate, since what
            follows that statement depends on how it is grounded.
        """
        self._world = world
        self._rebuilt = {}
        self._cut = False
        return self.rebuilt(self.root)

    def carry_over_conditions(self) -> None:
        """
        Gives every node rebuilt by the latest :meth:`build_for_trial` as a node of its
        own kind the transition conditions of the node it was rebuilt from, once it
        joined a statechart: reading the nodes rebuilt from those they read, and the
        current state of those left out. The rebuilt :attr:`root` keeps the conditions
        of what holds it, and a statement or another node choosing its child, which
        is rebuilt as something else, keeps those it was given there.
        """
        counterparts = {
            node: rebuilt
            for node, rebuilt in self._rebuilt.items()
            if rebuilt is not None
        }
        for node, rebuilt in counterparts.items():
            if node is self.root or isinstance(node, CompositeNodeChoosingItsChild):
                continue
            for transition_kind in TransitionKind:
                rebuilt.set_condition(
                    transition_kind,
                    DerivedConditionVariable.reading_counterparts(
                        node.get_condition(transition_kind), counterparts.get
                    ),
                )

    def rebuilt(self, node: StatechartNode) -> StatechartNode | None:
        """
        :return: `node` rebuilt onto :attr:`_world`, or None if it is left out, if a
            branch is cut at it, or if it cannot be rebuilt without what is left out.
        """
        if node not in self._rebuilt:
            self._rebuilt[node] = self._rebuild_anew(node)
        return self._rebuilt[node]

    def _rebuild_anew(self, node: StatechartNode) -> StatechartNode | None:
        """
        :return: `node` rebuilt onto :attr:`_world`, see :meth:`rebuilt`.
        """
        if node in self.left_out:
            return None
        if isinstance(node, StatementNode):
            return self._rebuild_statement(node)
        if isinstance(node, CompositeNodeChoosingItsChild):
            return self._rebuild_running_child(node)
        if isinstance(node, Action):
            return node.rebound_to(self._world)
        return node.rebuilt_by(self)

    def _rebuild_statement(self, statement: StatementNode) -> StatechartNode | None:
        """
        :return: The candidate of `statement` from :attr:`candidates`, or None if it
            has none, which cuts the branch.
        """
        candidate = self.candidates.get(statement)
        if candidate is None:
            self._cut = True
            return None
        return candidate.rebound_to(self._world)

    def _rebuild_running_child(
        self, node: CompositeNodeChoosingItsChild
    ) -> StatechartNode | None:
        """
        :return: The child `node` runs now, rebuilt, or None if it runs none, which cuts
            the branch, since what `node` will run is not chosen yet.
        """
        latest_child = node.latest_child
        if latest_child is not None and not latest_child.life_cycle_state.is_terminal:
            return self.rebuilt(latest_child)
        self._cut = True
        return None

    def rebuilt_children(
        self,
        owner: StatechartNode,
        children: list[StatechartNode],
    ) -> list[StatechartNode]:
        """
        Differs from :meth:`~cramph.node.NodeRebuild.rebuilt_children` in also leaving
        out, if `owner` runs its children in turn, every child after the first one a
        branch was cut in.
        """
        cut_before = self._cut
        cut_in_children = False
        rebuilt_children = []
        for child in children:
            self._cut = False
            rebuilt = self.rebuilt(child)
            if rebuilt is not None:
                rebuilt_children.append(rebuilt)
            cut_in_children = cut_in_children or self._cut
            if self._cut and isinstance(
                owner, ControlStructureNodeRunningItsChildrenInTurn
            ):
                break
        self._cut = cut_before or cut_in_children
        return rebuilt_children

    def rebound(self, value: Any) -> Any:
        """
        :return: `value` referring to :attr:`_world`.
        """
        return self._world.rebind_world_entities(value)

    def __str__(self) -> str:
        candidates = " beside ".join(map(str, self.candidates.values()))
        return f"{candidates} in {self.root.name}"


@dataclass
class ActionTrial:
    """
    Tries grounded actions against a copy of the world, to check that a candidate can
    succeed before it is attempted for real.

    One copy serves every candidate: after each attempt its model is rolled back and its
    state restored, and when the executor's world has changed since, the copy replays
    those model and state changes instead of being taken anew. Collision rules changed
    after the copy was taken are not carried over.

    Trials never publish to a synchronizer and always run simulated, in an executor of
    :attr:`trial_executor_type` over the copy, with the context extensions of
    :attr:`executor` rebound to the copy, and the base put where a candidate drives it
    by :attr:`plan_adaptations`.
    While the executor is debugging, the copy is shown in RViz under its own frame
    prefix and marker topic.
    """

    executor: PlanExecutor
    """
    The executor the candidates are grounded for.

    Only ever read from: a trial never mutates its world, and the candidates themselves
    are left untouched too, so they can still run for real afterwards.
    """

    trial_executor_type: type[PlanExecutor]
    """
    The executor that runs a candidate in the copy, which simulates the robot.
    """

    plan_adaptations: list[PlanTransformation] = field(
        default_factory=lambda: [
            WriteTheOdometryInsteadOfDriving(),
            KeepStretchInFrontOfTheContainerItCloses(),
        ],
        kw_only=True,
    )
    """
    The plan transformations adapting every candidate to being tried, in place of the
    plan adaptations of :attr:`trial_executor_type`.
    """

    copy_marker_alpha: float = field(default=0.9, kw_only=True)
    """
    The opacity the copy is drawn with while debugging, so it can be told apart from the
    world it copies where the two overlap.
    """

    _copied_world: World | None = field(default=None, init=False, repr=False)
    """
    The copy candidates are tried against.
    """

    _source_versions: tuple[int, int] | None = field(
        default=None, init=False, repr=False
    )
    """
    The model and state versions of the executor's world when the copy last matched it,
    used to notice that it has moved on and the copy has to be caught up.
    """

    _replayed_modification_blocks: int = field(default=0, init=False, repr=False)
    """
    How many of the modification blocks of the executor's world the copy already holds.
    """

    @property
    def world(self) -> World:
        """
        :return: The world the candidates are grounded in.
        """
        return self.executor.context.world

    def succeeds(self, rebuild: SubtreeRebuild) -> bool:
        """
        Run what `rebuild` builds for a trial against the copy, see
        :meth:`SubtreeRebuild.build_for_trial`, and restore the copy afterwards.

        It is rebuilt onto the copy because an action that modifies the model (attaching
        a grasped body, say) requires the entities it is given to belong to the world
        being modified, and because a statechart node belongs to one statechart only.
        The version to roll back to is read here rather than when the copy is taken, so
        each attempt undoes only its own modifications.

        :param rebuild: Builds what is tried.
        :return: True if what `rebuild` builds runs to completion without raising an
            :class:`~cramph.exceptions.ExecutionFailure`.
        """
        world = self._copy()
        version = world.get_world_model_manager().version

        with world.reset_state_context():
            try:
                executor = self._trial_executor(world)
                statechart = Statechart(context=executor.context)
                # A parallel is ended by what holds it.
                statechart.add_node(Sequence(nodes=[rebuild.build_for_trial(world)]))
                rebuild.carry_over_conditions()
                executor.compile(statechart)
                executor.execute()
                return True
            except ExecutionFailure as failure:
                logger.info(f"{rebuild} failed its trial: {failure}")
                return False
            finally:
                # Undo the model changes before leaving the reset context restores the
                # state, which needs the degrees of freedom it was snapshotted with.
                world.rollback_to_version(version)

    def _trial_executor(self, world: World) -> PlanExecutor:
        """
        :param world: The copy a candidate is tried in.
        :return: An executor running a candidate in `world`, configured like
            :attr:`executor`, the robot and everything else its context extensions refer
            to rebound to `world`, adapting the candidate with :attr:`plan_adaptations`,
            and showing `world` in RViz while it is debugging.
        """
        return self.trial_executor_type(
            StatechartContext.with_extensions(
                world,
                [
                    world.rebind_world_entities(extension)
                    for extension in self.executor.supplied_context_extensions
                ],
            ),
            robot_adaptations=self.executor.robot_adaptations,
            plan_adaptations=self.plan_adaptations,
            collision_avoidance=self.executor.collision_avoidance,
            extensions=self._trial_visualizations(),
        )

    def _trial_visualizations(self) -> list[ExecutorExtension]:
        """
        :return: The RViz visualization of the copy, through the ROS node of
            :attr:`executor`, under its own frame prefix and marker topic, while the
            executor is debugging; empty otherwise.
        """
        if not self.executor.debug:
            return []
        return [
            RvizVisualization(
                ros_node=self.executor.ros_node,
                collision_visualization=True,
                frame_prefix=ActionTrialVisualization.FRAME_PREFIX,
                marker_topic=ActionTrialVisualization.MARKER_TOPIC,
                marker_alpha=self.copy_marker_alpha,
            )
        ]

    def _copy(self) -> World:
        """
        :return: The copy to try candidates against, caught up with the executor's world
            if that has changed since the copy last matched it.
        """
        versions = (
            self.world.get_world_model_manager().version,
            self.world.state.version,
        )
        if self._copied_world is None:
            self._take_copy()
        elif self._source_versions != versions:
            self._catch_up()
        self._source_versions = versions
        return self._copied_world

    def _take_copy(self) -> None:
        """
        Copy the executor's world.
        """
        self._copied_world = deepcopy(self.world)
        self._replayed_modification_blocks = len(
            self.world.get_world_model_manager().model_modification_blocks
        )

    def _catch_up(self) -> None:
        """
        Bring the copy up to date with the executor's world: replay the modifications
        made to it since, the way copying it replays all of them, and take over its
        state.

        The copy's own modifications are all rolled back by then, so it still matches
        the world as it was when it last caught up.
        """
        modification_blocks = (
            self.world.get_world_model_manager().model_modification_blocks
        )
        with self._copied_world.modify_world():
            for block in modification_blocks[self._replayed_modification_blocks :]:
                block.update_references_for_world_and_apply(world=self._copied_world)
            self._copied_world.state.merge_state(self.world.state)
        self._replayed_modification_blocks = len(modification_blocks)

    def discard(self) -> None:
        """
        Release the copy, so the next trial takes a fresh one.
        """
        self._copied_world = None
        self._source_versions = None


# %% grounding underspecified actions while the statechart runs


@dataclass
class UnderspecifiedChildChooser(PlanChildChooser[StatementNode]):
    """
    Grounds the statement of every :class:`~cramph.composites.StatementNode` of a
    statechart into the action it runs next, in a sequence of its own for the nodes a
    plan transformation puts beside it.

    The statements that start with a parallel are grounded together, each into a
    candidate that suits the others, by the first of them that asks; every other one is
    handed its candidate then. A statement whose candidate failed grounds a candidate of
    its own.

    A candidate is tried out first, in what is left to run around it, see
    :meth:`SubtreeRebuild.unfinished_nodes`, so one that fails its trial is discarded
    without ever touching the real world, and a bad parameterization cannot poison a
    later attempt.
    """

    executor: PlanExecutor
    """
    The executor whose statechart holds the nodes, which also tries their candidates.
    """

    trial_executor_type: type[PlanExecutor]
    """
    The executor the candidates are tried in, see :class:`ActionTrial`.
    """

    trial: ActionTrial = field(init=False)
    """
    The trial every node tries its candidates against, so they all try them in one copy
    of the world.
    """

    def __post_init__(self) -> None:
        self.trial = ActionTrial(
            executor=self.executor, trial_executor_type=self.trial_executor_type
        )

    def choose_child(self, node: StatementNode) -> StatechartNode | None:
        """
        :return: The sequence holding the next candidate of `node` that succeeds its
            trial, beside the candidates handed to the statements grounded together
            with it, see :meth:`statements_grounded_with`; or None once no candidate,
            or no combination of them, is left that succeeds its trial.
        """
        grounding = node.context.require_extension(StatementGrounding)
        root = SubtreeRebuild.root_around(node)
        statements = self.statements_grounded_with(node)
        if len(statements) == 1:
            return self._candidate_alone(node, root, grounding)
        return self._candidate_together(node, statements, root, grounding)

    def _candidate_alone(
        self,
        node: StatementNode,
        root: StatechartNode,
        grounding: StatementGrounding,
    ) -> StatechartNode | None:
        """
        :return: The sequence holding the next candidate of `node` that succeeds its
            trial, or None once there is none.
        """
        candidate = node.next_candidate(
            grounding.query_backend, grounding.candidates_to_try
        )
        while candidate is not None:
            if self.trial.succeeds(
                SubtreeRebuild.unfinished_nodes(root, {node: candidate})
            ):
                return Sequence(name=f"{candidate.name}/steps", nodes=[candidate])
            candidate = node.next_candidate(
                grounding.query_backend, grounding.candidates_to_try
            )
        node.stop_grounding()
        return None

    def _candidate_together(
        self,
        node: StatementNode,
        statements: list[StatementNode],
        root: StatechartNode,
        grounding: StatementGrounding,
    ) -> StatechartNode | None:
        """
        Grounds `statements` anew, into the first combination of their candidates that
        succeeds its trial, and hands every statement but `node` its candidate of it.
        Once no combination is left, every one of `statements` gives up.

        :return: The sequence holding the candidate of `node` of that combination, or
            None once there is none.
        """
        for statement in statements:
            statement.stop_grounding()
        combinations = StatementsGroundedTogether(
            statements=statements,
            query_backend=grounding.query_backend,
            default_candidate_limit=grounding.candidates_to_try,
        )
        for combination in combinations:
            if not self.trial.succeeds(
                SubtreeRebuild.unfinished_nodes(root, combination)
            ):
                continue
            combinations.take(combination)
            for statement in statements:
                if statement is not node:
                    statement.choose_child_with(
                        CandidateHandOver(candidate=combination[statement])
                    )
            candidate = combination[node]
            return Sequence(name=f"{candidate.name}/steps", nodes=[candidate])
        for statement in statements:
            statement.stop_grounding()
            if statement is not node:
                statement.give_up()
        return None

    @classmethod
    def statements_grounded_with(cls, node: StatementNode) -> list[StatementNode]:
        """
        :param node: A statement asking for its next child.
        :return: The statements that start with the outermost parallel `node` starts
            with and that have chosen no child since their latest reset, `node` among
            them, whether they run yet or not; just `node` if it starts with no parallel
            or chose a child before.
        """
        parallel = next(
            (
                ancestor
                for ancestor in reversed(node.path)
                if isinstance(ancestor, Parallel)
                and node in cls.statements_starting_with(ancestor)
            ),
            None,
        )
        if parallel is None:
            return [node]
        return cls.statements_starting_with(parallel)

    @classmethod
    def statements_starting_with(cls, node: StatechartNode) -> list[StatementNode]:
        """
        :return: The statements that start when `node` starts and have chosen no child
            since their latest reset: `node` itself, or those of every child of a
            parallel and of the first child of a node running its children in turn.
            None in an action, which grounds its own.
        """
        if isinstance(node, StatementNode):
            return [node] if node.latest_child is None else []
        if isinstance(node, Action):
            return []
        if isinstance(node, Parallel):
            children = node.nodes
        elif isinstance(node, ControlStructureNodeRunningItsChildrenInTurn):
            children = node.nodes[:1]
        else:
            return []
        return [
            statement
            for child in children
            for statement in cls.statements_starting_with(child)
        ]

    def cleanup(self) -> None:
        """
        Release the trial's world copy.
        """
        self.trial.discard()

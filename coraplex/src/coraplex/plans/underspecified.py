from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields

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
    ControlStructureNode,
    ControlStructureNodeRunningItsChildrenInTurn,
    GenericChoosingNode,
    GenericNodeGroundingStatements,
    Parallel,
    Sequence,
    StatementNode,
    ParallelStatementNode,
)
from cramph.executor import ExecutorExtension
from cramph.exceptions import ExecutionFailure
from cramph.node import StatechartNode
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
class BuiltSubtree:
    """
    A node of a plan and everything below it rebuilt onto a world, and the candidate
    each of its statements that was grounded beforehand runs.
    """

    root: StatechartNode
    """
    The rebuilt node.
    """

    candidates: dict[StatementNode, Action]
    """
    The candidate each statement of :attr:`root` that was grounded beforehand runs.
    """

    originals: dict[StatementNode, StatementNode]
    """
    The statement each statement of :attr:`candidates` was rebuilt from.
    """

    def hand_over_candidates(self) -> None:
        """
        Let every statement of :attr:`candidates` run its candidate, once :attr:`root`
        joined a statechart.
        """
        for statement, candidate in self.candidates.items():
            statement.choose_child_with(CandidateHandOver(candidate=candidate))


@dataclass
class SubtreeRebuild:
    """
    Rebuilds a node of a plan and everything below it onto a world, putting candidates
    in for its statements.

    Every node is built anew from the fields it is constructed with, an action from its
    parameters, see :meth:`~coraplex.robot_plans.actions.base.Action.rebound_to`, a
    statement with the candidate it is put in for, see :meth:`build` and
    :meth:`build_for_trial`, and any other node choosing its child as the child it runs
    now.
    """

    root: StatechartNode
    """
    The node rebuilt with everything below it.
    """

    candidates: dict[StatementNode, Action]
    """
    The candidate each statement of :attr:`root` runs once rebuilt; a statement without
    one grounds one itself.
    """

    left_out: set[StatechartNode] = field(default_factory=set)
    """
    The nodes below :attr:`root` left out of the rebuilt one, with everything below
    them.
    """

    _world: World | None = field(default=None, init=False, repr=False)
    """
    The world the current :meth:`build` rebuilds the nodes onto.
    """

    _for_trial: bool = field(default=False, init=False, repr=False)
    """
    Whether the current rebuild is built for a trial, see :meth:`build_for_trial`.
    """

    _handed_over: dict[StatementNode, Action] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The candidate each statement rebuilt by the current :meth:`build` is handed.
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
    def remainder_of(
        cls, root: StatechartNode, proposals: dict[StatementNode, Action]
    ) -> SubtreeRebuild:
        """
        :param root: A running node of a plan.
        :param proposals: The candidate proposed for each statement being grounded,
            below `root`, `root` itself or below an
            :class:`~cramph.composites.ParallelStatementNode` there.
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

    def build(self, world: World) -> BuiltSubtree:
        """
        :param world: The world the rebuilt nodes refer to.
        :return: :attr:`root` rebuilt onto `world` to run, every statement a statement
            of its own, which grounds a candidate itself once the one it is handed
            failed.
        """
        root = self._rebuild_onto(world, for_trial=False)
        return BuiltSubtree(
            root=root,
            candidates=self._handed_over,
            originals={
                self._rebuilt[statement]: statement
                for statement in self.candidates
                if self._rebuilt.get(statement) in self._handed_over
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
        return self._rebuild_onto(world, for_trial=True)

    def _rebuild_onto(self, world: World, for_trial: bool) -> StatechartNode:
        """
        :param world: The world the rebuilt nodes refer to.
        :param for_trial: Whether the rebuild is built for a trial.
        :return: :attr:`root` rebuilt onto `world`.
        """
        self._world = world
        self._for_trial = for_trial
        self._handed_over = {}
        self._rebuilt = {}
        self._cut = False
        return self._rebuild_node(self.root)

    def _rebuild_node(self, node: StatechartNode) -> StatechartNode | None:
        """
        :return: `node` rebuilt onto :attr:`_world`, or None if it is left out.
        """
        if node not in self._rebuilt:
            self._rebuilt[node] = self._rebuild_anew(node)
        return self._rebuilt[node]

    def _rebuild_anew(self, node: StatechartNode) -> StatechartNode | None:
        """
        :return: `node` rebuilt onto :attr:`_world`, or None if it is left out, if a branch is
            cut at it, or if it is a control structure node left with no child.
        """
        if node in self.left_out:
            return None
        if isinstance(node, StatementNode):
            return self._rebuild_statement(node)
        if isinstance(node, CompositeNodeChoosingItsChild):
            return self._rebuild_running_child(node)
        if isinstance(node, Action):
            return node.rebound_to(self._world)
        arguments = {
            constructor_field.name: self._rebuild_value(
                node, getattr(node, constructor_field.name)
            )
            for constructor_field in fields(node)
            if constructor_field.init
        }
        if isinstance(node, ControlStructureNode) and not arguments["nodes"]:
            return None
        return type(node)(**arguments)

    def _rebuild_statement(self, statement: StatementNode) -> StatechartNode | None:
        """
        :return: For a trial, the candidate of `statement` from :attr:`candidates`, or
            None if it has none; otherwise a statement of its own grounding the
            statement of `statement`, handed that candidate if it has one.
        """
        candidate = self.candidates.get(statement)
        if self._for_trial:
            if candidate is None:
                self._cut = True
                return None
            return candidate.rebound_to(self._world)
        rebuilt = StatementNode(statement=statement.statement)
        if candidate is not None:
            self._handed_over[rebuilt] = candidate.rebound_to(self._world)
        return rebuilt

    def _rebuild_running_child(
        self, node: CompositeNodeChoosingItsChild
    ) -> StatechartNode | None:
        """
        :return: The child `node` runs now, rebuilt; the parallel of an
            :class:`~cramph.composites.ParallelStatementNode` that runs none yet,
            rebuilt with the candidates of its statements; or None for any other node
            that runs none, which cuts the branch, since what `node` will run is not
            chosen yet.
        """
        latest_child = node.latest_child
        if latest_child is not None and not latest_child.life_cycle_state.is_terminal:
            return self._rebuild_node(latest_child)
        if isinstance(node, ParallelStatementNode):
            return self._rebuild_node(node.parallel)
        self._cut = True
        return None

    def _rebuild_value(self, owner: StatechartNode, value: Any) -> Any:
        """
        :param owner: The node `value` is a constructor argument of.
        :return: `value` referring to :attr:`_world`, the nodes in it rebuilt.
        """
        if isinstance(value, StatechartNode):
            return self._rebuild_node(value)
        if isinstance(value, list) and any(
            isinstance(element, StatechartNode) for element in value
        ):
            return self._rebuild_children(owner, value)
        return self._world.rebind_world_entities(value)

    def _rebuild_children(
        self,
        owner: StatechartNode,
        children: list[StatechartNode],
    ) -> list[StatechartNode]:
        """
        :param owner: The node holding `children`.
        :return: `children` rebuilt, leaving out those left out, and, if `owner` runs
            its children in turn, every child after the first one a branch was cut in.
        """
        cut_before = self._cut
        cut_in_children = False
        rebuilt_children = []
        for child in children:
            self._cut = False
            rebuilt = self._rebuild_node(child)
            if rebuilt is not None:
                rebuilt_children.append(rebuilt)
            cut_in_children = cut_in_children or self._cut
            if self._cut and isinstance(
                owner, ControlStructureNodeRunningItsChildrenInTurn
            ):
                break
        self._cut = cut_before or cut_in_children
        return rebuilt_children

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
            universal_adaptations=self.executor.universal_adaptations,
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
class ChooserGroundingStatements(PlanChildChooser[GenericNodeGroundingStatements], ABC):
    """
    Grounds the statements of a node into the first combination of their candidates
    whose trial succeeds, and runs that combination as the node's next child.

    A combination is tried out first, so one that fails its trial is discarded without
    ever touching the real world, and a bad parameterization cannot poison a later
    attempt. It is tried in what is left to run around the node, see
    :meth:`SubtreeRebuild.remainder_of`, beside what runs in a parallel with it.
    """

    trial: ActionTrial
    """
    The trial every node tries its combinations against, so they all try them in one
    copy of the world.
    """

    def choose_child(
        self, node: GenericNodeGroundingStatements
    ) -> StatechartNode | None:
        """
        :return: The child running the next combination that succeeds its trial, or
            None once the node has no combination left that succeeds its trial.
        """
        grounding = node.context.require_extension(StatementGrounding)
        root = SubtreeRebuild.root_around(node)
        combination = node.next_combination(
            grounding.query_backend, grounding.candidates_to_try
        )
        while combination is not None:
            if self.trial.succeeds(SubtreeRebuild.remainder_of(root, combination)):
                return self.child_running(node, combination)
            combination = node.next_combination(
                grounding.query_backend, grounding.candidates_to_try
            )
        node.stop_grounding()
        return None

    @abstractmethod
    def child_running(
        self,
        node: GenericNodeGroundingStatements,
        combination: dict[StatementNode, Action],
    ) -> StatechartNode:
        """
        :param node: The node whose statements were grounded.
        :param combination: The candidate of each statement of `node`, which succeeded
            its trial.
        :return: The child of `node` running `combination`.
        """

    def cleanup(self) -> None:
        """
        Release the trial's world copy.
        """
        self.trial.discard()


@dataclass
class UnderspecifiedChildChooser(ChooserGroundingStatements[StatementNode]):
    """
    Grounds the statement of every :class:`~cramph.composites.StatementNode` of a
    statechart into the action it runs next, each time its previous action failed.
    """

    def child_running(
        self, node: StatementNode, combination: dict[StatementNode, Action]
    ) -> StatechartNode:
        """
        :return: The candidate of `node`, in a sequence of its own for the nodes a plan
            transformation puts beside it.
        """
        candidate = combination[node]
        return Sequence(name=f"{candidate.name}/steps", nodes=[candidate])


@dataclass
class ParallelStatementChildChooser(ChooserGroundingStatements[ParallelStatementNode]):
    """
    Grounds the statements of every
    :class:`~cramph.composites.ParallelStatementNode` together, and runs its parallel
    with them.

    It chooses once per run of the node: a statement whose child failed grounds a
    candidate of its own, see :class:`UnderspecifiedChildChooser`, so a branch that
    succeeded never runs again.
    """

    _built: dict[ParallelStatementNode, BuiltSubtree] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The parallel built for each node, until it joined the statechart.
    """

    def child_running(
        self,
        node: ParallelStatementNode,
        combination: dict[StatementNode, Action],
    ) -> StatechartNode:
        """
        :return: The parallel of `node`, its statements handed their candidates, held by
            a sequence, which ends it.
        """
        built = SubtreeRebuild(root=node.parallel, candidates=combination).build(
            node.context.world
        )
        # Each statement goes on from its candidate once that failed, rather than
        # grounding its statement anew from the first candidate.
        for rebuilt, statement in built.originals.items():
            rebuilt.ground_on_from(
                statement, node.candidates_read_for(statement), combination[statement]
            )
        self._built[node] = built
        return Sequence(nodes=[built.root])

    def prepare_chosen_child(
        self, node: ParallelStatementNode, child: StatechartNode
    ) -> None:
        """
        Differs from :meth:`PlanChildChooser.prepare_chosen_child` in first handing
        every grounded statement of the parallel its candidate, so that the parallel
        arrives wherever it runs with them chosen.
        """
        self._built.pop(node).hand_over_candidates()
        super().prepare_chosen_child(node, child)

    def failure_monitors_for(
        self, node: ParallelStatementNode, child: StatechartNode
    ) -> list[StatechartNode]:
        """
        Differs from :meth:`PlanChildChooser.failure_monitors_for` in handing none:

        every candidate is watched for a stall by a monitor of its own, and a branch
        grounding a candidate anew stops nothing beside it.
        """
        return []

    def cleanup(self) -> None:
        """
        Differs from :meth:`ChooserGroundingStatements.cleanup` in also releasing the
        parallels not yet chosen.
        """
        super().cleanup()
        self._built.clear()


# %% grounding the statements that start with a parallel together


@dataclass
class GroundTheStatementsOfAParallelTogether(PlanTransformation[Parallel]):
    """
    Puts the branches of every parallel that start with a statement into one
    :class:`~cramph.composites.ParallelStatementNode`, so that their statements are
    grounded together, each to a candidate suiting the others, before any of them runs.

    The other branches stay in the parallel and start with it. A parallel whose branches
    all start with a statement, or that needs a number of its children to succeed, is put
    into the :class:`~cramph.composites.ParallelStatementNode` as a whole.
    """

    def is_applicable(self, plan_node: Parallel) -> bool:
        holder = plan_node.parent_node
        return (
            not isinstance(plan_node, Action)
            and (holder is None or isinstance(holder, ControlStructureNode))
            and bool(self.statements_starting_with(plan_node))
        )

    def apply(self, plan_node: Parallel) -> None:
        statechart = plan_node.statechart
        statements = self.statements_starting_with(plan_node)
        grouped = [
            branch
            for branch in plan_node.nodes
            if self.statements_starting_with(branch)
        ]
        if (
            len(grouped) == len(plan_node.nodes)
            or plan_node.minimum_success is not None
        ):
            statechart.replace(
                plan_node,
                ParallelStatementNode(parallel=plan_node, statements=statements),
            )
            return
        first, *others = grouped
        for branch in others:
            statechart.remove_node(branch)
        statechart.replace(
            first,
            ParallelStatementNode(
                parallel=Parallel(nodes=grouped), statements=statements
            ),
        )

    @classmethod
    def statements_starting_with(cls, node: StatechartNode) -> list[StatementNode]:
        """
        :return: The statements that start when `node` starts and have not chosen a
            child: `node` itself, or those of every child of a parallel and of the first
            child of a node running its children in turn. None in an action, which
            grounds its own.
        """
        if isinstance(node, StatementNode):
            return [] if node.chosen_children else [node]
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

from __future__ import annotations

import logging
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field

from typing_extensions import (
    TYPE_CHECKING,
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
    GenericChoosingNode,
    Sequence,
    StatementNode,
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

    def succeeds(self, action: Action) -> bool:
        """
        Run `action` against the copy and restore the copy afterwards.

        The action is rebuilt from its own parameters, rebound onto the copy, because an
        action that modifies the model (attaching a grasped body, say) requires the
        entities it is given to belong to the world being modified, and because a
        statechart node belongs to one statechart only.

        The version to roll back to is read here rather than when the copy is taken, so
        each attempt undoes only its own modifications.

        :param action: The grounded action to try out.
        :return: True if `action` runs to completion without raising an
            :class:`~cramph.exceptions.ExecutionFailure`.
        """
        world = self._copy()
        candidate = self._rebind_action_to_world(action, world)
        version = world.get_world_model_manager().version

        with world.reset_state_context():
            try:
                executor = self._trial_executor(world)
                statechart = Statechart(context=executor.context)
                # The candidate runs in a sequence of its own, the way it runs for real,
                # so the nodes a plan transformation puts beside it are tried with it.
                statechart.add_node(Sequence(nodes=[candidate]))
                executor.compile(statechart)
                executor.execute()
                return True
            except ExecutionFailure as failure:
                logger.info(f"{action} failed its trial: {failure}")
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

    @classmethod
    def _rebind_action_to_world(cls, action: Action, world: World) -> Action:
        """
        :param action: The grounded action to try out.
        :param world: The copy to try it against.
        :return: A new action with the parameters of `action`, referring to `world`.
            The actions among those parameters are built anew the same way, since each
            of them becomes a node of the statechart the trial runs.
        """
        return type(action)(
            **{
                name: (
                    cls._rebind_action_to_world(value, world)
                    if isinstance(value, Action)
                    else world.rebind_world_entities(value)
                )
                for name, value in action.designator_parameter.items()
            }
        )

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

    A grounded action is tried out first, so one that fails its trial is discarded
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
        :return: The sequence holding the next grounded action that succeeds its
            trial, or None once the statement ran out of actions that succeed their
            trial.
        """
        grounding = node.context.require_extension(StatementGrounding)
        proposal = node.next_candidate(
            grounding.query_backend, grounding.candidates_to_try
        )
        while proposal is not None:
            if self.trial.succeeds(proposal):
                return Sequence(name=f"{proposal.name}/steps", nodes=[proposal])
            proposal = node.next_candidate(
                grounding.query_backend, grounding.candidates_to_try
            )
        node.stop_grounding()
        return None

    def cleanup(self) -> None:
        """
        Release the trial's world copy.
        """
        self.trial.discard()

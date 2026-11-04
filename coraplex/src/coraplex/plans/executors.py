from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta

from rclpy.node import Node
from typing_extensions import TYPE_CHECKING

from coraplex.exceptions import (
    DebuggingWithoutRosNode,
    MotionDidNotFinish,
    PlanNotCompiled,
)
from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from coraplex.plans.failures import (
    CandidateLimitReached,
    EmptyUnderspecified,
    MotionExceededSimulationTimeLimit,
    MotionMadeNoProgress,
    MotionViolatedCollisionAvoidance,
)
from coraplex.plans.plan_transformation import PlanRewriting, PlanTransformation
from coraplex.plans.underspecified import UnderspecifiedChildChooser
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.plan_transformations import (
    CloseFingersToObjectWidth,
    DetectWithRoboKudo,
    LetContactCarryTheBodies,
    WriteTheOdometryInsteadOfCommandingAPose,
)
from coraplex.robot_plans.robot_specific_plan_transformations import (
    DriveWithTheDifferentialDrive,
    KeepStretchFromTurningWhileMovingItsTool,
    KeepStretchInFrontOfTheContainerItCloses,
    KeepTheTorsoUprightWhileUsingATool,
    MoveStretchToolCenterPointInAStraightLine,
)
from cramph.composites import ChildChooserAccess, StatementNode
from cramph.context import ContextExtension
from cramph.data_types import LifeCycleValues
from cramph.exceptions import EmptyStatechartError
from cramph.executor import (
    Executor,
    ExecutorExtension,
    StatechartExecutor,
    SteppedSimulationPacer,
)
from cramph.node import StatechartNode, TerminalNode
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.exceptions import (
    CollisionViolatedError,
    NoProgressError,
)
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
)
from giskardpy.motion_statechart.graph_node import ConvergingTask, EndMotion
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from giskardpy.middleware.ros2.giskard_client_executor import GiskardClientExecutor
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.symbolic_math.symbolic_math import logic_or
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.multi_sim import MujocoSim

logger = logging.getLogger(__name__)


# %% executing a plan


@dataclass
class PlanExecutor(Executor, ABC):
    """
    Executes a plan: the top-level nodes of a statechart built in :attr:`context`.

    The extensions of :attr:`context` describe the robot and how the plan is grounded,
    such as the :class:`~coraplex.plans.context_extensions.RobotAccess` of the robot
    performing it. A :class:`~coraplex.plans.context_extensions.StatementGrounding` and a
    :class:`~coraplex.plans.plan_transformation.PlanRewriting` with default values are
    added unless one is there; the executor replaces the latter with one offering every
    node to :attr:`robot_adaptations` and :attr:`plan_adaptations` first. A context
    serves one executor. :meth:`compile` adds what a plan runs with: the collision
    avoidance of the robot of :attr:`context` if asked for, one
    :class:`~giskardpy.motion_statechart.graph_node.EndMotion` once every plan node
    succeeded or one of them ended without succeeding, and the stall detection.
    Underspecified actions are grounded while it runs, see
    :class:`~coraplex.plans.underspecified.UnderspecifiedChildChooser`.
    """

    supplied_context_extensions: list[ContextExtension] = field(init=False)
    """
    The extensions :attr:`context` held when it was given to this executor, before the
    executor added its own.
    """

    ros_node: Node | None = field(default=None, kw_only=True)
    """
    The ROS node the plan communicates through, if any.
    """

    collision_avoidance: bool = field(default=False, kw_only=True)
    """
    Whether the robot avoids colliding with its surroundings and with itself.
    """

    debug: bool = field(default=False, kw_only=True)
    """
    Whether every copy of the world a candidate is tried in is published to RViz, which
    needs :attr:`ros_node`.
    """

    child_chooser: UnderspecifiedChildChooser = field(init=False)
    """
    Grounds the underspecified actions of the plan while it runs.
    """

    plan_nodes: list[StatechartNode] = field(default_factory=list, init=False)
    """
    The top-level nodes of the plan compiled last.
    """

    robot_adaptations: list[PlanTransformation] = field(
        default_factory=lambda: [
            DriveWithTheDifferentialDrive(),
            MoveStretchToolCenterPointInAStraightLine(),
            KeepStretchFromTurningWhileMovingItsTool(),
            KeepTheTorsoUprightWhileUsingATool(),
        ],
        kw_only=True,
    )
    """
    The plan transformations adapting every plan to the robot performing each of its
    nodes, however the executor runs it, offered every node before
    :attr:`plan_adaptations`.
    """

    plan_adaptations: list[PlanTransformation] = field(
        default_factory=list, kw_only=True
    )
    """
    The plan transformations adapting every plan to how this executor runs it, offered
    every node before the transformations of the plan's own
    :class:`~coraplex.plans.plan_transformation.PlanRewriting`.
    """

    def __post_init__(self) -> None:
        if self.debug and self.ros_node is None:
            raise DebuggingWithoutRosNode()
        self.supplied_context_extensions = list(self.context.extensions.values())
        self.child_chooser = UnderspecifiedChildChooser(
            executor=self, trial_executor_type=KinematicallySimulatedPlanExecutor
        )
        # Raises for a context another executor runs on before anything of it changed.
        self.context.add_extension(ChildChooserAccess(choosers=[self.child_chooser]))
        rewriting = self.context.get_extension(PlanRewriting)
        if rewriting is not None:
            self.context.extensions[type(rewriting)] = (
                rewriting.copy_with_preceding_transformations(self.adaptations)
            )
        super().__post_init__()

    def default_context_extensions(self) -> list[ContextExtension]:
        return [
            StatementGrounding(),
            PlanRewriting(transformations=self.adaptations),
        ]

    @property
    def adaptations(self) -> list[PlanTransformation]:
        """
        :return: The transformations this executor rewrites every plan with before the
            plan's own: :attr:`robot_adaptations`, then :attr:`plan_adaptations`.
        """
        return [*self.robot_adaptations, *self.plan_adaptations]

    def default_executor_extensions(self) -> list[ExecutorExtension]:
        return []

    def prepare(self, statechart: Statechart) -> None:
        """
        Add what the plan in `statechart` runs with, and rewrite it with the plan
        transformations, without compiling it.

        :param statechart: The statechart holding the plan, built in :attr:`context`.
        :raises StatechartOfDifferentContextError: If `statechart` was not built in
            :attr:`context`.
        :raises EmptyStatechartError: If `statechart` holds no node.
        """
        Executor.compile(self, statechart)
        plan_nodes = list(statechart.top_level_nodes)
        if not plan_nodes:
            raise EmptyStatechartError()
        rewriting = self.context.require_extension(PlanRewriting)
        for plan_node in plan_nodes:
            rewriting.rewrite(plan_node)
        # a transformation may have put a plan node below a node holding it
        self.plan_nodes = list(statechart.top_level_nodes)
        if self.collision_avoidance:
            robot = self.context.require_extension(RobotAccess).robot
            statechart.add_node(ExternalCollisionAvoidance(robot=robot))
            statechart.add_node(SelfCollisionAvoidance(robot=robot))
        self._add_end_of_plan(statechart)
        self._add_stall_detection(statechart)

    def _add_end_of_plan(self, statechart: Statechart) -> None:
        """
        End the statechart once every plan node succeeded, or one of them ended without
        succeeding, so the plan no longer can.

        :param statechart: The statechart the plan was added to.
        """
        end_of_plan = EndMotion.when_all_true(self.plan_nodes)
        end_of_plan.start_condition = logic_or(
            end_of_plan.start_condition,
            *[plan_node.is_failed_or_interrupted for plan_node in self.plan_nodes],
        )
        statechart.add_node(end_of_plan)

    def _add_stall_detection(self, statechart: Statechart) -> None:
        """
        Cancel the statechart once a plan node stops approaching its goal.

        Only the motions a plan node holds when it is compiled are watched; an action a
        statement node chooses later is watched by its own stall monitor. A plan node
        holding no motion yet is not watched at all, since it would read as stalled from
        its first tick.

        :param statechart: The statechart the plan was added to, which expanded it.
        """
        for plan_node in self.plan_nodes:
            if not any(
                isinstance(node, ConvergingTask)
                for node in [plan_node, *plan_node.descendants]
            ):
                continue
            still_progressing = StillProgressing(monitored_node=plan_node)
            statechart.add_node(still_progressing)
            statechart.add_node(still_progressing.cancel_motion())

    def compile(self, statechart: Statechart) -> None:
        """
        Prepare the plan in `statechart`, see :meth:`prepare`, and compile it, so that
        :meth:`execute` can run it.

        :param statechart: The statechart holding the plan, built in :attr:`context`.
        """
        self.prepare(statechart)
        super().compile(statechart)

    def execute(self) -> None:
        """
        Run the compiled plan until every plan node succeeded.

        :raises PlanNotCompiled: If no plan was compiled.
        :raises MotionDidNotFinish: If the plan did not succeed.
        :raises EmptyUnderspecified: If an underspecified action ran out of actions to
            try.
        :raises CandidateLimitReached: If an underspecified action tried as many actions
            as it may.
        :raises MotionMadeNoProgress: When the plan stops approaching its goal.
        :raises MotionExceededSimulationTimeLimit: When a simulated plan runs for longer
            than :attr:`SimulatedPlanExecutor.simulation_time_limit`.
        :raises MotionViolatedCollisionAvoidance: When the plan brings bodies closer to
            each other than collision avoidance allows.
        """
        if self.statechart is None:
            raise PlanNotCompiled()
        try:
            self._run()
        except NoProgressError as stalled:
            raise MotionMadeNoProgress(stalled) from stalled
        except CollisionViolatedError as violation:
            raise MotionViolatedCollisionAvoidance(violation) from violation
        self._raise_if_plan_failed()

    @abstractmethod
    def _run(self) -> None:
        """
        Run the compiled plan until it ended, leaving every node in the life cycle state
        it ended in.
        """

    def close(self) -> None:
        """
        Release what this executor acquired to run its plans, once no further plan is
        run with it.
        """

    def _plan_failed(self) -> bool:
        """
        :return: Whether a plan node ended without succeeding, so the plan no longer
            can succeed.
        """
        return any(
            plan_node.life_cycle_state.is_terminal
            and plan_node.life_cycle_state != LifeCycleValues.SUCCEEDED
            for plan_node in self.plan_nodes
        )

    def _raise_if_plan_failed(self) -> None:
        """
        :raises CandidateLimitReached: If a statement node tried as many actions
            as it may, which is why the plan failed.
        :raises EmptyUnderspecified: If a statement node ran out of actions to
            try, which is why the plan failed.
        :raises MotionDidNotFinish: If the plan failed for any other reason.
        """
        if not self._plan_failed():
            return
        self._raise_if_out_of_candidates()
        motion_did_not_finish = MotionDidNotFinish(
            [
                node
                for node in self.statechart.nodes
                if not isinstance(node, TerminalNode)
                and node.life_cycle_state
                not in [LifeCycleValues.SUCCEEDED, LifeCycleValues.NOT_STARTED]
            ]
        )
        logger.error(motion_did_not_finish.error_message())
        raise motion_did_not_finish

    def _raise_if_out_of_candidates(self) -> None:
        """
        :raises CandidateLimitReached: If a statement node tried as many actions
            as it may.
        :raises EmptyUnderspecified: If a statement node ran out of actions to
            try.
        """
        for node in self.statechart.get_nodes_by_type(StatementNode):
            if not node.ran_out_of_children:
                continue
            if node.reached_candidate_limit:
                raise CandidateLimitReached(
                    node=node, candidate_limit=node.candidate_limit
                )
            raise EmptyUnderspecified(node=node)


@dataclass
class SimulatedPlanExecutor(PlanExecutor, StatechartExecutor, ABC):
    """
    Ticks the plan's statechart in a simulated world, driving the robot's motions with
    the :class:`~giskardpy.motion_control.MotionControl` among its :attr:`extensions`,
    which is added with the controller of :meth:`create_qp_controller_config` unless one
    is given.
    """

    simulation_time_per_action: timedelta = field(
        default=timedelta(seconds=40), kw_only=True
    )
    """
    The simulated time every action of the plan adds to :attr:`simulation_time_limit`.
    """

    simulation_time_limit: timedelta = field(init=False)
    """
    The simulated time after which the plan is given up on, however it is progressing:

    :attr:`simulation_time_per_action` for every action in the statechart, and at least
    once. Counted again whenever nodes join the statechart.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.simulation_time_limit = self.simulation_time_per_action

    def default_executor_extensions(self) -> list[ExecutorExtension]:
        return [
            RosNodeAccess(self.ros_node),
            MotionControl(qp_controller_config=self.create_qp_controller_config()),
        ]

    @abstractmethod
    def create_qp_controller_config(self) -> QPControllerConfig:
        """
        :return: A new configuration of the controller driving the robot's motions,
            suited to how the world is simulated.
        """

    def compile(self, statechart: Statechart) -> None:
        """
        Differs from :meth:`PlanExecutor.compile` in also counting
        :attr:`simulation_time_limit`.
        """
        super().compile(statechart)
        self._count_simulation_time_limit()

    def after_recompile(self) -> None:
        """
        Differs from :meth:`~cramph.executor.StatechartExecutor.after_recompile` in also
        counting :attr:`simulation_time_limit` again, since nodes joined.
        """
        super().after_recompile()
        self._count_simulation_time_limit()

    def _count_simulation_time_limit(self) -> None:
        """
        Set :attr:`simulation_time_limit` from the actions the statechart holds now.
        """
        actions = len(self.statechart.get_nodes_by_type(Action))
        self.simulation_time_limit = self.simulation_time_per_action * max(1, actions)

    def _run(self) -> None:
        """
        Tick the statechart until it or the plan ended, waiting for :attr:`pacer` after
        every tick and, once it ended, until the world caught up with the last command.

        The statechart's own stall monitor decides when a plan is hopeless;
        :attr:`simulation_time_limit` only cuts off a plan that keeps creeping for
        longer than the actions it holds are given.

        :raises NoProgressError: When the plan stops approaching its goal.
        :raises MotionExceededSimulationTimeLimit: When the plan runs for longer than
            :attr:`simulation_time_limit`.
        """
        try:
            while not self._is_over():
                if timedelta(seconds=self.time) >= self.simulation_time_limit:
                    raise MotionExceededSimulationTimeLimit(self.simulation_time_limit)
                self.tick()
                self.pacer.sleep()
            # The controller stops commanding, so a velocity-servoed base stops too.
            MotionControl.set_velocity_acceleration_jerk_to_zero(self.context.world)
            self.context.world.notify_state_change()
            self.pacer.wait_for_standstill()
        finally:
            MotionControl.set_velocity_acceleration_jerk_to_zero(self.context.world)
            self.finish_run()

    def _is_over(self) -> bool:
        """
        :return: Whether the statechart ended, every plan node succeeded or the plan
            failed.
        """
        return (
            self.statechart.is_ended()
            or self._plan_failed()
            or all(
                plan_node.life_cycle_state == LifeCycleValues.SUCCEEDED
                for plan_node in self.plan_nodes
            )
        )


@dataclass
class KinematicallySimulatedPlanExecutor(SimulatedPlanExecutor):
    """
    Ticks the plan's statechart in a world whose state follows every command exactly,
    with no physics in between.
    """

    plan_adaptations: list[PlanTransformation] = field(
        default_factory=lambda: [
            WriteTheOdometryInsteadOfCommandingAPose(),
            KeepStretchInFrontOfTheContainerItCloses(),
        ],
        kw_only=True,
    )

    def create_qp_controller_config(self) -> QPControllerConfig:
        return QPControllerConfig.create_with_fast_simulation_defaults()


@dataclass
class PhysicallySimulatedPlanExecutor(SimulatedPlanExecutor):
    """
    Ticks the plan's statechart against a physically simulated world, whose servos take
    every command as their set point while the physics advances one control period
    between two ticks.

    An object is held by friction alone, so a grasp that misses drops it.
    """

    simulation: MujocoSim = field(kw_only=True)
    """
    The physically simulated world the plan is performed in; it has to be started with
    :meth:`~semantic_digital_twin.adapters.multi_sim.MujocoSim.start_stepped_simulation`
    already.
    """

    pacer: SteppedSimulationPacer = field(init=False)
    """
    Steps :attr:`simulation` one control period after every tick.
    """

    plan_adaptations: list[PlanTransformation] = field(
        default_factory=lambda: [LetContactCarryTheBodies()],
        kw_only=True,
    )
    """
    The plan transformations adapting every plan to being performed in physics. Unless
    one is given, a :class:`~coraplex.robot_plans.plan_transformations.CloseFingersToObjectWidth`
    is put first that squeezes by the
    :attr:`~semantic_digital_twin.robots.robot_part_mixins.HasTwoFingers.squeeze` of the
    robot's two-finger grippers.
    """

    def __post_init__(self) -> None:
        self.pacer = SteppedSimulationPacer(self.simulation)
        if not any(
            isinstance(adaptation, CloseFingersToObjectWidth)
            for adaptation in self.plan_adaptations
        ):
            self.plan_adaptations = [
                CloseFingersToObjectWidth(squeeze=self._squeeze_of_the_grippers()),
                *self.plan_adaptations,
            ]
        super().__post_init__()

    def _squeeze_of_the_grippers(self) -> float:
        """
        :return: The squeeze the two-finger grippers of the robot in :attr:`context`
            name, or the default one where it has none.
        """
        robot = self.context.require_extension(RobotAccess).robot
        [squeeze] = {
            arm.end_effector.squeeze
            for arm in robot.all_arms
            if isinstance(arm.end_effector, HasTwoFingers)
        } or {HasTwoFingers.squeeze}
        return squeeze

    def create_qp_controller_config(self) -> QPControllerConfig:
        return QPControllerConfig.create_with_physical_simulation_defaults()


@dataclass
class RealRobotPlanExecutor(PlanExecutor):
    """
    Sends the plan's statechart to Giskard, grounding every underspecified action
    Giskard reaches against the world as it is then.

    Giskard compiles the statechart once it receives it.
    """

    _giskard: GiskardClientExecutor | None = field(default=None, init=False, repr=False)
    """
    The connection to Giskard every plan is sent through, None until the first plan
    runs.
    """

    plan_adaptations: list[PlanTransformation] = field(
        default_factory=lambda: [CloseFingersToObjectWidth(), DetectWithRoboKudo()],
        kw_only=True,
    )

    def compile(self, statechart: Statechart) -> None:
        """
        Does what :meth:`PlanExecutor.compile` does, then tells every extension that the
        statechart is compiled.
        """
        super().compile(statechart)
        self._notify_compiled()

    def _run(self) -> None:
        """
        Send the statechart to Giskard and wait until it ended there, then tell every
        extension that the run stopped.
        """
        if self._giskard is None:
            self._giskard = GiskardClientExecutor(self.ros_node, context=self.context)
        try:
            self._giskard.compile(self.statechart)
            self._giskard.execute()
        finally:
            self.finish_run()

    def close(self) -> None:
        if self._giskard is None:
            return
        self._giskard.close()
        self._giskard = None

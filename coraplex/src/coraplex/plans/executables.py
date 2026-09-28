from __future__ import annotations

import logging
from contextlib import AbstractContextManager, ExitStack, nullcontext
from datetime import datetime
from dataclasses import dataclass, field

from typing_extensions import Callable, List, Dict, ClassVar, Optional, TYPE_CHECKING

from coraplex.datastructures.enums import ExecutionType
from coraplex.exceptions import (
    MotionDidNotFinish,
    ConditionNotSatisfied,
    UnknownExecutionType,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
)
from giskardpy.motion_statechart.graph_node import CancelMotion
from giskardpy.motion_statechart.graph_node import EndMotion, Goal, Task
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
    StateHistoryObserver,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.ros_executor import Ros2Executor
from krrood.entity_query_language.factories import evaluate_condition
from krrood.symbolic_math.symbolic_math import Scalar, trinary_logic_not
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import StateHistory
    from coraplex.robot_plans.actions.base import ActionDescription

    from coraplex.plans.condition_nodes import ConditionNode
    from coraplex.plans.plan_node import MotionNode
    from coraplex.plans.underspecified import UnderspecifiedNode
    from coraplex.datastructures.dataclasses import Context

logger = logging.getLogger(__name__)


# %% native motion history


@dataclass
class MotionPlanHistory(StateHistoryObserver):
    """
    Project native motion history onto the corresponding plan nodes.
    """

    statechart: MotionStatechart
    """
    The chart whose recorded transitions drive plan progress.
    """

    motion_mappings: dict[MotionNode, Task]
    """
    Plan motions and the native tasks realizing them.
    """

    def __post_init__(self) -> None:
        """
        Bind plan motions before observing compilation and execution.
        """
        for node in self.motion_mappings:
            node.motion_statechart = self.statechart
        self.statechart.history.add_observer(self)

    def on_state_change(self, history: StateHistory) -> None:
        """
        Publish the transitions recorded in the newest native snapshot.

        :param history: The updated native state history.
        """
        current = history.history[-1].life_cycle_state
        previous = history.history[-2].life_cycle_state if len(history) > 1 else None
        for node, task in self.motion_mappings.items():
            current_state = LifeCycleValues(int(current[task]))
            previous_state = (
                LifeCycleValues(int(previous[task]))
                if previous is not None
                else LifeCycleValues.NOT_STARTED
            )
            if current_state == previous_state:
                continue
            if current_state == LifeCycleValues.NOT_STARTED:
                if previous_state in (LifeCycleValues.RUNNING, LifeCycleValues.PAUSED):
                    self._end_motion(node, LifeCycleValues.INTERRUPTED)
                node.status = current_state
                node.start_time = None
                node.end_time = None
                continue
            if previous_state == LifeCycleValues.NOT_STARTED:
                node.status = LifeCycleValues.RUNNING
                node.start_time = datetime.now()
                node.end_time = None
                node.plan.notify_node_started(node)
            node.status = current_state
            if current_state.is_terminal:
                self._end_motion(node, current_state)

    def end_active_motions(self, outcome: LifeCycleValues | None = None) -> None:
        """
        Report executor termination for plan motions still in progress.

        :param outcome: The executor's failure outcome, or None to use native task
            verdicts.
        """
        for node, task in self.motion_mappings.items():
            if node.status not in (LifeCycleValues.RUNNING, LifeCycleValues.PAUSED):
                continue
            self._end_motion(
                node,
                (
                    outcome
                    if outcome is not None
                    else LifeCycleValues.verdict_for(
                        self.statechart.observation_state[task]
                    )
                ),
            )

    def _end_motion(self, node: MotionNode, outcome: LifeCycleValues) -> None:
        """
        Publish the terminal boundary of one native motion attempt.

        :param node: The motion whose attempt ended.
        :param outcome: The native terminal state to publish.
        """
        node.status = outcome
        node.end_time = datetime.now()
        node.plan.notify_node_ended(node)

    def stop(self) -> None:
        """
        Release the native history subscription.
        """
        self.statechart.history.remove_observer(self)


@dataclass
class Executable:
    """
    Base class for executable units.
    """

    execution_list: List[Executable] = field(default_factory=list)
    """
    List of executables that comprises this executable.
    """

    context: Context = field(kw_only=True)
    """
    Coraplex context which should be used to execute this executable.
    """

    @property
    def giskard_executables(self) -> List[GiskardExecutable]:
        """
        :return: The giskard executables this unit is made of, in execution order.
        """
        return [
            giskard_executable
            for executable in self.execution_list
            for giskard_executable in executable.giskard_executables
        ]

    def execute(self) -> None:
        """
        Executes the unit.
        """
        for executable in self.execution_list:
            executable.execute()


@dataclass
class GiskardExecutable(Executable):
    """
    Executable for everything that can be added to a motion state chart, this includes
    the motions and the pre- and postconditions.
    """

    root_node: Goal = field(kw_only=True)
    """
    The goal below which every motion of this executable lives.
    """

    motion_state_chart: MotionStatechart = field(
        default_factory=MotionStatechart, kw_only=True
    )
    """
    Giskard's motion state chart for this executable.

    It is created once and only ever extended, because a compiled chart can no longer
    grow: :meth:`~giskardpy.motion_statechart.motion_statechart.MotionStatechart.compile`
    binds its updaters to the state arrays that adding a node would replace.
    """

    motion_mappings: Dict[MotionNode, Task] = field(default_factory=dict, kw_only=True)
    """
    Mapping from the motion nodes of the plan to their giskard tasks, in execution
    order.
    """

    pre_condition_node: Optional[ConditionNode] = field(default=None, kw_only=True)
    """
    Optional pre-condition of the action this executable belongs to.

    Carried on the executable but not evaluated during execution at present, see
    :meth:`_add_condition_monitors`.
    """

    post_condition_node: Optional[ConditionNode] = field(default=None, kw_only=True)
    """
    Optional post-condition of the action this executable belongs to.

    Carried on the executable but not evaluated during execution at present, see
    :meth:`_add_condition_monitors`.
    """

    execution_type: ClassVar[Optional[ExecutionType]] = None
    """
    The execution type used for all giskard executables, managed by
    :py:class:`pycram.motion_executor.ExecutionEnvironment`.
    """

    collision_avoidance: ClassVar[bool] = False
    """
    Whether the robot avoids colliding with its surroundings and with itself, managed by
    :py:class:`pycram.motion_executor.ExecutionEnvironment`.

    Adds an
    :class:`~giskardpy.motion_statechart.goals.collision_avoidance.ExternalCollisionAvoidance`
    and a
    :class:`~giskardpy.motion_statechart.goals.collision_avoidance.SelfCollisionAvoidance`
    to the motion state chart.
    """

    @property
    def giskard_executables(self) -> List[GiskardExecutable]:
        """
        :return: This executable, which is the only giskard executable it is made of.
        """
        return [self]

    def prepare_for_execution(self) -> None:
        """
        Extend the motion state chart with the nodes that terminate it.

        This runs just before compilation rather than during parsing, because the
        execution type is only known once an
        :py:class:`~coraplex.execution_environment.ExecutionEnvironment` is entered.
        """
        end_trigger = self.root_node.goal_reached
        if GiskardExecutable.collision_avoidance:
            self.motion_state_chart.add_node(ExternalCollisionAvoidance())
            self.motion_state_chart.add_node(SelfCollisionAvoidance())

        end_motion = EndMotion()
        end_motion.start_condition = end_trigger
        self.motion_state_chart.add_node(end_motion)

    def _add_condition_monitors(self, end_trigger: Scalar) -> Scalar:
        """
        Add the pre- and post-condition nodes to the motion state chart and wire them to
        the root node and the end trigger of the motion state chart.

        The pre-condition gates the start of the motions, the post-condition gates the
        successful end of the motion, and a
        :class:`~giskardpy.motion_statechart.graph_node.CancelMotion` aborts the motion if
        either is observed to be false.

        .. note:: Currently unused. Conditions are kept out of the chart while evaluating
            them inside it is being reworked; this stays so they can be wired back in.

        :param end_trigger: The trigger which ends the motion state chart.
        :return: The end trigger, gated by the post-condition when there is one.
        """
        from coraplex.plans.condition_nodes import condition_monitor

        if self.pre_condition_node is not None and self.context.evaluate_conditions:
            pre_monitor = condition_monitor(self.pre_condition_node)
            self.motion_state_chart.add_node(pre_monitor)
            # only start the motion once the pre-condition holds
            self.root_node.start_condition = pre_monitor.observation_variable
            # abort if the pre-condition is observed to be false
            pre_cancel = CancelMotion(
                exception=self._condition_not_satisfied(
                    self.pre_condition_node,
                    action_node=self.pre_condition_node.action_node.action,
                )
            )
            pre_cancel.start_condition = trinary_logic_not(
                pre_monitor.observation_variable
            )
            self.motion_state_chart.add_node(pre_cancel)

        if self.post_condition_node is not None and self.context.evaluate_conditions:
            post_monitor = condition_monitor(self.post_condition_node)
            # only evaluate the post-condition once the motion is done
            post_monitor.start_condition = end_trigger
            self.motion_state_chart.add_node(post_monitor)
            end_trigger = post_monitor.observation_variable
            # abort if the post-condition is observed to be false
            post_cancel = CancelMotion(
                exception=self._condition_not_satisfied(
                    self.post_condition_node,
                    action_node=self.post_condition_node.action_node.action,
                )
            )
            post_cancel.start_condition = trinary_logic_not(
                post_monitor.observation_variable
            )
            self.motion_state_chart.add_node(post_cancel)
        return end_trigger

    @staticmethod
    def _condition_not_satisfied(
        condition_node: ConditionNode,
        action_node: ActionDescription,
    ) -> ConditionNotSatisfied:
        return ConditionNotSatisfied(
            pre_condition=condition_node.pre_condition,
            action=action_node.__class__,
            condition=condition_node.condition,
        )

    def execute(self) -> None:
        """
        Completes the motion state chart and executes it according to the execution
        type.
        """
        if len(self.motion_mappings) == 0:
            return
        if GiskardExecutable.execution_type == ExecutionType.NO_EXECUTION:
            return
        self.prepare_for_execution()

        match GiskardExecutable.execution_type:
            case ExecutionType.SIMULATED:
                self._execute_simulation()
            case ExecutionType.REAL:
                self._execute_real()
            case _:
                raise UnknownExecutionType(GiskardExecutable.execution_type)

    def _execute_simulation(self) -> None:
        """
        Execute the native chart while projecting its recorded motion states.
        """
        executor = Ros2Executor(
            context=MotionStatechartContext(
                world=self.context.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=50, prediction_horizon=4, verbose=False
                ),
            ),
            ros_node=self.context.ros_node,
        )
        with ExitStack() as cleanup:
            history = MotionPlanHistory(self.motion_state_chart, self.motion_mappings)
            cleanup.callback(history.stop)
            cleanup.callback(executor.context.cleanup)
            cleanup.callback(
                self.motion_state_chart.cleanup_nodes, context=executor.context
            )
            cleanup.callback(executor.set_velocity_acceleration_jerk_to_zero)
            try:
                executor.compile(self.motion_state_chart)
                for _ in range(
                    len(self.motion_mappings) * self.context.ticks_per_motion
                ):
                    executor.tick()
                    if executor.motion_statechart.is_end_motion():
                        history.end_active_motions()
                        return
                unfinished_nodes = [
                    node
                    for node in self.motion_state_chart.nodes
                    if node.life_cycle_state
                    not in [LifeCycleValues.SUCCEEDED, LifeCycleValues.NOT_STARTED]
                ]
                raise MotionDidNotFinish(unfinished_nodes)
            except BaseException as error:
                history.end_active_motions(
                    LifeCycleValues.FAILED
                    if isinstance(error, Exception)
                    else LifeCycleValues.INTERRUPTED
                )
                raise

    def _execute_real(self) -> None:
        """
        Executes the motion state chart on the real robot via giskard while monitoring
        for interrupts.
        """
        self.context.giskard_wrapper.execute(self.motion_state_chart)


@dataclass
class ConditionExecutable(Executable):
    """
    An executable unit for a condition node.
    """

    condition_node: ConditionNode = field(kw_only=True)
    """
    The condition node to execute.
    """

    def execute(self) -> None:
        """
        Executes the condition node.
        """
        if evaluate_condition(self.condition_node.condition):
            return True
        raise ConditionNotSatisfied(
            pre_condition=self.condition_node.pre_condition,
            action=self.condition_node.__class__,
            condition=self.condition_node.condition,
        )


@dataclass
class MoveBranchExecutable(Executable):
    """
    Executable that moves a body under a new parent, keeping the body's own connection
    so an actively driven body stays drivable afterwards.
    """

    body: Body = field(kw_only=True)
    """
    The root of the branch in the kinematic structure that is moved.
    """

    new_parent: Body = field(kw_only=True)
    """
    The new parent to which the branch is moved.
    """

    execution_scope: Callable[[], AbstractContextManager[None]] = field(
        default=nullcontext, kw_only=True, repr=False, compare=False
    )

    def execute(self) -> None:
        """
        Move the branch and report the attached node's execution outcome.
        """
        with self.execution_scope():
            self.context.world.move_branch(self.body, self.new_parent)


@dataclass
class UnderspecifiedExecutable(Executable):
    """
    Executable for an underspecified node whose resolution is deferred to execution
    time.

    Because it is not a :class:`GiskardExecutable`, it acts as a boundary in the
    execution list: every preceding executable runs (and mutates the world) before it
    is reached. Only then is the underspecified statement grounded, so the query sees
    the correct world state (e.g. the torso already raised, the object already in the
    gripper). Candidates are tried in order until one executes without raising a
    :class:`~pycram.plans.failures.PlanFailure`; if the generator is exhausted,
    :class:`~pycram.plans.failures.EmptyUnderspecified` is raised.
    """

    node: UnderspecifiedNode = field(kw_only=True)
    """
    The underspecified node that is grounded when this executable is reached.
    """

    def execute(self) -> None:
        from coraplex.plans.failures import PlanFailure, EmptyUnderspecified

        while self.node.advance():
            try:
                self.node.current_candidate.parse().execute()
                self.node.stop_grounding()
                return
            except PlanFailure:
                continue
        raise EmptyUnderspecified()

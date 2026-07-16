from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta

from typing_extensions import List, Dict, ClassVar, Optional, TYPE_CHECKING

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion, Task
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.ros_executor import Ros2Executor
from krrood.entity_query_language.factories import evaluate_condition
from coraplex.datastructures.enums import ExecutionType
from coraplex.exceptions import (
    MotionDidNotFinish,
    ConditionNotSatisfied,
    UnknownExecutionType,
)
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

from giskardpy.motion_statechart.graph_node import CancelMotion
from krrood.symbolic_math.symbolic_math import (
    trinary_logic_and,
    trinary_logic_not,
    trinary_logic_or,
)

if TYPE_CHECKING:
    from giskardpy.middleware.ros2.python_interface import GiskardWrapper
    from coraplex.robot_plans.actions.base import ActionDescription

    from coraplex.plans.condition_nodes import ConditionNode
    from coraplex.plans.plan_node import MotionNode, UnderspecifiedNode, ActionNode
    from coraplex.datastructures.dataclasses import Context

logger = logging.getLogger(__name__)


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

    synchronize_time_delta: timedelta = field(
        default=timedelta(seconds=1), kw_only=True
    )
    """
    Time delta that is waited between executables when executing on the real robot.

    Is done to prevent synchronization issues
    """

    def execute(self) -> None:
        """
        Executes the unit.
        """
        for executable in self.execution_list:
            if GiskardExecutable.execution_type == ExecutionType.REAL:
                time.sleep(self.synchronize_time_delta.seconds)
            executable.execute()


@dataclass
class GiskardExecutable(Executable):
    """
    Executable for everything that can be added to a Motion state chart, this includes
    the motions, pre -and postconditions and the pause and interrupt calls.
    """

    motion_mappings: Dict[MotionNode, Task] = field(kw_only=True)
    """
    Mapping from the motion nodes of the plan to their giskard tasks, in execution
    order.
    """

    pre_condition_node: Optional[ConditionNode] = field(default=None, kw_only=True)
    """
    Optional pre-condition of the action this executable belongs to.

    If set, the motion only starts once the condition is observed to hold and the motion
    is aborted (with :class:`ConditionNotSatisfied`) if it does not.
    """

    post_condition_node: Optional[ConditionNode] = field(default=None, kw_only=True)
    """
    Optional post-condition of the action this executable belongs to.

    If set, it is evaluated after the motion finished; the motion only ends successfully
    if the condition is observed to hold, otherwise it is aborted.
    """

    execution_type: ClassVar[Optional[ExecutionType]] = None
    """
    The execution type used for all giskard executables, managed by
    :py:class:`pycram.motion_executor.ExecutionEnvironment`.
    """

    collision_avoidance: ClassVar[bool] = False
    """
    Whether an :class:`~giskardpy.motion_statechart.goals.collision_avoidance.ExternalCo
    llisionAvoidance` is added to the motion state chart, managed by
    :py:class:`pycram.motion_executor.ExecutionEnvironment`.
    """

    _current_motion_state_chart: MotionStatechart = field(init=False, default=None)
    """
    Currently build motion state chart, internal only for managing the building the msc.
    """

    @property
    def motion_state_chart(self) -> MotionStatechart:
        """
        Giskard's motion state chart constructed from the motions of this executable.

        If a pre- and/or post-condition is set, it is added as a
        :class:`~giskardpy.motion_statechart.monitors.payload_monitors.ThreadedPredicateMonitor`
        and wired into the chart:

        - the pre-condition gates the start of the motion sequence,
        - the post-condition gates the successful end of the motion,
        - a :class:`~giskardpy.motion_statechart.graph_node.CancelMotion` aborts
          the motion if either condition is observed to be false.
        """
        self._current_motion_state_chart = MotionStatechart()
        if self.execution_type == ExecutionType.REAL:
            self._current_motion_state_chart.add_node(
                seq := Sequence(list(self.motion_mappings.values()))
            )
            self._current_motion_state_chart.add_node(EndMotion.when_true(seq))
            return self._current_motion_state_chart

        tasks = list(self.motion_mappings.values())
        for task in tasks:
            self._current_motion_state_chart.add_node(task)
        first_task = tasks[0]

        end_trigger = tasks[-1].observation_variable

        if self.execution_type == ExecutionType.SIMULATED:
            skip_end_conditions = self._add_pause_interrupt(tasks)

            # The motion is done when the last task finished or the first skipped
            # (interrupted) task is reached.
            if skip_end_conditions:
                end_trigger = trinary_logic_or(end_trigger, *skip_end_conditions)

            self._add_condition_monitors(first_task, end_trigger)
        if GiskardExecutable.collision_avoidance:
            self._current_motion_state_chart.add_node(ExternalCollisionAvoidance())

        end_motion = EndMotion()
        end_motion.start_condition = end_trigger
        self._current_motion_state_chart.add_node(end_motion)
        return self._current_motion_state_chart

    def _add_condition_monitors(
        self, first_task: Task, end_trigger: ObservationStateValues
    ):
        """
        Adds the pre -and postcondition nodes to the Motion state chart and wires them
        to the first task and the end trigger of the motion state chart.

        :param end_trigger: The trigger which ends the motion state chart.
        """
        from coraplex.plans.condition_nodes import condition_monitor

        if self.pre_condition_node is not None and self.context.evaluate_conditions:
            pre_monitor = condition_monitor(self.pre_condition_node)
            self._current_motion_state_chart.add_node(pre_monitor)
            # only start the motion once the pre-condition holds
            first_task.start_condition = pre_monitor.observation_variable
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
            self._current_motion_state_chart.add_node(pre_cancel)

        if self.post_condition_node is not None and self.context.evaluate_conditions:
            post_monitor = condition_monitor(self.post_condition_node)
            # only evaluate the post-condition once the motion is done
            post_monitor.start_condition = end_trigger
            self._current_motion_state_chart.add_node(post_monitor)
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
            self._current_motion_state_chart.add_node(post_cancel)

    def _add_pause_interrupt(self, tasks: List[Task]) -> List[ObservationStateValues]:
        """
        Wire the tasks as an interruptible/pausable sequence.

        Each task carries two monitors bound to its originating plan node:

        - a pause monitor feeding the task's pause_condition, so the *active*
          motion is held (and later resumed) when its plan node is paused;
        - an interrupt monitor gating the *next* task's start. An interrupt lets
          the currently active motion finish but prevents the subsequent ones
          from starting ("finish active, skip rest"). When a not-yet-started task
          is reached while interrupted, the motion ends there.

        :param tasks: The list of tasks that are were added to the motion state chart
        :returns: List of skip conditions for the case if a task is interrupted
        """
        from coraplex.plans.condition_nodes import PlanNodeStatusMonitor

        skip_end_conditions = []
        plan_nodes = list(self.motion_mappings.keys())
        for index, (plan_node, task) in enumerate(zip(plan_nodes, tasks)):
            # a task is done once its own goal is observed (as giskard's Sequence does)
            task.end_condition = task.observation_variable

            pause_monitor = PlanNodeStatusMonitor(
                predicate=lambda node=plan_node: node.is_paused,
                name=f"paused#{index}",
            )
            self._current_motion_state_chart.add_node(pause_monitor)
            task.pause_condition = pause_monitor.observation_variable

            interrupt_monitor = PlanNodeStatusMonitor(
                predicate=lambda node=plan_node: node.is_interrupted,
                name=f"interrupted#{index}",
            )
            self._current_motion_state_chart.add_node(interrupt_monitor)
            if index > 0:
                previous_done = tasks[index - 1].observation_variable
                # start only once the previous motion finished and this one is not
                # interrupted ...
                task.start_condition = trinary_logic_and(
                    previous_done,
                    trinary_logic_not(interrupt_monitor.observation_variable),
                )
                # ... otherwise, if we reach it while interrupted, the sequence ends.
                skip_end_conditions.append(
                    trinary_logic_and(
                        previous_done, interrupt_monitor.observation_variable
                    )
                )
        return skip_end_conditions

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

    @property
    def is_interrupted(self) -> bool:
        return any(node.is_interrupted for node in self.motion_mappings)

    @property
    def is_paused(self) -> bool:
        return any(node.is_paused for node in self.motion_mappings)

    def execute(self) -> None:
        """
        Builds the motion state chart from the motions and executes it according to the
        execution type.
        """
        if len(self.motion_mappings) == 0:
            return

        match GiskardExecutable.execution_type:
            case ExecutionType.SIMULATED:
                self._execute_simulation()
            case ExecutionType.REAL:
                self._execute_real()
            case ExecutionType.NO_EXECUTION:
                return
            case _:
                raise UnknownExecutionType(GiskardExecutable.execution_type)

    def _execute_simulation(self) -> None:
        """
        Compiles the motion state chart and ticks it in the world of the context until
        it is done.
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
        motion_state_chart = self.motion_state_chart
        executor.compile(motion_state_chart)

        counter = 0
        while counter < len(self.motion_mappings) * 2000:
            # Interrupting and pausing are handled inside the motion state chart by
            # per-task monitors (see motion_state_chart): an interrupt ends the
            # motion via EndMotion, a pause holds the active task via its
            # pause_condition. While paused we simply do not tick, so the pause does
            # not consume the tick budget.
            if self.is_paused:
                time.sleep(0.01)
                continue

            executor.tick()
            counter += 1
            if executor.motion_statechart.is_end_motion():
                break

        executor._set_velocity_acceleration_jerk_to_zero()
        executor.motion_statechart.cleanup_nodes(context=executor.context)
        executor.context.cleanup()

        if not executor.motion_statechart.is_end_motion():
            failed_nodes = [
                node
                for node in motion_state_chart.nodes
                if node.life_cycle_state
                not in [LifeCycleValues.DONE, LifeCycleValues.NOT_STARTED]
            ]
            logger.error(f"Failed Nodes: {failed_nodes}")
            raise MotionDidNotFinish(failed_nodes)

    def _execute_real(self) -> None:
        """
        Executes the motion state chart on the real robot via giskard while monitoring
        for interrupts.
        """
        from giskardpy.middleware.ros2.python_interface import GiskardWrapper

        giskard = GiskardWrapper(self.context.ros_node, world=self.context.world)

        giskard.execute(self.motion_state_chart)


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
class ModelChangeExecutable(Executable):
    """
    Executable that re-attaches a body to a new parent in the world model while keeping
    its current global pose.
    """

    body: Body = field(kw_only=True)
    """
    The body that is re-attached.
    """

    new_parent: Body = field(kw_only=True)
    """
    The body the moved body is attached to afterwards.
    """

    def execute(self) -> None:
        """
        Re-parent the body to ``new_parent`` while preserving its global pose.
        """
        obj_transform = self.context.world.compute_forward_kinematics(
            self.new_parent, self.body
        )
        with self.context.world.modify_world():
            self.context.world.remove_connection(self.body.parent_connection)
            # TODO: this shouldn't be fixed but 6DOF
            connection = FixedConnection(
                parent=self.new_parent,
                child=self.body,
                parent_T_connection_expression=obj_transform,
            )

            # connection = Connection6DoF.create_with_dofs(
            #     parent=self.new_parent, child=self.body, world=self.context.world, parent_T_connection_expression=obj_transform
            # )
            self.context.world.add_connection(connection)
            # connection.origin = obj_transform


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

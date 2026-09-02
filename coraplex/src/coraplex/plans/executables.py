from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta

from typing_extensions import List, Dict, ClassVar, Optional, TYPE_CHECKING

from coraplex.datastructures.enums import ExecutionType
from coraplex.exceptions import UnknownExecutionType
from coraplex.plans.failures import EmptyUnderspecified
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import (
    CancelMotion,
    EndMotion,
    MotionStatechartNode,
    Task,
)
from giskardpy.motion_statechart.monitors.payload_monitors import CountControlCycles
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.ros_executor import Ros2Executor
from krrood.entity_query_language.factories import evaluate_condition
from krrood.symbolic_math.symbolic_math import (
    trinary_logic_and,
    trinary_logic_not,
    trinary_logic_or,
)
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
)
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from giskardpy.middleware.ros2.python_interface import GiskardWrapper

    from coraplex.plans.condition_nodes import ConditionNode
    from coraplex.plans.plan_node import MotionNode, UnderspecifiedNode
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

    def execute(self) -> None:
        """
        Executes the unit.
        """
        for executable in self.execution_list:
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

    control_cycles_per_motion: ClassVar[int] = 2000
    """
    How many control cycles one motion may spend before it is considered stuck.
    """

    @property
    def control_cycle_budget(self) -> int:
        """
        :return: The control cycles all motions of this executable may spend together.
        """
        return len(self.motion_mappings) * self.control_cycles_per_motion

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
                exception=self.pre_condition_node.not_satisfied_failure()
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
                exception=self.post_condition_node.not_satisfied_failure()
            )
            post_cancel.start_condition = trinary_logic_not(
                post_monitor.observation_variable
            )
            self._current_motion_state_chart.add_node(post_cancel)

    def _add_motion_watchdogs(self) -> None:
        """
        Give every motion a watchdog that cancels the chart once the motion spent its
        control cycle budget without reaching its goal.

        The cancelling node carries a failure naming the motion node it watches, so the
        failure is attributed to the plan node whose motion did not finish instead of to
        whichever node happens to catch it. A watchdog only counts while its motion
        runs, so a stuck motion is reported as soon as it alone spent its budget, rather
        than once the whole chart ran out of time.

        The watchdogs are wired last, so they inherit the start conditions that the
        interrupt and pre-condition wiring put on their tasks.
        """
        for index, (motion_node, task) in enumerate(self.motion_mappings.items()):
            watchdog = CountControlCycles(
                control_cycles=self.control_cycles_per_motion,
                name=f"budget_spent#{index}",
            )
            self._current_motion_state_chart.add_node(watchdog)
            watchdog.start_condition = task.start_condition
            watchdog.end_condition = task.observation_variable

            cancel = CancelMotion(
                exception=motion_node.did_not_finish_failure([task]),
                name=f"did_not_finish#{index}",
            )
            cancel.start_condition = trinary_logic_and(
                watchdog.observation_variable,
                trinary_logic_not(task.observation_variable),
            )
            self._current_motion_state_chart.add_node(cancel)

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

    def owning_motion_node(self, statechart_node: MotionStatechartNode) -> MotionNode:
        """
        Resolve a motion state chart node to the motion of the plan that produced it.

        Nodes added while a goal expands are attributed to the motion that owns the
        goal, so the walk goes up to the top level before the mapping is consulted.
        Nodes that no motion produced, for example monitors and collision avoidance,
        fall back to the first motion of this executable.

        :param statechart_node: The node of the motion state chart to attribute.
        :return: The motion node the given node belongs to.
        """
        tasks_to_motion_nodes = {
            task: motion_node for motion_node, task in self.motion_mappings.items()
        }
        top_level_node = statechart_node
        while top_level_node.parent_node is not None:
            top_level_node = top_level_node.parent_node
        return tasks_to_motion_nodes.get(
            top_level_node, next(iter(self.motion_mappings))
        )

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
        try:
            while counter < self.control_cycle_budget:
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
        finally:
            # Also runs when a CancelMotion node raises out of tick, which would
            # otherwise leave the controller and the world in the state of the aborted
            # motion.
            executor.set_velocity_acceleration_jerk_to_zero()
            executor.motion_statechart.cleanup_nodes(context=executor.context)
            executor.context.cleanup()

        if executor.motion_statechart.is_end_motion():
            return
        # TODO: Check if these really are failed tasks
        failed_nodes = [
            node
            for node in motion_state_chart.nodes
            if node.life_cycle_state
            not in [LifeCycleValues.DONE, LifeCycleValues.NOT_STARTED]
        ]
        logger.error(f"Failed Nodes: {failed_nodes}")
        unfinished_motion = (
            self.owning_motion_node(failed_nodes[0])
            if failed_nodes
            else next(iter(self.motion_mappings))
        )
        raise unfinished_motion.did_not_finish_failure(failed_nodes)

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
        raise self.condition_node.not_satisfied_failure()


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

    giskard_idle_settle_delta: timedelta = field(
        default=timedelta(seconds=0.3), kw_only=True
    )
    """
    Time to wait after publishing the model change on the real robot.

    Giskard only applies buffered world updates, and only republishes tf, while its
    behavior tree is idle between goals (tree tick period is 50ms); this delay gives it
    a few idle ticks to catch up before the next motion goal is sent, instead of relying
    on however much idle time happens to fall out of the surrounding plan's timing.
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
            connection = Connection6DoF.create_with_dofs(
                parent=self.new_parent,
                child=self.body,
                world=self.context.world,
                parent_T_connection_expression=obj_transform,
            )
            self.context.world.add_connection(connection)
        if GiskardExecutable.execution_type == ExecutionType.REAL:
            time.sleep(self.giskard_idle_settle_delta.total_seconds())


@dataclass
class UnderspecifiedExecutable(Executable):
    """
    Executable for an underspecified node whose resolution is deferred to execution
    time.

    Because it is not a :class:`GiskardExecutable`, it acts as a boundary in the
    execution list: every preceding executable runs (and mutates the world) before it
    is reached. Only then is the underspecified statement grounded, so the query sees
    the correct world state (e.g. the torso already raised, the object already in the
    gripper).

    One candidate is grounded and run per execution. A candidate's
    :class:`~coraplex.plans.failures.PlanFailure` is not swallowed here: it escalates
    along the plan tree to the underspecified node, which resolves it by running again
    and thereby advancing to the next candidate. Once the generator is exhausted,
    :class:`~coraplex.plans.failures.EmptyUnderspecified` is raised instead.
    """

    node: UnderspecifiedNode = field(kw_only=True)
    """
    The underspecified node that is grounded when this executable is reached.
    """

    def execute(self) -> None:
        if not self.node.advance():
            raise EmptyUnderspecified(node=self.node)
        self.node.current_candidate.parse().execute()
        self.node.stop_grounding()

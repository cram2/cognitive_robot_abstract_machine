from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta

from typing_extensions import List, Dict, ClassVar, Optional, TYPE_CHECKING

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
)
from giskardpy.motion_statechart.graph_node import CancelMotion
from giskardpy.motion_statechart.graph_node import EndMotion, Goal, Task
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
from krrood.symbolic_math.symbolic_math import Scalar, trinary_logic_not
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
    from coraplex.robot_plans.actions.base import ActionDescription

    from coraplex.plans.condition_nodes import ConditionNode
    from coraplex.plans.plan_node import MotionNode
    from coraplex.plans.underspecified import UnderspecifiedNode
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
    Whether an :class:`~giskardpy.motion_statechart.goals.collision_avoidance.ExternalCo
    llisionAvoidance` is added to the motion state chart, managed by
    :py:class:`pycram.motion_executor.ExecutionEnvironment`.
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
        end_trigger = self.root_node.observation_variable
        if GiskardExecutable.collision_avoidance:
            self.motion_state_chart.add_node(ExternalCollisionAvoidance())

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
            executor.tick()
            counter += 1
            if executor.motion_statechart.is_end_motion():
                break

        executor.set_velocity_acceleration_jerk_to_zero()
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

    def execute(self) -> None:
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

"""
The nodes of a motion statechart and the tasks built on them.

A node hands its constraints for the QP controller and its debug expressions over in its
artifacts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from functools import cached_property

import numpy as np
from typing_extensions import Self

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.forward_kinematics_binding import (
    ForwardKinematicsBinding,
)
from giskardpy.motion_statechart.constraint_builders import GeometricConstraintBuilder
from giskardpy.motion_statechart.context import MotionControlContext
from cramph.node import EndedByOwner
from cramph.context import StatechartContext
from giskardpy.motion_statechart.exceptions import MissingErrorSignalError
from giskardpy.motion_statechart.mixins import (
    HasGoalBindingPolicyParameter,
    HasThresholdParameter,
    HasWeightParameter,
)
from cramph.plotters.plot_specs import NodePlotSpec, plot_specification_field
from giskardpy.qp.constraint_collection import ConstraintCollection
from krrood.symbolic_math.symbolic_math import (
    FloatVariable,
    Scalar,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.spatial_types import (
    Point3,
    Vector3,
    Quaternion,
    RotationMatrix,
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

from cramph.node import NodeArtifacts, StatechartNode, EndStatechart
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain
from cramph.world_access import WorldAccess


@dataclass(kw_only=True)
class DebugExpression:
    """
    Symbolic expressions used for debugging only.

    Allows you to keep track of any expression and evaluate them later in debug mode.
    """

    name: str
    """
    Name used for this expression in some debugging tools.
    """

    expression: (
        Scalar
        | Point3
        | Vector3
        | Quaternion
        | RotationMatrix
        | HomogeneousTransformationMatrix
        | Pose
    )
    """
    The tracked expression; spatial types are additionally rendered as RViz markers.
    """

    color: Color = field(
        default_factory=lambda: Color.from_rgba(red=1, green=0, blue=0, opacity=1)
    )
    """
    The color used when this expression is rendered in visualization tools.
    """

    def __repr__(self) -> str:
        return self.name

    @property
    def evaluated(self) -> np.ndarray:
        """
        :return: The current value of the tracked expression.
        """
        return self.expression.evaluate()


@dataclass(kw_only=True)
class MotionNodeArtifacts(NodeArtifacts):
    """
    The artifacts of a node that takes part in motion control.
    """

    constraints: ConstraintCollection = field(default_factory=ConstraintCollection)
    """
    A collection of constraints that describe a motion task.
    """

    error: Scalar | None = field(default=None)
    """
    How far this node is from its goal.

    Set by :class:`ConvergingTask`, which derives :attr:`observation` from it, and used
    to watch whether the node is still converging.
    """

    debug_expressions: list[DebugExpression] = field(default_factory=list)
    """
    A list of symbolic expressions used for debugging only.

    While in debug mode, you can call .evaluate() on them to get their current value.
    """

    @classmethod
    def from_node_artifacts(cls, *, artifacts: NodeArtifacts) -> Self:
        """
        :param artifacts: The artifacts a node built.
        :return: `artifacts` itself if it already describes motion, otherwise motion
            artifacts with the same observation and nothing else.
        """
        if isinstance(artifacts, cls):
            return artifacts
        return cls(observation=artifacts.observation)

    @cached_property
    def geometry(self) -> GeometricConstraintBuilder:
        """
        Builder for high-level geometric constraints (point, vector, and rotation goals,
        and Cartesian velocity limits) that writes into :attr:`constraints`.

        :return: The builder for this collection of artifacts.
        """
        return GeometricConstraintBuilder(collection=self.constraints)


@dataclass(repr=False, eq=False, kw_only=True)
class MotionStatechartNode(StatechartNode):
    """
    A node of a motion statechart, which may contribute constraints, an error signal and
    debug expressions to motion control.
    """

    def create_structure_copy(self) -> MotionStatechartNode:
        return StructureCopyMotionStatechartNode(name=self.name)

    @property
    def artifacts(self) -> MotionNodeArtifacts:
        return super().artifacts

    def apply_artifacts(self, *, artifacts: NodeArtifacts) -> None:
        motion_artifacts = MotionNodeArtifacts.from_node_artifacts(artifacts=artifacts)
        motion_artifacts.constraints.link_to_motion_statechart_node(node=self)
        super().apply_artifacts(artifacts=motion_artifacts)

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Describe this node in terms of constraints, observation and debug expressions.

        :return: The artifacts describing this node. It is normal for nodes that don't
            directly affect the motion to return empty artifacts.
        """
        return MotionNodeArtifacts()

    def compute_world_access(self) -> WorldAccess:
        """
        :return: What :meth:`~cramph.node.StatechartNode.compute_world_access`
            returns, with the degrees of freedom of a connection with an active one that
            its constraints move, and the degrees of freedom its error reads or its
            constraints depend on but cannot move.
        """
        artifacts = self.artifacts
        in_constraints = frozenset(
            degree_of_freedom
            for constraint in [
                *artifacts.constraints.equality_constraints,
                *artifacts.constraints.inequality_constraints,
            ]
            for degree_of_freedom in WorldAccess.degrees_of_freedom_in_expression(
                expression=constraint.expression
            )
        )
        moved = in_constraints & frozenset(
            degree_of_freedom
            for connection in self.context.world.connections
            if connection.active_dofs
            for degree_of_freedom in connection.dofs
        )
        return (
            super()
            .compute_world_access()
            .union_with(
                other=WorldAccess(
                    moved_degrees_of_freedom=moved,
                    read_degrees_of_freedom=WorldAccess.degrees_of_freedom_in_expression(
                        expression=artifacts.error
                    )
                    | (in_constraints - moved),
                )
            )
        )

    @property
    def constraint_collection(self) -> ConstraintCollection:
        """
        :return: The constraints this node built.
        """
        return self.artifacts.constraints

    @property
    def debug_expressions(self) -> list[DebugExpression]:
        """
        :return: The debug expressions registered by this node during build.
        """
        return self.artifacts.debug_expressions

    @property
    def error_signal(self) -> Scalar | None:
        """
        :return: The error signal produced during build, if any.
        """
        return self.artifacts.error


def velocity_convergence_expression(
    context: StatechartContext,
    joint_convergence_threshold: float,
    minimum_threshold: float,
    maximum_threshold: float,
    degrees_of_freedom: list[DegreeOfFreedom] | None = None,
    minimum_time: float = 1.0,
    reference_cycle_variable: FloatVariable | None = None,
) -> Scalar:
    """
    Builds a trinary expression that is true once every given degree of freedom's
    velocity has dropped below a threshold derived from its own maximum velocity, and at
    least ``minimum_time`` simulated seconds of trajectory time have elapsed.

    :param context: Supplies the world's active degrees of freedom and control cycle
        timing.
    :param joint_convergence_threshold: Fraction of a degree of freedom's maximum
        velocity below which it is considered settled.
    :param minimum_threshold: Lower bound for the per-degree-of-freedom velocity
        threshold.
    :param maximum_threshold: Upper bound for the per-degree-of-freedom velocity
        threshold.
    :param degrees_of_freedom: Degrees of freedom to check for convergence. Defaults to
        every active degree of freedom in the world when ``None``. Those without an
        upper velocity limit are skipped, since no threshold can be derived for them.
    :param minimum_time: Minimum elapsed control time before the expression can become
        true.
    :param reference_cycle_variable: Cycle count elapsed time is measured from, instead
        of the start of the whole motion chart. Pass a variable a caller updates in its
        own ``on_start`` so ``minimum_time`` gates on how long that caller has been
        active, not on how many cycles the entire chart has already ticked through.
        ``None`` keeps the chart-wide behaviour.
    :return: A trinary :class:`~krrood.symbolic_math.symbolic_math.Scalar` expression,
        true once the given degrees of freedom have settled.
    """
    degrees_of_freedom = (
        degrees_of_freedom
        if degrees_of_freedom is not None
        else context.world.active_degrees_of_freedom
    )
    ref = []
    symbols = []
    for dof in degrees_of_freedom:
        if dof.limits.upper.velocity is None:
            # nothing to derive a threshold from, so this degree of freedom cannot
            # converge by this measure; environment joints are routinely parsed
            # without a velocity limit
            continue
        velocity_limit = dof.limits.upper.velocity * joint_convergence_threshold
        velocity_limit = min(max(minimum_threshold, velocity_limit), maximum_threshold)
        ref.append(velocity_limit)
        symbols.append(dof.variables.velocity)

    time_step = context.require_extension(
        extension_type=MotionControlContext
    ).qp_controller_config.control_time_step.total_seconds()
    elapsed_cycles = context.tick_variable
    if reference_cycle_variable is not None:
        elapsed_cycles = elapsed_cycles - reference_cycle_variable
    trajectory_longer_than_minimum_time = elapsed_cycles * time_step > minimum_time
    return sm.trinary_logic_and(
        trajectory_longer_than_minimum_time,
        sm.logic_all(sm.abs(sm.Vector(symbols)) < sm.Vector(ref)),
    )


@dataclass(eq=False, repr=False, kw_only=True)
class StructureCopyMotionStatechartNode(EndedByOwner, MotionStatechartNode):
    """
    Stands in for a motion statechart node in a structure copy, see
    :meth:`~cramph.statechart.Statechart.create_structure_copy`.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class Task(EndedByOwner, MotionStatechartNode, HasWeightParameter):
    """
    Tasks are MotionStatechartNodes that add motion constraints.

    A task stops holding what it reached the moment it is released, so its owner decides
    when it succeeded.
    """

    plot_specifications: NodePlotSpec = plot_specification_field(
        NodePlotSpec.create_task_style
    )

    def create_structure_copy(self) -> Task:
        return Task(name=self.name)


@dataclass(eq=False, repr=False, kw_only=True)
class ConvergingTask(Task, HasThresholdParameter, ABC):
    """
    A task that drives a single scalar error towards zero and counts as having reached
    its goal once that error is within :attr:`threshold`.

    Reaching the goal is not by itself a reason to end: the same task is a milestone in
    a sequence and an invariant to hold inside a goal that grasps something. Whatever
    ends it reads what the task observes to decide whether it succeeded.

    Subclasses declare the error rather than the observation, so that "reached the goal"
    is defined in one place, and so that how fast the goal is being approached can be
    measured. Tasks that enforce an invariant instead of converging, such as a velocity
    limit or a collision predicate, are plain :class:`Task` and write their own
    observation.
    """

    def build(self) -> MotionNodeArtifacts:
        """
        Build the task and derive its observation from its error.

        Being within :attr:`threshold` is what this task observes about the world, and
        succeeding means reaching it, which is what any node is judged by.
        """
        artifacts = super().build()
        if not isinstance(artifacts, MotionNodeArtifacts) or artifacts.error is None:
            raise MissingErrorSignalError(node=self)
        artifacts.observation = self.goal_reached_at(error=artifacts.error)
        return artifacts

    def goal_reached_at(self, *, error: Scalar) -> Scalar:
        """
        :param error: An error of this task, in its own units.
        :return: Whether this task observes its goal as reached at that error.
        """
        return error <= self.threshold

    @abstractmethod
    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Add the motion constraints of this task and set
        :attr:`MotionNodeArtifacts.error` to the error they drive to zero.

        :return: The artifacts describing this task.
        """

    @property
    def error_signal(self) -> Scalar:
        """
        :return: The error signal produced during build.
        """
        error_signal = super().error_signal
        if error_signal is None:
            raise MissingErrorSignalError(node=self)
        return error_signal

    @property
    def normalized_error(self) -> Scalar:
        """
        The error divided by :attr:`threshold`, so that a value of at most 1 means the
        goal is reached.

        Dividing out the threshold makes errors of different tasks, and of different
        units, comparable against a single convergence rate.

        :return: The threshold relative error of this task.
        """
        return self.error_signal / self.threshold


# %% nodes reading where their goal lies


@dataclass(eq=False, repr=False, kw_only=True)
class BindsGoalReferenceFrame(StatechartNode, HasGoalBindingPolicyParameter, ABC):
    """
    A node whose goal is given in a frame that may move, and which reads where that
    frame lies as :attr:`goal_binding_policy` says.
    """

    root_T_goal_reference_frame: HomogeneousTransformationMatrix | None = field(
        default=None,
        init=False,
        repr=False,
    )
    """
    The pose of :attr:`goal_reference_frame` relative to :attr:`goal_binding_root`, as
    :attr:`goal_binding_policy` fixes it; None until this node is set up.
    """

    _goal_reference_frame_binding: ForwardKinematicsBinding = field(
        init=False, repr=False
    )
    """
    Fixes :attr:`root_T_goal_reference_frame` to the current pose of
    :attr:`goal_reference_frame`.
    """

    _goal_binding_world_access: WorldAccess = field(
        default_factory=WorldAccess, init=False, repr=False
    )
    """
    The degrees of freedom and frames the latest binding of the goal reference frame
    read.
    """

    @property
    @abstractmethod
    def goal_reference_frame(self) -> KinematicStructureEntity:
        """
        :return: The frame the goal of this node is given in.
        """

    @property
    @abstractmethod
    def goal_binding_root(self) -> KinematicStructureEntity:
        """
        :return: The frame the pose of :attr:`goal_reference_frame` is read relative to.
        """

    @property
    def _goal_reference_frame_name(self) -> PrefixedName:
        """
        :return: The name the variables of :attr:`root_T_goal_reference_frame` are
            created under.
        """
        return PrefixedName(name="root_T_goal_ref", prefix=str(self.name))

    def _root_T_goal_reference_frame_to_bind(
        self,
    ) -> HomogeneousTransformationMatrix | None:
        """
        :return: The matrix :meth:`set_up` binds, or None to bind one over new
            variables.
        """
        return None

    def set_up(self) -> None:
        """
        Bind :attr:`root_T_goal_reference_frame`.
        """
        super().set_up()
        self._goal_reference_frame_binding = ForwardKinematicsBinding(
            name=self._goal_reference_frame_name,
            kinematic_chain=KinematicChain(
                root=self.goal_binding_root, tip=self.goal_reference_frame
            ),
            float_variable_data=self.context.float_variable_data,
            root_T_tip=self._root_T_goal_reference_frame_to_bind(),
        )
        self.root_T_goal_reference_frame = self._goal_reference_frame_binding.root_T_tip
        self._bind_goal_reference_frame()

    def on_start(self) -> None:
        """
        Differs from the inherited ``on_start`` in also binding the goal reference frame
        again under :attr:`GoalBindingPolicy.Bind_on_start`, so the goal is read from
        where that frame is each time the node starts.
        """
        super().on_start()
        if self.goal_binding_policy == GoalBindingPolicy.Bind_on_start:
            self._bind_goal_reference_frame()

    def _bind_goal_reference_frame(self) -> None:
        """
        Fix :attr:`root_T_goal_reference_frame` to where the goal reference frame lies
        now, and remember the degrees of freedom and frames between
        :attr:`goal_binding_root` and :attr:`goal_reference_frame` that this read.
        """
        self._goal_reference_frame_binding.bind(world=self.context.world)
        root_chain, tip_chain = self.context.world.compute_split_chain_of_connections(
            self.goal_binding_root, self.goal_reference_frame
        )
        chain = [*root_chain, *tip_chain]
        self._goal_binding_world_access = WorldAccess(
            read_degrees_of_freedom=frozenset(
                degree_of_freedom
                for connection in chain
                for degree_of_freedom in connection.dofs
            ),
            read_frames=frozenset(
                frame
                for connection in chain
                for frame in (connection.parent, connection.child)
            ),
        )

    def compute_world_access(self) -> WorldAccess:
        """
        :return: What :meth:`~cramph.node.StatechartNode.compute_world_access`
            returns, with the degrees of freedom and frames the latest binding of the
            goal reference frame read.
        """
        return (
            super()
            .compute_world_access()
            .union_with(other=self._goal_binding_world_access)
        )


@dataclass(eq=False, repr=False, kw_only=True)
class EndMotion(EndStatechart):
    """
    Ends the motion once the world has settled.
    """

    joint_convergence_threshold: float = 0.01
    """
    Fraction of a degree of freedom's maximum velocity below which it is considered
    settled.

    Only used while at least one active degree of freedom exists; see :meth:`build`.
    """

    minimum_threshold: float = 0.01
    """
    Lower bound for the per-degree-of-freedom velocity threshold.
    """

    maximum_threshold: float = 0.06
    """
    Upper bound for the per-degree-of-freedom velocity threshold.
    """

    def create_structure_copy(self) -> EndMotion:
        return EndMotion(name=self.name)

    def build_artifacts(self) -> NodeArtifacts:
        """
        Reports "done" only once the world has actually settled, so the motion isn't cut
        short while the controller is still commanding nonzero velocity.

        .. note:: If the world has no active degrees of freedom, there is nothing to
            converge, so this reports done immediately once running, same as before.
        """
        if not self.context.world.active_degrees_of_freedom:
            return super().build_artifacts()
        observation = velocity_convergence_expression(
            context=self.context,
            joint_convergence_threshold=self.joint_convergence_threshold,
            minimum_threshold=self.minimum_threshold,
            maximum_threshold=self.maximum_threshold,
        )
        return NodeArtifacts(observation=observation)

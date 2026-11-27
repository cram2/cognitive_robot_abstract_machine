"""
Tasks turning a plane of a link onto a goal plane.
"""

from dataclasses import dataclass, field

from giskardpy.motion_statechart.forward_kinematics_binding import (
    ForwardKinematicsBinding,
)
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.graph_node import (
    ConvergingTask,
    MotionNodeArtifacts,
    DebugExpression,
)
from giskardpy.motion_statechart.mixins import (
    HasCruiseAngularVelocityParameter,
    HasKinematicChainParameter,
    HasReferenceNormalParameter,
    HasRootLinkParameter,
    HasTipNormalParameter,
)
from krrood_spatial_math.spatial_types import Vector3Dimensional
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain

# %% aligning a plane of a link


@dataclass(eq=False, repr=False, kw_only=True)
class AlignPlanes(
    ConvergingTask,
    HasKinematicChainParameter,
    HasTipNormalParameter,
    HasReferenceNormalParameter,
    HasCruiseAngularVelocityParameter,
):
    """
    Will orient the plane fixed on the tip of :attr:`kinematic_chain` to align with the
    plane fixed on its root, with an error in radians.

    The planes are represented by their normal vectors, :attr:`tip_normal` and
    :attr:`reference_normal`.
    """

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """
    Above collision avoidance by default.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build motion constraints that rotate the tip plane onto the goal plane.

        :return: The artifacts of this task, whose error is the angle between the two
            plane normals.
        """
        artifacts = MotionNodeArtifacts()
        tip_V_tip_normal = self.express_tip_normal_in_tip()
        tip_V_tip_normal.scale(1)
        root_V_root_normal = self.context.world.transform(
            target_frame=self.kinematic_chain.root, spatial_object=self.reference_normal
        )
        root_V_root_normal.scale(1)

        root_R_tip = self.context.world.compose_forward_kinematics_expression(
            self.kinematic_chain.root, self.kinematic_chain.tip
        ).rotation
        root_V_tip_normal = root_R_tip @ tip_V_tip_normal
        root_V_tip_normal.scale(1)

        root_V_tip_normal.visualisation_frame = self.kinematic_chain.tip
        current_normal_debug_expression = DebugExpression(
            name=f"{self.name}/current_normal",
            expression=root_V_tip_normal,
            color=Color.from_rgba(red=1, green=0, blue=0, opacity=1),
        )
        # a copy, as transforming returns the reference normal itself when it is in the root
        root_V_reference_normal_at_tip = Vector3Dimensional.from_iterable(
            root_V_root_normal
        )
        root_V_reference_normal_at_tip.visualisation_frame = self.kinematic_chain.tip
        reference_normal_debug_expression = DebugExpression(
            name=f"{self.name}/reference_normal",
            expression=root_V_reference_normal_at_tip,
            color=Color.from_rgba(red=0, green=0, blue=1, opacity=1),
        )
        artifacts.debug_expressions.append(current_normal_debug_expression)
        artifacts.debug_expressions.append(reference_normal_debug_expression)

        artifacts.geometry.add_vector_goal_constraints(
            frame_V_current=root_V_tip_normal,
            frame_V_goal=root_V_root_normal,
            cruise_velocity=self.cruise_angular_velocity,
            quadratic_weight=self.weight,
        )
        artifacts.error = root_V_tip_normal.angle_between(root_V_root_normal)
        return artifacts

    def express_tip_normal_in_tip(self) -> Vector3Dimensional:
        """
        :return: The normal of the tip plane, expressed in the tip of
            :attr:`kinematic_chain`.
        """
        return self.context.world.transform(
            target_frame=self.kinematic_chain.tip, spatial_object=self.tip_normal
        )


# %% keeping what a gripper holds upright


@dataclass(eq=False, repr=False, kw_only=True)
class KeepHeldBodyUpright(AlignPlanes, HasRootLinkParameter):
    """
    Turns the body an end effector holds when this task starts until its z-axis points
    along the z-axis of :attr:`root_link`.

    An end effector with no body below its tool frame, as when contact alone carries the
    body, keeps the direction in its tool frame that points up when this task starts.
    """

    end_effector: EndEffector
    """
    The end effector holding the body.
    """

    kinematic_chain: KinematicChain = field(init=False)
    """
    The chain from :attr:`root_link` to the tool frame of :attr:`end_effector`.
    """

    reference_normal: Vector3Dimensional = field(init=False)
    """
    The z-axis of :attr:`root_link`.
    """

    tip_normal: Vector3Dimensional = field(init=False)
    """
    The z-axis of the held body, before its pose in the tool frame is applied.
    """

    _tool_T_held_body: ForwardKinematicsBinding = field(init=False, repr=False)
    """
    The pose of the held body in the tool frame, bound when this task starts.
    """

    def set_up(self) -> None:
        """
        Expresses the plane to align over the pose this task binds when it starts.
        """
        super().set_up()
        tool_frame = self.end_effector.tool_frame
        self.kinematic_chain = KinematicChain(root=self.root_link, tip=tool_frame)
        self.reference_normal = Vector3Dimensional.Z(reference_frame=self.root_link)
        self._tool_T_held_body = ForwardKinematicsBinding(
            name=PrefixedName("tool_T_held_body", str(self.name)),
            kinematic_chain=KinematicChain(root=tool_frame, tip=tool_frame),
            float_variable_data=self.context.float_variable_data,
        )
        self._tool_T_held_body.bind(world=self.context.world)
        self.tip_normal = Vector3Dimensional.Z(reference_frame=tool_frame)

    def express_tip_normal_in_tip(self) -> Vector3Dimensional:
        """
        Unlike :meth:`AlignPlanes.express_tip_normal_in_tip`, applies the pose of the
        held body in the tool frame, bound when this task starts.
        """
        return self._tool_T_held_body.root_T_tip @ self.tip_normal

    def on_start(self) -> None:
        """
        Binds the pose of the body held now, or of :attr:`root_link` if none is.
        """
        super().on_start()
        held_body = self.end_effector.held_body
        self._tool_T_held_body.change_tip(
            tip=self.root_link if held_body is None else held_body,
            world=self.context.world,
        )

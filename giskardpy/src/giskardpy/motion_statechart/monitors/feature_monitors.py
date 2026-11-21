"""
Monitors that observe a geometric relation between a feature on the tip of a kinematic
chain and one on its root, such as a height, a distance or an angle.
"""

from __future__ import annotations

from dataclasses import field, dataclass

import krrood.symbolic_math.symbolic_math as sm
from cramph.node import EndedByOwner
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.mixins import (
    HasKinematicChainParameter,
    HasLowerAngleParameter,
    HasLowerLimitParameter,
    HasReferenceNormalParameter,
    HasReferencePointParameter,
    HasReferenceVectorParameter,
    HasThresholdParameter,
    HasTipNormalParameter,
    HasTipPointParameter,
    HasTipVectorParameter,
    HasUpperAngleParameter,
    HasUpperLimitParameter,
)
from cramph.node import NodeArtifacts
from krrood_spatial_math.spatial_types import Point3Dimensional, Vector3Dimensional

# %% monitors on geometric features


@dataclass(eq=False, repr=False, kw_only=True)
class FeatureFunctionMonitor(
    EndedByOwner, MotionStatechartNode, HasKinematicChainParameter
):
    """
    Base for monitors that compare a controlled feature, attached to the tip of
    :attr:`kinematic_chain`, with a reference feature attached to its root, expressed in
    the frame of the root.
    """

    reference_feature: Point3Dimensional | Vector3Dimensional = field(init=False)
    """
    Reference feature, typically the target for `controlled_feature`.
    """

    controlled_feature: Point3Dimensional | Vector3Dimensional = field(init=False)
    """
    Controlled feature, typically moved towards `reference_feature`.
    """

    def build_artifacts(self) -> NodeArtifacts:
        """
        Transform the controlled and reference features into the root link frame and
        store them on the node for use by subclasses.
        """
        root_reference_feature = self.context.world.transform(
            target_frame=self.kinematic_chain.root,
            spatial_object=self.reference_feature,
        )
        tip_controlled_feature = self.context.world.transform(
            target_frame=self.kinematic_chain.tip,
            spatial_object=self.controlled_feature,
        )

        root_T_tip = self.context.world.compose_forward_kinematics_expression(
            self.kinematic_chain.root, self.kinematic_chain.tip
        )
        if isinstance(self.controlled_feature, Point3Dimensional):
            self.root_P_controlled_feature = root_T_tip @ tip_controlled_feature
        elif isinstance(self.controlled_feature, Vector3Dimensional):
            self.root_V_controlled_feature = root_T_tip @ tip_controlled_feature

        if isinstance(self.reference_feature, Point3Dimensional):
            self.root_P_reference_feature = root_reference_feature
        if isinstance(self.reference_feature, Vector3Dimensional):
            self.root_V_reference_feature = root_reference_feature

        return NodeArtifacts()


@dataclass(eq=False, repr=False, kw_only=True)
class HeightMonitor(
    FeatureFunctionMonitor,
    HasTipPointParameter,
    HasReferencePointParameter,
    HasLowerLimitParameter,
    HasUpperLimitParameter,
):
    """
    Observes ``True`` while the height difference between the controlled and reference
    point along the root z-axis is within ``[lower_limit, upper_limit]``.
    """

    def build_artifacts(self) -> NodeArtifacts:
        self.reference_feature = self.reference_point
        self.controlled_feature = self.tip_point
        artifacts = super().build_artifacts()

        distance = (
            self.root_P_controlled_feature - self.root_P_reference_feature
        ) @ Vector3Dimensional.Z()
        artifacts.observation = sm.logic_and(
            distance >= self.lower_limit,
            distance <= self.upper_limit,
        )
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class PerpendicularMonitor(
    FeatureFunctionMonitor,
    HasTipNormalParameter,
    HasReferenceNormalParameter,
    HasThresholdParameter,
):
    """
    Observes ``True`` while the controlled normal is perpendicular to the reference
    normal within ``threshold``, which bounds the dot product of the two normals.
    """

    def build_artifacts(self) -> NodeArtifacts:
        self.reference_feature = self.reference_normal
        self.controlled_feature = self.tip_normal
        artifacts = super().build_artifacts()

        expr = self.root_V_reference_feature[:3] @ self.root_V_controlled_feature[:3]
        artifacts.observation = sm.abs(expr) <= self.threshold
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class DistanceMonitor(
    FeatureFunctionMonitor,
    HasTipPointParameter,
    HasReferencePointParameter,
    HasLowerLimitParameter,
    HasUpperLimitParameter,
):
    """
    Observes ``True`` while the planar distance (ignoring the z-axis) between the
    controlled and reference point is within ``[lower_limit, upper_limit]``.
    """

    def build_artifacts(self) -> NodeArtifacts:
        self.reference_feature = self.reference_point
        self.controlled_feature = self.tip_point
        artifacts = super().build_artifacts()

        root_V_diff = self.root_P_controlled_feature - self.root_P_reference_feature
        root_V_diff[2] = 0.0
        distance = root_V_diff.norm()
        artifacts.observation = sm.logic_and(
            distance >= self.lower_limit,
            distance <= self.upper_limit,
        )
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class AngleMonitor(
    FeatureFunctionMonitor,
    HasTipVectorParameter,
    HasReferenceVectorParameter,
    HasLowerAngleParameter,
    HasUpperAngleParameter,
):
    """
    Observes ``True`` while the angle between the controlled and reference vector is
    within ``[lower_angle, upper_angle]``.
    """

    def build_artifacts(self) -> NodeArtifacts:
        self.reference_feature = self.reference_vector
        self.controlled_feature = self.tip_vector
        artifacts = super().build_artifacts()

        expr = self.root_V_reference_feature.angle_between(
            self.root_V_controlled_feature
        )
        artifacts.observation = sm.logic_and(
            expr > self.lower_angle, expr < self.upper_angle
        )
        return artifacts

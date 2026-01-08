from dataclasses import field, dataclass
from typing import Union

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class FeatureMonitor(MotionStatechartNode):
    tip_link: Body
    root_link: Body
    reference_feature: Union[Point3, Vector3] = field(init=False)
    controlled_feature: Union[Point3, Vector3] = field(init=False)

    def __post_init__(self):
        root_reference_feature = context.world.transform(
            target_frame=self.root_link, spatial_object=self.reference_feature
        )
        tip_controlled_feature = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.controlled_feature
        )

        root_T_tip = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        )
        if isinstance(self.controlled_feature, Point3):
            self.root_P_controlled_feature = root_T_tip @ tip_controlled_feature
        elif isinstance(self.controlled_feature, Vector3):
            self.root_V_controlled_feature = root_T_tip @ tip_controlled_feature

        if isinstance(self.reference_feature, Point3):
            self.root_P_reference_feature = root_reference_feature
        if isinstance(self.reference_feature, Vector3):
            self.root_V_reference_feature = root_reference_feature


@dataclass
class HeightMonitor(FeatureMonitor):
    reference_point: Point3
    tip_point: Point3
    lower_limit: float
    upper_limit: float

    def __post_init__(self):
        self.reference_feature = self.reference_point
        self.controlled_feature = self.tip_point
        super().__post_init__()

        distance = (
            self.root_P_controlled_feature - self.root_P_reference_feature
        ) @ Vector3.Z()
        expr = sm.logic_and(
            distance >= self.lower_limit,
            distance <= self.upper_limit,
        )
        self.observation_expression = expr


@dataclass
class PerpendicularMonitor(FeatureMonitor):
    reference_normal: Vector3
    tip_normal: Vector3
    threshold: float = 0.01

    def __post_init__(self):
        self.reference_feature = self.reference_normal
        self.controlled_feature = self.tip_normal
        super().__post_init__()

        expr = self.root_V_reference_feature[:3] @ self.root_V_controlled_feature[:3]
        self.observation_expression = sm.abs(expr) <= self.threshold


@dataclass
class DistanceMonitor(FeatureMonitor):
    reference_point: Point3
    tip_point: Point3
    lower_limit: float
    upper_limit: float

    def __post_init__(self):
        self.reference_feature = self.reference_point
        self.controlled_feature = self.tip_point
        super().__post_init__()

        root_V_diff = self.root_P_controlled_feature - self.root_P_reference_feature
        root_V_diff[2] = 0.0
        distance = root_V_diff.norm()
        self.observation_expression = sm.logic_and(
            distance >= self.lower_limit,
            distance <= self.upper_limit,
        )


@dataclass
class AngleMonitor(FeatureMonitor):
    reference_vector: Vector3
    tip_vector: Vector3
    lower_angle: float
    upper_angle: float

    def __post_init__(self):
        self.reference_feature = self.reference_vector
        self.controlled_feature = self.tip_vector
        super().__post_init__()

        expr = sm.angle_between_vector(
            self.root_V_reference_feature, self.root_V_controlled_feature
        )
        self.observation_expression = sm.logic_and(
            sm.greater(expr, self.lower_angle), sm.less(expr, self.upper_angle)
        )

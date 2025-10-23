from __future__ import division

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA, Task
from giskardpy.utils.decorators import validated_dataclass
from semantic_digital_twin.spatial_types.symbol_manager import symbol_manager
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body


@validated_dataclass
class Pointing(Task):
    tip_link: Body
    goal_point: cas.Point3
    root_link: Body
    pointing_axis: cas.Vector3
    max_velocity: float = 0.3
    threshold: float = 0.01
    weight: float = WEIGHT_BELOW_CA

    def __post_init__(self):
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param goal_point: where to point pointing_axis at.
        :param root_link: root link of the kinematic chain.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        :param weight:
        """
        self.root_P_goal_point = god_map.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_point
        ).to_np()

        self.tip_V_pointing_axis = god_map.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        self.tip_V_pointing_axis.scale(1)

        root_T_tip = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_goal_point = symbol_manager.register_point3(
            name=f"{self.name}.root_P_goal_point",
            provider=lambda: self.root_P_goal_point,
        )
        root_P_goal_point.reference_frame = self.root_link
        tip_V_pointing_axis = cas.Vector3.from_iterable(self.tip_V_pointing_axis)

        root_V_goal_axis = root_P_goal_point - root_T_tip.to_position()
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip_link
        root_V_goal_axis.vis_frame = self.tip_link
        god_map.debug_expression_manager.add_debug_expression(
            "root_V_pointing_axis", root_V_pointing_axis, color=Color(1, 0, 0, 1)
        )
        god_map.debug_expression_manager.add_debug_expression(
            "goal_point", root_P_goal_point, color=Color(0, 0, 1, 1)
        )
        god_map.debug_expression_manager.add_debug_expression(
            "root_V_goal_axis", root_V_goal_axis, color=Color(0, 1, 0, 1)
        )
        self.add_vector_goal_constraints(
            frame_V_current=root_V_pointing_axis,
            frame_V_goal=root_V_goal_axis,
            reference_velocity=self.max_velocity,
            weight=self.weight,
        )
        self.observation_expression = (
            root_V_pointing_axis.angle_between(root_V_goal_axis) <= self.threshold
        )


@validated_dataclass
class PointingCone(Task):
    tip_link: Body
    goal_point: cas.Point3
    root_link: Body
    pointing_axis: cas.Vector3
    cone_theta: float = 0.0
    max_velocity: float = 0.3
    threshold: float = 0.01
    weight: float = WEIGHT_BELOW_CA

    def __post_init__(self):
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param goal_point: where to point pointing_axis at.
        :param root_link: root link of the kinematic chain.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        :param weight:
        """
        self.root_P_goal_point = god_map.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_point
        ).to_np()

        self.tip_V_pointing_axis = god_map.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        self.tip_V_pointing_axis.scale(1)

        root_T_tip = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_goal_point = symbol_manager.register_point3(
            name=f"{self.name}.root_P_goal_point",
            provider=lambda: self.root_P_goal_point,
        )
        root_P_goal_point.reference_frame = self.root_link
        tip_V_pointing_axis = cas.Vector3.from_iterable(self.tip_V_pointing_axis)

        root_V_goal_axis = root_P_goal_point - root_T_tip.to_position()
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip_link
        root_V_goal_axis.vis_frame = self.tip_link
        god_map.debug_expression_manager.add_debug_expression(
            "root_V_pointing_axis", root_V_pointing_axis, color=Color(1, 0, 0, 1)
        )
        god_map.debug_expression_manager.add_debug_expression(
            "goal_point", root_P_goal_point, color=Color(0, 0, 1, 1)
        )

        root_V_goal_axis_proj = root_V_pointing_axis.project_to_cone(
            root_V_goal_axis, self.cone_theta
        )
        root_V_goal_axis_proj.vis_frame = self.tip_link
        god_map.debug_expression_manager.add_debug_expression(
            "cone_axis", root_V_goal_axis, color=Color(1, 1, 0, 1)
        )
        god_map.debug_expression_manager.add_debug_expression(
            "projected_axis", root_V_goal_axis_proj, color=Color(1, 1, 0, 1)
        )

        self.add_vector_goal_constraints(
            frame_V_current=root_V_pointing_axis,
            frame_V_goal=root_V_goal_axis_proj,
            reference_velocity=self.max_velocity,
            weight=self.weight,
        )
        self.observation_expression = (
            root_V_pointing_axis.angle_between(root_V_goal_axis_proj) <= self.threshold
        )

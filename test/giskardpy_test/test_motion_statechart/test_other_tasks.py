from dataclasses import fields
from math import radians

import numpy as np
import pytest

from giskardpy.motion_statechart.weights import DefaultWeights
from cramph.data_types import LifeCycleValues, ObservationStateValues
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from giskardpy.motion_statechart.goals.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from cramph.composites import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
)
from giskardpy.motion_statechart.monitors.joint_monitors import JointPositionReached
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from cramph.statechart import (
    Statechart,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.motion_statechart.tasks.feature_functions import (
    AngleGoal,
    AlignPerpendicular,
    DistanceGoal,
    FeatureFunctionGoal,
    HeightGoal,
    ReachPoint,
)
from giskardpy.motion_statechart.tasks.pointing import (
    Pointing,
    PointingCone,
)
from semantic_digital_twin.specifications.connections import (
    RevoluteConnectionSpecification,
    ScrewConnectionSpecification,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Door,
    Bottle,
    BottleCap,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from krrood.utils import recursive_subclasses
from ..motion_control_context import create_context_with_motion_control


def angle_between_vector(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Returns the angle, in radians, between two vectors of length 3.

    :param v1: First vector.
    :param v2: Second vector.
    :return: Angle between ``v1`` and ``v2``, in radians.
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


class TestFeatureFunctions:
    """
    Test suite for feature function tasks (HeightGoal, DistanceGoal, etc.).
    """

    @pytest.mark.parametrize(
        "task_type",
        recursive_subclasses(FeatureFunctionGoal),
        ids=lambda task_type: task_type.__name__,
    )
    def test_a_feature_function_gives_way_to_collision_avoidance_by_default(
        self, task_type: type[FeatureFunctionGoal]
    ) -> None:
        weight = next(
            declared for declared in fields(task_type) if declared.name == "weight"
        )
        assert weight.default == DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE

    def test_height_goal_within_bounds(self, pr2_world_state_reset: World):
        """
        Test that HeightGoal successfully constrains the vertical distance between tip
        and reference points within specified bounds.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )

        tip_point = Point3(0, 0, 0, reference_frame=tip)
        reference_point = Point3(0, 0, 0, reference_frame=root)

        lower_limit = 0.3
        upper_limit = 0.5

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        height_goal = HeightGoal(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        statechart.add_node(height_goal)
        statechart.add_node(EndMotion.when_true(height_goal))

        executor.compile(statechart=statechart)
        executor.tick_until_end()

        assert height_goal.observation_state == ObservationStateValues.TRUE

        # Compute actual height difference
        root_P_tip = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=tip_point
        )
        root_P_ref = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=reference_point
        )
        height_diff = (root_P_tip.to_np()[:3] - root_P_ref.to_np()[:3])[2]

        assert (
            lower_limit <= height_diff <= upper_limit
        ), f"Height {height_diff:.4f} not in [{lower_limit}, {upper_limit}]"

    def test_height_goal_negative_bounds(self, pr2_world_state_reset: World):
        """
        Test HeightGoal with negative height bounds (tip below reference).
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )

        tip_point = Point3(0, 0, 0, reference_frame=tip)
        reference_point = Point3(0, 0, 1.0, reference_frame=root)

        lower_limit = -0.5
        upper_limit = -0.2

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        height_goal = HeightGoal(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        statechart.add_node(height_goal)
        statechart.add_node(EndMotion.when_true(height_goal))

        executor.compile(statechart=statechart)
        executor.tick_until_end()

        assert height_goal.observation_state == ObservationStateValues.TRUE

        # Verify actual height difference
        root_P_tip = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=tip_point
        )
        root_P_ref = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=reference_point
        )
        height_diff = (root_P_tip.to_np()[:3] - root_P_ref.to_np()[:3])[2]

        assert (
            lower_limit <= height_diff <= upper_limit
        ), f"Height {height_diff:.4f} not in [{lower_limit}, {upper_limit}]"

    def test_reach_point_drives_offset_point_onto_reference(
        self, pr2_world_state_reset: World
    ) -> None:
        """
        Test that ReachPoint drives a controlled tip point onto a reference point.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )

        tip_point = Point3(0.0, 0.0, 0.1, reference_frame=tip)
        root_P_tip_start = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=tip_point
        )
        reference_point = root_P_tip_start + Vector3(
            -0.1, 0.05, -0.05, reference_frame=root
        )

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        reach_point = ReachPoint(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
        )
        statechart.add_node(reach_point)
        statechart.add_node(EndMotion.when_true(reach_point))

        executor.compile(statechart=statechart)
        executor.tick_until_end()

        assert reach_point.observation_state == ObservationStateValues.TRUE

        root_P_tip_final = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=tip_point
        )
        assert (
            np.linalg.norm(root_P_tip_final.to_np() - reference_point.to_np())
            <= reach_point.threshold
        )

    def test_distance_goal_within_bounds(self, pr2_world_state_reset: World):
        """
        Test that DistanceGoal successfully constrains the horizontal distance (in x-y
        plane) between tip and reference points within specified bounds.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )

        tip_point = Point3(0, 0, 0, reference_frame=tip)
        reference_point = Point3(0, 0, 0, reference_frame=root)

        lower_limit = 0.5
        upper_limit = 0.7

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        distance_goal = DistanceGoal(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        statechart.add_node(distance_goal)
        statechart.add_node(EndMotion.when_true(distance_goal))

        executor.compile(statechart=statechart)
        executor.tick_until_end()

        assert distance_goal.observation_state == ObservationStateValues.TRUE

        # Compute actual horizontal distance
        root_P_tip = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=tip_point
        )
        root_P_ref = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=reference_point
        )
        diff = root_P_tip.to_np()[:3] - root_P_ref.to_np()[:3]
        # Distance in x-y plane only (ignore z)
        horizontal_distance = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

        assert (
            lower_limit <= horizontal_distance <= upper_limit
        ), f"Distance {horizontal_distance:.4f} not in [{lower_limit}, {upper_limit}]"

    def test_distance_goal_zero_distance(self, pr2_world_state_reset: World):
        """
        Test DistanceGoal with bounds that include zero (tip and reference at same x-y
        position).
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )

        tip_point = Point3(0, 0, 0, reference_frame=tip)
        # Reference point at same x-y but different z
        reference_point = Point3(0, 0, 0.5, reference_frame=root)

        lower_limit = 0.0
        upper_limit = 0.1

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        distance_goal = DistanceGoal(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        statechart.add_node(distance_goal)
        statechart.add_node(EndMotion.when_true(distance_goal))

        executor.compile(statechart=statechart)
        executor.tick_until_end()

        assert distance_goal.observation_state == ObservationStateValues.TRUE

        # Verify horizontal distance is near zero
        root_P_tip = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=tip_point
        )
        root_P_ref = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=reference_point
        )
        diff = root_P_tip.to_np()[:3] - root_P_ref.to_np()[:3]
        horizontal_distance = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

        assert (
            lower_limit <= horizontal_distance <= upper_limit
        ), f"Distance {horizontal_distance:.4f} not in [{lower_limit}, {upper_limit}]"

    def test_distance_goal_ignores_z_axis(self, pr2_world_state_reset: World):
        """
        Test that DistanceGoal only considers x-y plane distance and ignores z-axis.

        Even with large z difference, if x-y distance is within bounds, goal succeeds.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )

        tip_point = Point3(0, 0, 0, reference_frame=tip)
        # Reference point at specific x-y position but very different z
        reference_point = Point3(0.3, 0.4, 2.0, reference_frame=root)

        lower_limit = 0.45
        upper_limit = 0.55

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        distance_goal = DistanceGoal(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        statechart.add_node(distance_goal)
        statechart.add_node(EndMotion.when_true(distance_goal))

        executor.compile(statechart=statechart)
        executor.tick_until_end()

        assert distance_goal.observation_state == ObservationStateValues.TRUE

        # Verify z difference is large but goal still succeeded
        root_P_tip = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=tip_point
        )
        root_P_ref = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=reference_point
        )
        diff = root_P_tip.to_np()[:3] - root_P_ref.to_np()[:3]

        horizontal_distance = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

        assert (
            lower_limit <= horizontal_distance <= upper_limit
        ), f"Distance {horizontal_distance:.4f} not in [{lower_limit}, {upper_limit}]"

    def test_height_and_distance_combined(self, pr2_world_state_reset: World):
        """
        Test combining HeightGoal and DistanceGoal in parallel to constrain both
        vertical and horizontal distances simultaneously.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )

        tip_point = Point3(0, 0, 0, reference_frame=tip)
        reference_point = Point3(0, 0, 0, reference_frame=root)

        height_lower = 0.3
        height_upper = 0.5

        distance_lower = 0.4
        distance_upper = 0.6

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        combined_goal = Parallel(
            nodes=[
                HeightGoal(
                    root_link=root,
                    tip_link=tip,
                    tip_point=tip_point,
                    reference_point=reference_point,
                    lower_limit=height_lower,
                    upper_limit=height_upper,
                ),
                DistanceGoal(
                    root_link=root,
                    tip_link=tip,
                    tip_point=tip_point,
                    reference_point=reference_point,
                    lower_limit=distance_lower,
                    upper_limit=distance_upper,
                ),
            ]
        )
        statechart.add_node(combined_goal)
        statechart.add_node(EndMotion.when_true(combined_goal))

        executor.compile(statechart=statechart)
        executor.tick_until_end()

        assert combined_goal.observation_state == ObservationStateValues.TRUE

        # Verify both constraints are satisfied
        root_P_tip = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=tip_point
        )
        root_P_ref = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=reference_point
        )
        diff = root_P_tip.to_np()[:3] - root_P_ref.to_np()[:3]

        height_diff = diff[2]
        horizontal_distance = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

        assert (
            height_lower <= height_diff <= height_upper
        ), f"Height {height_diff:.4f} not in [{height_lower}, {height_upper}]"
        assert (
            distance_lower <= horizontal_distance <= distance_upper
        ), f"Distance {horizontal_distance:.4f} not in [{distance_lower}, {distance_upper}]"

    def test_distance_height_angle_perpendicular_combined(
        self, pr2_world_state_reset: World
    ):
        """
        Test combining DistanceGoal, HeightGoal, and AlignPerpendicular to constrain
        horizontal distance, vertical distance, and perpendicular alignment
        simultaneously.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )

        tip_point = Point3(0, 0, 0, reference_frame=tip)
        reference_point = Point3(0, 0, 0, reference_frame=root)

        tip_normal = Vector3(1, 0, 0, reference_frame=tip)
        reference_normal = Vector3(1, 0, 0, reference_frame=root)

        height_lower = 0.3
        height_upper = 0.5

        distance_lower = 0.4
        distance_upper = 0.6

        perpendicular_threshold = 0.01

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        height_goal = HeightGoal(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
            lower_limit=height_lower,
            upper_limit=height_upper,
        )
        distance_goal = DistanceGoal(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
            lower_limit=distance_lower,
            upper_limit=distance_upper,
        )
        align_perpendicular = AlignPerpendicular(
            root_link=root,
            tip_link=tip,
            tip_normal=tip_normal,
            reference_normal=reference_normal,
            threshold=perpendicular_threshold,
        )

        combined_goal = Parallel(
            nodes=[height_goal, distance_goal, align_perpendicular]
        )
        statechart.add_node(combined_goal)
        statechart.add_node(EndMotion.when_true(combined_goal))

        executor.compile(statechart=statechart)
        executor.tick_until_end()

        assert combined_goal.observation_state == ObservationStateValues.TRUE
        assert height_goal.observation_state == ObservationStateValues.TRUE
        assert distance_goal.observation_state == ObservationStateValues.TRUE
        assert align_perpendicular.observation_state == ObservationStateValues.TRUE

        root_P_tip = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=tip_point
        )
        root_P_ref = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=reference_point
        )
        diff = root_P_tip.to_np()[:3] - root_P_ref.to_np()[:3]

        height_diff = diff[2]
        horizontal_distance = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

        assert (
            height_lower <= height_diff <= height_upper
        ), f"Height {height_diff:.4f} not in [{height_lower}, {height_upper}]"
        assert (
            distance_lower <= horizontal_distance <= distance_upper
        ), f"Distance {horizontal_distance:.4f} not in [{distance_lower}, {distance_upper}]"

        root_V_tip_normal = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=tip_normal
        )
        root_V_tip_normal.scale(1)
        root_V_ref_normal = pr2_world_state_reset.transform(
            target_frame=root, spatial_object=reference_normal
        )
        root_V_ref_normal.scale(1)

        v_tip_normal = root_V_tip_normal.to_np()[:3]
        v_ref_normal = root_V_ref_normal.to_np()[:3]

        eps = 1e-9
        assert np.linalg.norm(v_ref_normal) > eps, "reference normal became zero-length"
        assert np.linalg.norm(v_tip_normal) > eps, "tip normal became zero-length"

        perp_angle = angle_between_vector(v_tip_normal, v_ref_normal)
        target = np.pi / 2

        assert abs(perp_angle - target) <= perpendicular_threshold, (
            f"AlignPerpendicular failed: final angle {perp_angle:.6f} rad, "
            f"target {target:.6f} rad, threshold {perpendicular_threshold:.6f} rad"
        )


def test_pointing(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)

    goal_point = Point3(2, 0, 0, reference_frame=root)
    pointing_axis = Vector3.X(reference_frame=tip)

    pointing = Pointing(
        root_link=root,
        tip_link=tip,
        goal_point=goal_point,
        pointing_axis=pointing_axis,
    )
    statechart.add_node(pointing)
    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = pointing.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()


def test_pointing_lowers_the_torso_when_the_head_cannot_tilt_far_enough(
    pr2_world_state_reset: World,
):
    """
    A point on the floor close in front of a PR2 whose torso is fully raised lies
    further below the head than its tilt joint reaches, and lowering the torso brings it
    within reach.

    The torso only moves the head, never turns it, so pointing has to count where the
    head is, not just which way it faces, for the torso to help.
    """
    world = pr2_world_state_reset
    base = world.get_kinematic_structure_entity_by_name("base_link")
    head = world.get_kinematic_structure_entity_by_name("head_tilt_link")
    torso = world.get_connection_by_name("torso_lift_joint")
    head_tilt = world.get_connection_by_name("head_tilt_joint")
    torso.position = torso.dof.limits.upper.position
    world.notify_state_change()

    # Steeper below the raised head than the tilt joint reaches, but not once the torso
    # is lowered.
    world_P_head = head.global_pose.to_np()[:3, 3]
    steepest_tilt_at_full_height = head_tilt.dof.limits.upper.position + 0.03
    distance = world_P_head[2] / np.tan(steepest_tilt_at_full_height)
    goal_point = Point3(
        world_P_head[0] + distance, world_P_head[1], 0.0, reference_frame=world.root
    )

    executor = StatechartExecutor(
        context=StatechartContext(world=world),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    pointing = Pointing(
        root_link=base,
        tip_link=head,
        goal_point=goal_point,
        pointing_axis=Vector3.X(reference_frame=head),
    )
    statechart.add_node(pointing)
    statechart.add_node(still_progressing := StillProgressing(monitored_node=pointing))
    statechart.add_node(still_progressing.cancel_motion())
    end = EndMotion()
    end.start_condition = pointing.observes_true
    statechart.add_node(end)

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert torso.position < torso.dof.limits.upper.position - 0.05


def test_pointing_follows_a_goal_point_given_as_an_expression_of_a_moving_frame(
    pr2_world_copy: World,
) -> None:
    """
    A goal given as the expression of where its frame is follows that frame while the
    pointing runs, and the joint moving the frame still reaches its own goal.
    """
    world = pr2_world_copy
    base = world.get_kinematic_structure_entity_by_name("base_link")
    head = world.get_kinematic_structure_entity_by_name("head_tilt_link")
    with world.modify_world():
        region = Body(name=PrefixedName("region"))
        world.add_kinematic_structure_entity(region)
        region_joint = PrismaticConnection.create_with_dofs(
            world=world,
            parent=world.root,
            child=region,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                2, 0, 1, reference_frame=world.root
            ),
            axis=Vector3.Y(),
            dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                lower_position=-1.0, upper_position=1.0, maximum_speed=0.5
            ),
        )
        world.add_connection(region_joint)

    executor = StatechartExecutor(
        context=StatechartContext(world=world),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    world_T_region = world.compose_forward_kinematics_expression(world.root, region)
    world_T_region.reference_frame = world.root
    pointing = Pointing(
        root_link=base,
        tip_link=head,
        goal_point=world_T_region.position,
        pointing_axis=Vector3.X(reference_frame=head),
    )
    region_goal = JointPositionList(
        goal_state=JointState.from_mapping({region_joint: 0.8})
    )
    statechart.add_node(both := Parallel(nodes=[pointing, region_goal]))
    end = EndMotion()
    end.start_condition = both.observes_true
    statechart.add_node(end)
    executor.compile(statechart=statechart)

    executor.tick_until_end()

    assert region_joint.position == pytest.approx(0.8, abs=region_goal.threshold)
    head_pose = head.global_pose.to_np()
    head_to_region = region.global_pose.to_np()[:3, 3] - head_pose[:3, 3]
    angle = np.arccos(
        np.dot(head_pose[:3, 0], head_to_region) / np.linalg.norm(head_to_region)
    )
    assert angle <= pointing.threshold


def test_pointing_cone(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)

    goal_point = Point3(-1, 0, 5, reference_frame=root)
    pointing_axis = Vector3.X(tip)
    cone_theta = radians(20)
    pointing_cone = PointingCone(
        root_link=root,
        tip_link=tip,
        goal_point=goal_point,
        pointing_axis=pointing_axis,
        cone_theta=cone_theta,
    )
    statechart.add_node(pointing_cone)
    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = pointing_cone.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    # Check if angle between pointing axis and tip->goal vector is within the cone
    root_V_pointing_axis = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=pointing_axis
    )
    root_V_pointing_axis.scale(1)
    v_pointing = root_V_pointing_axis.to_np()[:3]

    root_P_goal = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=goal_point
    )
    tip_origin_in_root = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=Point3(0, 0, 0, reference_frame=tip)
    )
    root_V_goal_axis = root_P_goal - tip_origin_in_root
    root_V_goal_axis.scale(1)
    v_goal = root_V_goal_axis.to_np()[:3]

    eps = 1e-9
    assert np.linalg.norm(v_pointing) > eps, "pointing axis became zero-length"
    assert np.linalg.norm(v_goal) > eps, "tip->goal vector became zero-length"

    angle = angle_between_vector(v_pointing, v_goal)

    assert (
        angle <= cone_theta + pointing_cone.threshold
    ), f"PointingCone failed: final angle {angle:.6f} rad > cone_theta {cone_theta:.6f} rad"


def test_align_planes(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)

    goal_normal = Vector3.X(reference_frame=root)
    tip_normal = Vector3.Y(reference_frame=tip)

    align_planes = AlignPlanes(
        root_link=root, tip_link=tip, goal_normal=goal_normal, tip_normal=tip_normal
    )
    statechart.add_node(align_planes)

    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = align_planes.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    # Check if the angle between normal vectors is below the threshold
    root_V_goal_normal = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=goal_normal
    )
    root_V_goal_normal.scale(1)
    root_V_tip_normal = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=tip_normal
    )
    root_V_tip_normal.scale(1)
    v_tip = root_V_tip_normal.to_np()[:3]
    v_goal = root_V_goal_normal.to_np()[:3]

    eps = 1e-9
    assert np.linalg.norm(v_goal) > eps, "goal normal became zero-length"
    assert np.linalg.norm(v_tip) > eps, "tip normal became zero-length"

    angle = angle_between_vector(v_tip, v_goal)

    assert (
        angle <= align_planes.threshold
    ), f"AlignPlanes failed: final angle {angle:.6f} rad > threshold {align_planes.threshold:.6f} rad"


def test_align_perpendicular(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)

    goal_normal = Vector3.X(reference_frame=root)
    tip_normal = Vector3.X(reference_frame=tip)

    align_perp = AlignPerpendicular(
        root_link=root,
        tip_link=tip,
        reference_normal=goal_normal,
        tip_normal=tip_normal,
    )
    statechart.add_node(align_perp)

    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = align_perp.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    # Check if the angle between normals is (approximately) 90 degrees
    root_V_goal_normal = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=goal_normal
    )
    root_V_goal_normal.scale(1)
    root_V_tip_normal = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=tip_normal
    )
    root_V_tip_normal.scale(1)

    v_tip = root_V_tip_normal.to_np()[:3]
    v_goal = root_V_goal_normal.to_np()[:3]

    eps = 1e-9
    assert np.linalg.norm(v_goal) > eps, "goal normal became zero-length"
    assert np.linalg.norm(v_tip) > eps, "tip normal became zero-length"

    angle = angle_between_vector(v_tip, v_goal)
    target = np.pi / 2

    assert abs(angle - target) <= align_perp.threshold, (
        f"AlignPerpendicular failed: final angle {angle:.6f} rad, "
        f"target {target:.6f} rad, threshold {align_perp.threshold:.6f} rad"
    )


def test_angle_goal(pr2_world_state_reset: World):
    """
    Ensure AngleGoal drives the angle between tip_vector and reference_vector into the
    interval [lower_angle, upper_angle].
    """
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)

    tip_vector = Vector3.Y(reference_frame=tip)
    reference_vector = Vector3.X(reference_frame=root)

    lower_angle = radians(30)
    upper_angle = radians(32)

    angle_goal = AngleGoal(
        root_link=root,
        tip_link=tip,
        tip_vector=tip_vector,
        reference_vector=reference_vector,
        lower_angle=lower_angle,
        upper_angle=upper_angle,
    )
    statechart.add_node(angle_goal)

    statechart.add_node(EndMotion.when_true(angle_goal))

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    root_V_tip = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=tip_vector
    )
    root_V_tip.scale(1)
    root_V_ref = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=reference_vector
    )
    root_V_ref.scale(1)

    v_tip = root_V_tip.to_np()[:3]
    v_ref = root_V_ref.to_np()[:3]

    eps = 1e-9
    assert np.linalg.norm(v_tip) > eps, "tip_vector became zero-length"
    assert np.linalg.norm(v_ref) > eps, "reference_vector became zero-length"

    angle = angle_between_vector(v_tip, v_ref)

    assert (
        lower_angle <= angle <= upper_angle
    ), f"AngleGoal failed: final angle {angle:.6f} rad not in [{lower_angle:.6f}, {upper_angle:.6f}]"


class TestArticulateGraspedMechanism:
    def test_the_mechanism_is_articulated_once_its_parts_reached_their_goals(
        self, prismatic_bot2: World
    ):
        """
        Both parts are created by the goal and end only when it ends, so it reads what
        they observe now.
        """
        executor = StatechartExecutor(
            context=StatechartContext(world=prismatic_bot2),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(
            open_goal := ArticulateGraspedMechanism(
                environment_link=prismatic_bot2.get_body_by_name("robot"),
                tip_link=prismatic_bot2.get_body_by_name("robot2"),
                goal_joint_state=0.0,
            )
        )
        executor.compile(statechart=motion_statechart)

        assert set(open_goal._observation_expression.free_variables()) == {
            part.observation_variable for part in open_goal.nodes
        }

    def test_open(self, pr2_world_copy, tmp_path):

        world_T_door = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.5, z=1, yaw=np.pi, reference_frame=pr2_world_copy.root
        )
        world_T_hinge = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.5, y=-0.5, z=1, yaw=np.pi, reference_frame=pr2_world_copy.root
        )
        limits = DegreeOfFreedomLimits.from_position_range_and_speed(
            lower_position=-np.pi / 2, upper_position=np.pi / 2, maximum_speed=1
        )
        with pr2_world_copy.modify_world():
            door = Door.create_with_new_body_in_world(
                name="door",
                world=pr2_world_copy,
                world_root_T_self=world_T_door,
                parent_connection_specification=RevoluteConnectionSpecification(
                    dof_limits=limits,
                    axis=Vector3.Z(),
                    connection_T_child=world_T_hinge.inverse() @ world_T_door,
                ),
            )

            handle = Handle.create_with_new_body_in_world(
                name="handle",
                world=pr2_world_copy,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.5,
                    y=0.45,
                    z=1,
                    yaw=np.pi,
                    reference_frame=pr2_world_copy.root,
                ),
            )

            door.add(handle)

        root_C_hinge = door.movable_joint

        r_tip = pr2_world_copy.get_body_by_name("r_gripper_tool_frame")
        handle = pr2_world_copy.get_semantic_annotations_by_type(Handle)[0].root
        open_goal = 1
        close_goal = -1

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_copy),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        statechart.add_nodes(
            [
                Sequence(
                    nodes=[
                        CartesianPose(
                            root_link=pr2_world_copy.root,
                            tip_link=r_tip,
                            goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                                yaw=np.pi, reference_frame=handle
                            ),
                        ),
                        opening := Parallel(
                            nodes=[
                                ArticulateGraspedMechanism(
                                    tip_link=r_tip,
                                    environment_link=handle,
                                    goal_joint_state=open_goal,
                                ),
                                JointPositionReached(
                                    connection=root_C_hinge,
                                    position=open_goal,
                                    name="opened",
                                ),
                            ]
                        ),
                        closing := Parallel(
                            nodes=[
                                ArticulateGraspedMechanism(
                                    tip_link=r_tip,
                                    environment_link=handle,
                                    goal_joint_state=close_goal,
                                ),
                                JointPositionReached(
                                    connection=root_C_hinge,
                                    position=close_goal,
                                    name="closed",
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )
        statechart.add_node(EndMotion.when_true(statechart.nodes[0]))

        executor.compile(statechart=statechart)
        executor.tick_until_end()
        statechart.draw(str(tmp_path / "muh.pdf"))

        assert opening.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert closing.life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_unscrew_and_tighten_bottle_cap(self, pr2_world_copy):
        screw_pitch = 0.03
        unscrew_goal = 2 * np.pi
        limits = DegreeOfFreedomLimits.from_position_range_and_speed(
            lower_position=0, upper_position=unscrew_goal, maximum_speed=1
        )
        with pr2_world_copy.modify_world():
            # The bottle lies on its side, its thread axis pointing towards the robot.
            Bottle.create_with_new_body_in_world(
                name="bottle",
                world=pr2_world_copy,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.77, y=-0.2, z=0.9, reference_frame=pr2_world_copy.root
                ),
                scale=Scale(0.2, 0.06, 0.06),
            )

            bottle_cap = BottleCap.create_with_new_body_in_world(
                name="bottle_cap",
                world=pr2_world_copy,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.65, y=-0.2, z=0.9, reference_frame=pr2_world_copy.root
                ),
                scale=Scale(0.02, 0.04, 0.04),
                # The thread axis points from the bottle out through the cap, here
                # towards the robot, so unscrewing moves the cap away from the bottle.
                parent_connection_specification=ScrewConnectionSpecification(
                    axis=Vector3(-1, 0, 0),
                    dof_limits=limits,
                    screw_pitch=screw_pitch,
                ),
            )

        root_C_screw = bottle_cap.movable_joint
        cap_body = bottle_cap.root
        r_tip = pr2_world_copy.get_body_by_name("r_gripper_tool_frame")

        world_T_cap_before = pr2_world_copy.compute_forward_kinematics_np(
            pr2_world_copy.root, cap_body
        )

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_copy),
            extensions=[MotionControl()],
        )
        unscrew_statechart = Statechart(context=executor.context)
        unscrew_statechart.add_nodes(
            [
                sequence := Sequence(
                    nodes=[
                        # Grasp with the roll axis of the gripper collinear with the
                        # screw axis, so the wrist can follow the cap's rotation.
                        CartesianPose(
                            root_link=pr2_world_copy.root,
                            tip_link=r_tip,
                            goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                                reference_frame=cap_body
                            ),
                        ),
                        open := ArticulateGraspedMechanism(
                            tip_link=r_tip,
                            environment_link=cap_body,
                            goal_joint_state=unscrew_goal,
                        ),
                    ]
                ),
            ]
        )
        unscrew_statechart.add_node(EndMotion.when_true(sequence))

        executor.compile(statechart=unscrew_statechart)
        executor.tick_until_end()

        assert open.life_cycle_state == LifeCycleValues.SUCCEEDED

        # One full turn must have moved the cap one screw pitch along the screw axis,
        # away from the bottle (towards the robot, -x).
        world_T_cap_unscrewed = pr2_world_copy.compute_forward_kinematics_np(
            pr2_world_copy.root, cap_body
        )
        cap_translation = world_T_cap_unscrewed[:3, 3] - world_T_cap_before[:3, 3]
        np.testing.assert_allclose(cap_translation, [-screw_pitch, 0.0, 0.0], atol=1e-3)

        # Tighten again, this time commanding how far the cap must travel back down
        # the thread instead of how far it must rotate.
        unscrew_travel_distance = float(np.linalg.norm(cap_translation))
        tightened_goal_joint_state = (
            unscrew_goal
            - root_C_screw.rotation_angle_for_travel_distance(unscrew_travel_distance)
        )

        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_copy),
            extensions=[MotionControl()],
        )
        tighten_statechart = Statechart(context=executor.context)
        tighten_statechart.add_nodes(
            [
                close := ArticulateGraspedMechanism(
                    tip_link=r_tip,
                    environment_link=cap_body,
                    goal_joint_state=tightened_goal_joint_state,
                ),
            ]
        )
        tighten_statechart.add_node(EndMotion.when_true(close))

        executor.compile(statechart=tighten_statechart)
        executor.tick_until_end()

        assert close.observation_state == ObservationStateValues.TRUE

        # The tightened cap must be back at its initial pose.
        world_T_cap_tightened = pr2_world_copy.compute_forward_kinematics_np(
            pr2_world_copy.root, cap_body
        )
        np.testing.assert_allclose(
            world_T_cap_tightened[:3, 3], world_T_cap_before[:3, 3], atol=1e-3
        )

    def _expanded_nodes(self, goal: ArticulateGraspedMechanism, world: World):
        """
        Expands a goal on a statechart of its own and returns the children it built.
        """
        statechart = Statechart(context=create_context_with_motion_control(world))
        statechart.add_node(goal)
        goal.expand()
        return goal.nodes

    def test_the_mechanism_yields_to_collision_avoidance_but_not_the_grasp_by_default(
        self, prismatic_bot
    ):
        """
        A caller that asks for nothing in particular gets the weights the motion needs:
        following the mechanism gives way to collision avoidance, holding the grasped
        part does not.
        """
        grasped_part = prismatic_bot.get_body_by_name("robot")
        goal = ArticulateGraspedMechanism(
            tip_link=prismatic_bot.root,
            environment_link=grasped_part,
            goal_joint_state=0.0,
        )

        nodes = self._expanded_nodes(goal, prismatic_bot)

        mechanism_node = next(
            node for node in nodes if isinstance(node, JointPositionList)
        )
        grasp_node = next(node for node in nodes if isinstance(node, CartesianPose))
        assert mechanism_node.weight == DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
        assert grasp_node.weight == DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE

    def test_mechanism_and_grasp_are_weighed_separately(self, prismatic_bot):
        """
        Driving the mechanism and holding the grasped part are separate goals, so a
        caller can let the mechanism yield to collision avoidance while the grasp keeps
        outranking it.
        """
        grasped_part = prismatic_bot.get_body_by_name("robot")
        goal = ArticulateGraspedMechanism(
            tip_link=prismatic_bot.root,
            environment_link=grasped_part,
            goal_joint_state=0.0,
            mechanism_weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
            grasp_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        )

        nodes = self._expanded_nodes(goal, prismatic_bot)

        mechanism_node = next(
            node for node in nodes if isinstance(node, JointPositionList)
        )
        grasp_node = next(node for node in nodes if isinstance(node, CartesianPose))
        assert mechanism_node.weight == DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
        assert grasp_node.weight == DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE

    def test_a_goal_within_the_limits_is_kept(self, prismatic_bot):
        grasped_part = prismatic_bot.get_body_by_name("robot")
        connection = grasped_part.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        limits = connection.dof.limits
        within_limits = (limits.lower.position + limits.upper.position) / 2
        goal = ArticulateGraspedMechanism(
            tip_link=prismatic_bot.root,
            environment_link=grasped_part,
            goal_joint_state=within_limits,
        )

        self._expanded_nodes(goal, prismatic_bot)

        assert goal.goal_joint_state == within_limits

    def test_a_goal_beyond_the_upper_limit_is_clamped_to_it(self, prismatic_bot):
        """
        A commanded state the mechanism cannot reach is cut down to its upper limit,
        rather than left as a goal the solver can never observe as reached.
        """
        grasped_part = prismatic_bot.get_body_by_name("robot")
        goal = ArticulateGraspedMechanism(
            tip_link=prismatic_bot.root,
            environment_link=grasped_part,
            goal_joint_state=1e3,
        )

        self._expanded_nodes(goal, prismatic_bot)

        assert goal.goal_joint_state == goal.connection.dof.limits.upper.position

    def test_a_goal_below_the_lower_limit_is_clamped_to_it(self, prismatic_bot):
        grasped_part = prismatic_bot.get_body_by_name("robot")
        goal = ArticulateGraspedMechanism(
            tip_link=prismatic_bot.root,
            environment_link=grasped_part,
            goal_joint_state=-1e3,
        )

        self._expanded_nodes(goal, prismatic_bot)

        assert goal.goal_joint_state == goal.connection.dof.limits.lower.position

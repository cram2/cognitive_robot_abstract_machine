from math import radians

import numpy as np

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.open_close import Open, Close
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
)
from giskardpy.motion_statechart.monitors.joint_monitors import JointPositionReached
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
)
from giskardpy.motion_statechart.tasks.feature_functions import (
    AngleGoal,
    AlignPerpendicular,
    DistanceGoal,
    HeightGoal,
)
from giskardpy.motion_statechart.tasks.pointing import Pointing, PointingCone
from giskardpy.utils.math import angle_between_vector
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Door,
    Hinge,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)


class TestFeatureFunctions:
    """Test suite for feature function tasks (HeightGoal, DistanceGoal, etc.)."""

    def test_height_goal_within_bounds(self, pr2_world_state_reset: World):
        """
        Test that HeightGoal successfully constrains the vertical distance
        between tip and reference points within specified bounds.
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

        msc = MotionStatechart()
        height_goal = HeightGoal(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        msc.add_node(height_goal)
        msc.add_node(EndMotion.when_true(height_goal))

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

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

        msc = MotionStatechart()
        height_goal = HeightGoal(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        msc.add_node(height_goal)
        msc.add_node(EndMotion.when_true(height_goal))

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

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

    def test_distance_goal_within_bounds(self, pr2_world_state_reset: World):
        """
        Test that DistanceGoal successfully constrains the horizontal distance
        (in x-y plane) between tip and reference points within specified bounds.
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

        msc = MotionStatechart()
        distance_goal = DistanceGoal(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        msc.add_node(distance_goal)
        msc.add_node(EndMotion.when_true(distance_goal))

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

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
        Test DistanceGoal with bounds that include zero (tip and reference at same x-y position).
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

        msc = MotionStatechart()
        distance_goal = DistanceGoal(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        msc.add_node(distance_goal)
        msc.add_node(EndMotion.when_true(distance_goal))

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

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

        msc = MotionStatechart()
        distance_goal = DistanceGoal(
            root_link=root,
            tip_link=tip,
            tip_point=tip_point,
            reference_point=reference_point,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        msc.add_node(distance_goal)
        msc.add_node(EndMotion.when_true(distance_goal))

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

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
        Test combining HeightGoal and DistanceGoal in parallel to constrain
        both vertical and horizontal distances simultaneously.
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

        msc = MotionStatechart()
        combined_goal = Parallel(
            [
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
        msc.add_node(combined_goal)
        msc.add_node(EndMotion.when_true(combined_goal))

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

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
        Test combining DistanceGoal, HeightGoal, and AlignPerpendicular
        to constrain horizontal distance, vertical distance, and perpendicular
        alignment simultaneously.
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

        msc = MotionStatechart()
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

        combined_goal = Parallel([height_goal, distance_goal, align_perpendicular])
        msc.add_node(combined_goal)
        msc.add_node(EndMotion.when_true(combined_goal))

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

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

    msc = MotionStatechart()

    goal_point = Point3(2, 0, 0, reference_frame=root)
    pointing_axis = Vector3.X(reference_frame=tip)

    pointing = Pointing(
        root_link=root,
        tip_link=tip,
        goal_point=goal_point,
        pointing_axis=pointing_axis,
    )
    msc.add_node(pointing)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = pointing.observation_variable

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
        )
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()


def test_pointing_cone(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    msc = MotionStatechart()

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
    msc.add_node(pointing_cone)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = pointing_cone.observation_variable

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
        )
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

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

    msc = MotionStatechart()

    goal_normal = Vector3.X(reference_frame=root)
    tip_normal = Vector3.Y(reference_frame=tip)

    align_planes = AlignPlanes(
        root_link=root, tip_link=tip, goal_normal=goal_normal, tip_normal=tip_normal
    )
    msc.add_node(align_planes)

    end = EndMotion()
    msc.add_node(end)
    end.start_condition = align_planes.observation_variable

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
        )
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

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

    msc = MotionStatechart()

    goal_normal = Vector3.X(reference_frame=root)
    tip_normal = Vector3.X(reference_frame=tip)

    align_perp = AlignPerpendicular(
        root_link=root,
        tip_link=tip,
        reference_normal=goal_normal,
        tip_normal=tip_normal,
    )
    msc.add_node(align_perp)

    end = EndMotion()
    msc.add_node(end)
    end.start_condition = align_perp.observation_variable

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

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
    Ensure AngleGoal drives the angle between tip_vector and reference_vector
    into the interval [lower_angle, upper_angle].
    """
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    msc = MotionStatechart()

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
    msc.add_node(angle_goal)

    msc.add_node(EndMotion.when_true(angle_goal))

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

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


class TestOpenClose:
    def test_open(self, pr2_world_copy, tmp_path):

        with pr2_world_copy.modify_world():
            door = Door.create_with_new_body_in_world(
                name=PrefixedName("door"),
                world=pr2_world_copy,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.5, z=1, yaw=np.pi, reference_frame=pr2_world_copy.root
                ),
            )

            handle = Handle.create_with_new_body_in_world(
                name=PrefixedName("handle"),
                world=pr2_world_copy,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.5,
                    y=0.45,
                    z=1,
                    yaw=np.pi,
                    reference_frame=pr2_world_copy.root,
                ),
            )

            lower_limits = DerivativeMap()
            lower_limits.position = -np.pi / 2
            lower_limits.velocity = -1
            upper_limits = DerivativeMap()
            upper_limits.position = np.pi / 2
            upper_limits.velocity = 1

            hinge = Hinge.create_with_new_body_in_world(
                name=PrefixedName("hinge"),
                world=pr2_world_copy,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.5,
                    y=-0.5,
                    z=1,
                    yaw=np.pi,
                    reference_frame=pr2_world_copy.root,
                ),
                connection_limits=DegreeOfFreedomLimits(
                    lower=lower_limits, upper=upper_limits
                ),
                active_axis=Vector3.Z(),
            )

            door.add_handle(handle)
            door.add_hinge(hinge=hinge)

        root_C_hinge = door.hinge.root.parent_connection

        r_tip = pr2_world_copy.get_body_by_name("r_gripper_tool_frame")
        handle = pr2_world_copy.get_semantic_annotations_by_type(Handle)[0].root
        open_goal = 1
        close_goal = -1

        msc = MotionStatechart()
        msc.add_nodes(
            [
                Sequence(
                    [
                        CartesianPose(
                            root_link=pr2_world_copy.root,
                            tip_link=r_tip,
                            goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                                yaw=np.pi, reference_frame=handle
                            ),
                        ),
                        Parallel(
                            [
                                Open(
                                    tip_link=r_tip,
                                    environment_link=handle,
                                    goal_joint_state=open_goal,
                                ),
                                opened := JointPositionReached(
                                    connection=root_C_hinge,
                                    position=open_goal,
                                    name="opened",
                                ),
                            ]
                        ),
                        Parallel(
                            [
                                Close(
                                    tip_link=r_tip,
                                    environment_link=handle,
                                    goal_joint_state=close_goal,
                                ),
                                closed := JointPositionReached(
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
        msc.add_node(EndMotion.when_true(msc.nodes[0]))

        kin_sim = Executor(
            MotionStatechartContext(
                world=pr2_world_copy,
            )
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        msc.draw(str(tmp_path / "muh.pdf"))

        assert opened.observation_state == ObservationStateValues.TRUE
        assert closed.observation_state == ObservationStateValues.TRUE

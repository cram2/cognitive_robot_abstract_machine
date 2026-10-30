from __future__ import annotations

import numpy as np

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion, MotionStatechartNode
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from semantic_digital_twin.robots.hsrb import HSRBJoint
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

# %% helpers


def run_until_reached(world: World, goal: MotionStatechartNode) -> None:
    """
    Runs ``goal`` in ``world`` until it is reached.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    motion_statechart = Statechart(context=executor.context)
    motion_statechart.add_node(goal)
    motion_statechart.add_node(EndMotion.when_true(goal))
    executor.compile(statechart=motion_statechart)
    executor.tick_until_end()


def height_above_base(world: World, link_name: str) -> float:
    """
    :return: Height of the link called ``link_name`` above the HSR's base footprint.
    """
    return float(
        world.compute_forward_kinematics(
            world.get_kinematic_structure_entity_by_name("base_footprint"),
            world.get_kinematic_structure_entity_by_name(link_name),
        ).to_np()[2, 3]
    )


# %% joint goals on a mimic joint


def test_joint_goal_on_a_mimic_joint_moves_the_joint_it_mimics(
    hsr_world_copy: World, default_hsr_pose: dict[str, float]
) -> None:
    JointState.from_str_dict(default_hsr_pose, hsr_world_copy).apply_to(hsr_world_copy)
    torso_lift_joint: ActiveConnection1DOF = hsr_world_copy.get_connection_by_name(
        HSRBJoint.TORSO_LIFT
    )
    arm_lift_joint: ActiveConnection1DOF = hsr_world_copy.get_connection_by_name(
        HSRBJoint.ARM_LIFT
    )
    torso_goal = 0.1

    run_until_reached(
        hsr_world_copy,
        JointPositionList(
            goal_state=JointState.from_str_dict(
                {HSRBJoint.TORSO_LIFT: torso_goal, HSRBJoint.HAND_MOTOR: 1.23},
                hsr_world_copy,
            )
        ),
    )

    np.testing.assert_almost_equal(
        arm_lift_joint.position,
        (torso_goal - torso_lift_joint.offset) / torso_lift_joint.multiplier,
        decimal=2,
    )
    hand_palm_link = hsr_world_copy.get_kinematic_structure_entity_by_name(
        "hand_palm_link"
    )
    hand_T_finger_expected = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=-0.01675,
        pos_y=-0.0907,
        pos_z=0.0052,
        quat_x=-0.0434,
        quat_w=0.999,
        reference_frame=hand_palm_link,
    )
    np.testing.assert_almost_equal(
        hsr_world_copy.compute_forward_kinematics(
            hand_palm_link,
            hsr_world_copy.get_kinematic_structure_entity_by_name("hand_l_distal_link"),
        ).to_np(),
        hand_T_finger_expected.to_np(),
        decimal=2,
    )


def test_mimic_joint_velocity_limits_are_scaled_by_its_multiplier(
    hsr_world_copy: World,
) -> None:
    arm_lift_joint: ActiveConnection1DOF = hsr_world_copy.get_connection_by_name(
        HSRBJoint.ARM_LIFT
    )
    torso_lift_joint: ActiveConnection1DOF = hsr_world_copy.get_connection_by_name(
        HSRBJoint.TORSO_LIFT
    )

    assert (
        torso_lift_joint.dof.limits.upper.velocity
        == arm_lift_joint.dof.limits.upper.velocity * torso_lift_joint.multiplier
    )
    assert (
        torso_lift_joint.dof.limits.lower.velocity
        == arm_lift_joint.dof.limits.lower.velocity * torso_lift_joint.multiplier
    )


# %% cartesian goals through a mimic joint


def test_lifting_the_gripper_lifts_the_torso_along(
    hsr_world_copy: World, default_hsr_pose: dict[str, float]
) -> None:
    JointState.from_str_dict(default_hsr_pose, hsr_world_copy).apply_to(hsr_world_copy)
    torso_lift_joint: ActiveConnection1DOF = hsr_world_copy.get_connection_by_name(
        HSRBJoint.TORSO_LIFT
    )
    arm_lift_joint: ActiveConnection1DOF = hsr_world_copy.get_connection_by_name(
        HSRBJoint.ARM_LIFT
    )
    gripper = hsr_world_copy.get_kinematic_structure_entity_by_name(
        "hand_gripper_tool_frame"
    )
    arm_lift_before = arm_lift_joint.position
    torso_height_before = height_above_base(hsr_world_copy, "torso_lift_link")
    gripper_lift = 0.2

    run_until_reached(
        hsr_world_copy,
        CartesianPose(
            root_link=hsr_world_copy.get_kinematic_structure_entity_by_name(
                "base_footprint"
            ),
            tip_link=gripper,
            goal_pose=Pose.from_xyz_axis_angle(z=gripper_lift, reference_frame=gripper),
        ),
    )

    np.testing.assert_almost_equal(
        arm_lift_joint.position, arm_lift_before + gripper_lift, decimal=2
    )
    np.testing.assert_almost_equal(
        height_above_base(hsr_world_copy, "torso_lift_link"),
        torso_height_before + gripper_lift * torso_lift_joint.multiplier,
        decimal=2,
    )


def test_lifting_the_head_drives_the_joint_the_torso_mimics(
    hsr_world_copy: World, default_hsr_pose: dict[str, float]
) -> None:
    JointState.from_str_dict(default_hsr_pose, hsr_world_copy).apply_to(hsr_world_copy)
    torso_lift_joint: ActiveConnection1DOF = hsr_world_copy.get_connection_by_name(
        HSRBJoint.TORSO_LIFT
    )
    arm_lift_joint: ActiveConnection1DOF = hsr_world_copy.get_connection_by_name(
        HSRBJoint.ARM_LIFT
    )
    head = hsr_world_copy.get_kinematic_structure_entity_by_name("head_pan_link")
    arm_lift_before = arm_lift_joint.position
    head_height_before = height_above_base(hsr_world_copy, "head_pan_link")
    head_lift = 0.15

    run_until_reached(
        hsr_world_copy,
        CartesianPose(
            root_link=hsr_world_copy.get_kinematic_structure_entity_by_name(
                "base_footprint"
            ),
            tip_link=head,
            goal_pose=Pose.from_xyz_axis_angle(z=head_lift, reference_frame=head),
        ),
    )

    np.testing.assert_almost_equal(
        height_above_base(hsr_world_copy, "head_pan_link"),
        head_height_before + head_lift,
        decimal=2,
    )
    np.testing.assert_almost_equal(
        arm_lift_joint.position,
        arm_lift_before + head_lift / torso_lift_joint.multiplier,
        decimal=2,
    )

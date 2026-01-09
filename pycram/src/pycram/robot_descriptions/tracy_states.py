from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy

from ..datastructures.enums import (
    StaticJointState,
    Arms,
    GripperState as GripperStateEnum,
)
from ..joint_state import JointState, ArmState, GripperState, JointStateManager

right_park = ArmState(
    name=PrefixedName("tracy", "right_park"),
    joint_names=[
        "right_shoulder_pan_joint",
        "right_shoulder_lift_joint",
        "right_elbow_joint",
        "right_wrist_1_joint",
        "right_wrist_2_joint",
        "right_wrist_3_joint",
    ],
    joint_positions=[3.0, -2.1, -1.57, 0.5, 1.57, 0.0],
    state_type=StaticJointState.Park,
    arm=Arms.RIGHT,
)

left_park = ArmState(
    name=PrefixedName("tracy", "left_park"),
    joint_names=[
        "left_shoulder_pan_joint",
        "left_shoulder_lift_joint",
        "left_elbow_joint",
        "left_wrist_1_joint",
        "left_wrist_2_joint",
        "left_wrist_3_joint",
    ],
    joint_positions=[3.0, -1.0, 1.2, -0.5, 1.57, 0.0],
    state_type=StaticJointState.Park,
    arm=Arms.LEFT,
)

both_park = ArmState(
    name=PrefixedName("tracy", "both_park"),
    joint_names=[
        "left_shoulder_pan_joint",
        "left_shoulder_lift_joint",
        "left_elbow_joint",
        "left_wrist_1_joint",
        "left_wrist_2_joint",
        "left_wrist_3_joint",
        "right_shoulder_pan_joint",
        "right_shoulder_lift_joint",
        "right_elbow_joint",
        "right_wrist_1_joint",
        "right_wrist_2_joint",
        "right_wrist_3_joint",
    ],
    joint_positions=[3.0, -1.0, 1.2, -0.5, 1.57, 0.0, 3.0, -2.1, -1.57, 0.5, 1.57, 0.0],
    state_type=StaticJointState.Park,
    arm=Arms.BOTH,
)

left_gripper_open = GripperState(
    name=PrefixedName("tracy", "left_gripper_open"),
    joint_names=[
        "left_robotiq_85_left_knuckle_joint",
    ],
    joint_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    state_type=GripperStateEnum.OPEN,
    gripper=Arms.LEFT,
)

left_gripper_close = GripperState(
    name=PrefixedName("tracy", "left_gripper_close"),
    joint_names=[
        "left_robotiq_85_left_knuckle_joint",
    ],
    joint_positions=[0.7],
    state_type=GripperStateEnum.CLOSE,
    gripper=Arms.LEFT,
)

right_gripper_open = GripperState(
    name=PrefixedName("tracy", "right_gripper_open"),
    joint_names=[
        "right_robotiq_85_left_knuckle_joint",
    ],
    joint_positions=[0.0],
    state_type=GripperStateEnum.OPEN,
    gripper=Arms.RIGHT,
)

right_gripper_close = GripperState(
    name=PrefixedName("tracy", "right_gripper_close"),
    joint_names=[
        "right_robotiq_85_left_knuckle_joint",
    ],
    joint_positions=[0.8, -0.8, -0.8, 0.8, -0.8, 0.8],
    state_type=GripperStateEnum.CLOSE,
    gripper=Arms.RIGHT,
)

# CRITICAL: Register all states with Tracy class
JointStateManager().add_joint_states(
    Tracy,
    [
        right_park,
        left_park,
        both_park,
        right_gripper_open,
        right_gripper_close,
        left_gripper_open,
        left_gripper_close,
    ],
)

import numpy as np
import pytest

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from pycram.tf_transformations import quaternion_from_euler
from typing_extensions import Optional, List

from pycram.datastructures.enums import JointType
from pycram.datastructures.pose import PoseStamped
from pycram.validation.error_checkers import (
    PoseErrorChecker,
    PositionErrorChecker,
    OrientationErrorChecker,
    RevoluteJointPositionErrorChecker,
    PrismaticJointPositionErrorChecker,
    MultiJointPositionErrorChecker,
)
from pycram.validation.goal_validator import (
    GoalValidator,
    PoseGoalValidator,
    PositionGoalValidator,
    OrientationGoalValidator,
    JointPositionGoalValidator,
    MultiJointPositionGoalValidator,
    MultiPoseGoalValidator,
    MultiPositionGoalValidator,
    MultiOrientationGoalValidator,
)


@pytest.fixture
def goal_validator_world(immutable_model_world):
    world, robot_view, context = immutable_model_world
    robot_view.root.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_quaternion(0, 0, 0)
    )
    world.get_body_by_name("milk.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_quaternion(2.2, 2, 1)
    )
    world.get_body_by_name("breakfast_cereal.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_quaternion(2.2, 1.8, 1)
    )
    world.notify_state_change()
    return world, robot_view, context


def test_single_pose_goal(goal_validator_world):
    world, robot_view, context = goal_validator_world
    pose_goal_validators = PoseGoalValidator(
        lambda: PoseStamped.from_spatial_type(
            world.get_body_by_name("milk.stl").global_pose
        )
    )
    validate_pose_goal(pose_goal_validators, world)


def test_single_pose_goal_generic(goal_validator_world):
    world, robot_view, context = goal_validator_world
    pose_goal_validators = GoalValidator(
        PoseErrorChecker(),
        lambda: PoseStamped.from_spatial_type(
            world.get_body_by_name("milk.stl").global_pose
        ),
    )
    validate_pose_goal(pose_goal_validators, world)


def validate_pose_goal(goal_validator, world):
    milk_goal_pose = PoseStamped.from_list([2.5, 2.4, 1], frame=world.root)
    goal_validator.register_goal(milk_goal_pose)
    assert not (goal_validator.goal_achieved)
    assert goal_validator.actual_percentage_of_goal_achieved == 0
    assert goal_validator.current_error.tolist()[0] == pytest.approx(0.5, abs=0.001)
    assert goal_validator.current_error.tolist()[1] == pytest.approx(0, abs=0.001)
    world.get_body_by_name("milk.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.5, 2.4, 1, reference_frame=world.root
        )
    )
    assert (
        PoseStamped.from_spatial_type(world.get_body_by_name("milk.stl").global_pose)
        == milk_goal_pose,
    )
    assert goal_validator.goal_achieved
    assert goal_validator.actual_percentage_of_goal_achieved == 1
    assert goal_validator.current_error.tolist()[0] == pytest.approx(0, abs=0.001)
    assert goal_validator.current_error.tolist()[1] == pytest.approx(0, abs=0.001)


def test_single_position_goal_generic(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = GoalValidator(
        PositionErrorChecker(),
        lambda: PoseStamped.from_spatial_type(
            world.get_body_by_name("breakfast_cereal.stl").global_pose
        ).position.to_list(),
    )
    validate_position_goal(goal_validator, world)


def test_single_position_goal(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = PositionGoalValidator(
        lambda: PoseStamped.from_spatial_type(
            world.get_body_by_name("breakfast_cereal.stl").global_pose
        ).position.to_list()
    )
    validate_position_goal(goal_validator, world)


def validate_position_goal(goal_validator, world):
    cereal_goal_position = [3, 1.8, 1]
    goal_validator.register_goal(cereal_goal_position)
    assert not (goal_validator.goal_achieved)
    assert goal_validator.actual_percentage_of_goal_achieved == 0
    assert float(goal_validator.current_error[0]) == pytest.approx(0.8)
    world.get_body_by_name("breakfast_cereal.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            3, 1.8, 1, reference_frame=world.root
        )
    )
    assert (
        PoseStamped.from_spatial_type(
            world.get_body_by_name("breakfast_cereal.stl").global_pose
        ).position.to_list()
        == cereal_goal_position,
    )
    assert goal_validator.goal_achieved
    assert goal_validator.actual_percentage_of_goal_achieved == 1
    assert goal_validator.current_error == 0


def test_single_orientation_goal_generic(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = GoalValidator(
        OrientationErrorChecker(),
        lambda: PoseStamped.from_spatial_type(
            world.get_body_by_name("breakfast_cereal.stl").global_pose
        ).orientation.to_list(),
    )
    validate_orientation_goal(goal_validator, world)


def test_single_orientation_goal(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = OrientationGoalValidator(
        lambda: PoseStamped.from_spatial_type(
            world.get_body_by_name("breakfast_cereal.stl").global_pose
        ).orientation.to_list()
    )
    validate_orientation_goal(goal_validator, world)


def validate_orientation_goal(goal_validator, world):
    cereal_goal_orientation = quaternion_from_euler(0, 0, np.pi / 2)
    goal_validator.register_goal(cereal_goal_orientation)
    assert not (goal_validator.goal_achieved)
    assert goal_validator.actual_percentage_of_goal_achieved == 0
    assert goal_validator.current_error == [np.pi / 2]
    world.get_body_by_name("breakfast_cereal.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            quat_x=cereal_goal_orientation[0],
            quat_y=cereal_goal_orientation[1],
            quat_z=cereal_goal_orientation[2],
            quat_w=cereal_goal_orientation[3],
            reference_frame=world.root,
        )
    )
    for v1, v2 in zip(
        PoseStamped.from_spatial_type(
            world.get_body_by_name("breakfast_cereal.stl").global_pose
        ).orientation.to_list(),
        cereal_goal_orientation,
    ):
        assert v1 == pytest.approx(v2, abs=0.001)
    assert goal_validator.goal_achieved
    assert goal_validator.actual_percentage_of_goal_achieved == pytest.approx(
        1, abs=0.001
    )
    assert goal_validator.current_error.tolist()[0] == pytest.approx(0, abs=0.001)


def test_single_revolute_joint_position_goal_generic(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = GoalValidator(
        RevoluteJointPositionErrorChecker(),
        lambda name: world.state[world.get_degree_of_freedom_by_name(name).id].position,
    )
    validate_revolute_joint_position_goal(goal_validator, world=world)


def test_single_revolute_joint_position_goal(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = JointPositionGoalValidator(
        lambda name: world.state[world.get_degree_of_freedom_by_name(name).id].position
    )
    validate_revolute_joint_position_goal(goal_validator, JointType.REVOLUTE, world)


def validate_revolute_joint_position_goal(
    goal_validator, joint_type: Optional[JointType] = None, world=None
):
    goal_joint_position = -np.pi / 8
    joint_name = "l_shoulder_lift_joint"
    if joint_type is not None:
        goal_validator.register_goal(goal_joint_position, joint_type, joint_name)
    else:
        goal_validator.register_goal(goal_joint_position, joint_name)
    assert not goal_validator.goal_achieved
    assert goal_validator.actual_percentage_of_goal_achieved == 0
    assert goal_validator.current_error == abs(goal_joint_position)

    for percent in [0.5, 1]:
        world.state[
            world.get_degree_of_freedom_by_name("l_shoulder_lift_joint").id
        ].position = (goal_joint_position * percent)
        assert (
            world.state[
                world.get_degree_of_freedom_by_name("l_shoulder_lift_joint").id
            ].position
            == goal_joint_position * percent,
        )
        if percent == 1:
            assert goal_validator.goal_achieved
        else:
            assert not (goal_validator.goal_achieved)
        assert goal_validator.actual_percentage_of_goal_achieved == pytest.approx(
            percent, abs=0.001
        )
        assert goal_validator.current_error.tolist()[0] == pytest.approx(
            abs(goal_joint_position) * (1 - percent), abs=0.001
        )


def test_single_prismatic_joint_position_goal_generic(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = GoalValidator(
        PrismaticJointPositionErrorChecker(),
        lambda name: world.state[world.get_degree_of_freedom_by_name(name).id].position,
    )
    validate_prismatic_joint_position_goal(goal_validator, world=world)


def test_single_prismatic_joint_position_goal(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = JointPositionGoalValidator(
        lambda name: world.state[world.get_degree_of_freedom_by_name(name).id].position
    )
    validate_prismatic_joint_position_goal(goal_validator, JointType.PRISMATIC, world)


def validate_prismatic_joint_position_goal(
    goal_validator, joint_type: Optional[JointType] = None, world=None
):
    goal_joint_position = 0.2
    torso = "torso_lift_joint"
    achieved_percentage = [0.46946, 1]
    if joint_type is not None:
        goal_validator.register_goal(goal_joint_position, joint_type, torso)
    else:
        goal_validator.register_goal(goal_joint_position, torso)
    assert not (goal_validator.goal_achieved)
    assert goal_validator.actual_percentage_of_goal_achieved == 0
    assert goal_validator.current_error == 0.1885

    for percent, achieved_percentage in zip([0.5, 1], achieved_percentage):
        world.state[
            world.get_degree_of_freedom_by_name("torso_lift_joint").id
        ].position = (goal_joint_position * percent)
        assert (
            world.state[
                world.get_degree_of_freedom_by_name("torso_lift_joint").id
            ].position
            == goal_joint_position * percent,
        )
        if percent == 1:
            assert goal_validator.goal_achieved
        else:
            assert not (goal_validator.goal_achieved)
        assert goal_validator.actual_percentage_of_goal_achieved == pytest.approx(
            achieved_percentage, abs=0.001
        )
        assert goal_validator.current_error.tolist()[0] == pytest.approx(
            0.2 * (1 - percent), abs=0.01
        )


def test_multi_joint_goal_generic(goal_validator_world):
    world, robot_view, context = goal_validator_world
    joint_types = [JointType.PRISMATIC, JointType.REVOLUTE]
    goal_validator = GoalValidator(
        MultiJointPositionErrorChecker(joint_types),
        lambda x: [
            world.state[world.get_degree_of_freedom_by_name(name).id].position
            for name in x
        ],
    )
    validate_multi_joint_goal(goal_validator, world=world)


def test_multi_joint_goal(goal_validator_world):
    world, robot_view, context = goal_validator_world
    joint_types = [JointType.PRISMATIC, JointType.REVOLUTE]
    goal_validator = MultiJointPositionGoalValidator(
        lambda x: [
            world.state[world.get_degree_of_freedom_by_name(name).id].position
            for name in x
        ]
    )
    validate_multi_joint_goal(goal_validator, joint_types, world)


def validate_multi_joint_goal(
    goal_validator, joint_types: Optional[List[JointType]] = None, world=None
):
    goal_joint_positions = np.array([0.2, -np.pi / 4])
    achieved_percentage = [0.48474, 1]
    joint_names = ["torso_lift_joint", "l_shoulder_lift_joint"]
    if joint_types is not None:
        goal_validator.register_goal(goal_joint_positions, joint_types, joint_names)
    else:
        goal_validator.register_goal(goal_joint_positions, joint_names)
    assert not (goal_validator.goal_achieved)
    assert goal_validator.actual_percentage_of_goal_achieved == 0
    assert np.allclose(
        goal_validator.current_error,
        np.array([0.1885, abs(-np.pi / 4)]),
        atol=0.001,
    )

    for percent, achieved_percentage in zip([0.5, 1], achieved_percentage):
        current_joint_positions = goal_joint_positions * percent
        for joint_name, joint_position in zip(joint_names, current_joint_positions):
            world.state[world.get_degree_of_freedom_by_name(joint_name).id].position = (
                joint_position
            )
        assert np.allclose(
            world.state[
                world.get_degree_of_freedom_by_name("torso_lift_joint").id
            ].position,
            current_joint_positions[0],
            atol=0.001,
        )
        assert np.allclose(
            world.state[
                world.get_degree_of_freedom_by_name("l_shoulder_lift_joint").id
            ].position,
            current_joint_positions[1],
            atol=0.001,
        )
        if percent == 1:
            assert goal_validator.goal_achieved
        else:
            assert not (goal_validator.goal_achieved)
        assert goal_validator.actual_percentage_of_goal_achieved == pytest.approx(
            achieved_percentage, abs=0.001
        )
        assert goal_validator.current_error.tolist()[0] == pytest.approx(
            0.2 * (1 - percent), abs=0.01
        )
        assert goal_validator.current_error.tolist()[1] == pytest.approx(
            abs(-np.pi / 4) * (1 - percent), abs=0.001
        )


def test_list_of_poses_goal_generic(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = GoalValidator(
        PoseErrorChecker(is_iterable=True),
        lambda: [
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ),
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ),
        ],
    )
    validate_list_of_poses_goal(goal_validator, world)


def test_list_of_poses_goal(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = MultiPoseGoalValidator(
        lambda: [
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ),
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ),
        ]
    )
    validate_list_of_poses_goal(goal_validator, world)


def validate_list_of_poses_goal(goal_validator, world):
    position_goal = [0.0, 1.0, 0.0]
    orientation_goal = np.array([0, 0, np.pi / 2])
    poses_goal = [
        PoseStamped.from_list(
            position_goal,
            quaternion_from_euler(*orientation_goal.tolist()),
            world.root,
        ),
        PoseStamped.from_list(
            position_goal,
            quaternion_from_euler(*orientation_goal.tolist()),
            world.root,
        ),
    ]
    goal_validator.register_goal(poses_goal)
    assert not (goal_validator.goal_achieved)
    assert goal_validator.actual_percentage_of_goal_achieved == 0
    assert np.allclose(
        goal_validator.current_error,
        np.array([1.0, np.pi / 2, 1.0, np.pi / 2]),
        atol=0.001,
    )

    for percent in [0.5, 1]:
        current_orientation_goal = orientation_goal * percent
        current_pose_goal = PoseStamped.from_list(
            [0.0, 1.0 * percent, 0.0],
            quaternion_from_euler(*current_orientation_goal.tolist()),
            world.root,
        )
        world.get_body_by_name("base_footprint").parent_connection.origin = (
            current_pose_goal.to_spatial_type()
        )
        assert np.allclose(
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ).position.to_list(),
            current_pose_goal.position.to_list(),
            atol=0.001,
        )
        assert np.allclose(
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ).orientation.to_list(),
            current_pose_goal.orientation.to_list(),
            atol=0.001,
        )
        if percent == 1:
            assert goal_validator.goal_achieved
        else:
            assert not (goal_validator.goal_achieved)
        assert goal_validator.actual_percentage_of_goal_achieved == pytest.approx(
            percent, abs=0.001
        )
        assert goal_validator.current_error.tolist()[0] == pytest.approx(
            1 - percent, abs=0.001
        )
        assert goal_validator.current_error.tolist()[1] == pytest.approx(
            np.pi * (1 - percent) / 2, abs=0.001
        )
        assert goal_validator.current_error.tolist()[2] == pytest.approx(
            (1 - percent), abs=0.001
        )
        assert goal_validator.current_error.tolist()[3] == pytest.approx(
            np.pi * (1 - percent) / 2, abs=0.001
        )


def test_list_of_positions_goal_generic(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = GoalValidator(
        PositionErrorChecker(is_iterable=True),
        lambda: [
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ).position.to_list(),
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ).position.to_list(),
        ],
    )
    validate_list_of_positions_goal(goal_validator, world)


def test_list_of_positions_goal(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = MultiPositionGoalValidator(
        lambda: [
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ).position.to_list(),
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ).position.to_list(),
        ]
    )
    validate_list_of_positions_goal(goal_validator, world)


def validate_list_of_positions_goal(goal_validator, world):
    position_goal = [0.0, 1.0, 0.0]
    positions_goal = [position_goal, position_goal]
    goal_validator.register_goal(positions_goal)
    assert not (goal_validator.goal_achieved)
    assert goal_validator.actual_percentage_of_goal_achieved == 0
    assert np.allclose(goal_validator.current_error, np.array([1.0, 1.0]), atol=0.001)

    for percent in [0.5, 1]:
        current_position_goal = [0.0, 1.0 * percent, 0.0]
        world.get_body_by_name("base_footprint").parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(*current_position_goal)
        )
        assert np.allclose(
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ).position.to_list(),
            current_position_goal,
            atol=0.001,
        )
        if percent == 1:
            assert goal_validator.goal_achieved
        else:
            assert not (goal_validator.goal_achieved)
        assert goal_validator.actual_percentage_of_goal_achieved == pytest.approx(
            percent, abs=0.001
        )
        assert goal_validator.current_error.tolist()[0] == pytest.approx(
            1 - percent, abs=0.001
        )
        assert goal_validator.current_error.tolist()[1] == pytest.approx(
            1 - percent, abs=0.001
        )


def test_list_of_orientations_goal_generic(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = GoalValidator(
        OrientationErrorChecker(is_iterable=True),
        lambda: [
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ).orientation.to_list(),
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ).orientation.to_list(),
        ],
    )
    validate_list_of_orientations_goal(goal_validator, world)


def test_list_of_orientations_goal(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = MultiOrientationGoalValidator(
        lambda: [
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ).orientation.to_list(),
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ).orientation.to_list(),
        ]
    )
    validate_list_of_orientations_goal(goal_validator, world)


def validate_list_of_orientations_goal(goal_validator, world):
    orientation_goal = np.array([0, 0, np.pi / 2])
    orientations_goals = [
        quaternion_from_euler(*orientation_goal.tolist()),
        quaternion_from_euler(*orientation_goal.tolist()),
    ]
    goal_validator.register_goal(orientations_goals)
    assert not (goal_validator.goal_achieved)
    assert goal_validator.actual_percentage_of_goal_achieved == 0
    assert np.allclose(
        goal_validator.current_error,
        np.array([np.pi / 2, np.pi / 2]),
        atol=0.001,
    )

    for percent in [0.5, 1]:
        current_orientation_goal = orientation_goal * percent
        world.get_body_by_name("base_footprint").parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                pitch=current_orientation_goal[0],
                roll=current_orientation_goal[1],
                yaw=current_orientation_goal[2],
            )
        )
        assert np.allclose(
            PoseStamped.from_spatial_type(
                world.get_body_by_name("base_footprint").global_pose
            ).orientation.to_list(),
            quaternion_from_euler(*current_orientation_goal.tolist()),
            atol=0.001,
        )
        if percent == 1:
            assert goal_validator.goal_achieved
        else:
            assert not (goal_validator.goal_achieved)
        assert goal_validator.actual_percentage_of_goal_achieved == pytest.approx(
            percent, abs=0.001
        )
        assert goal_validator.current_error.tolist()[0] == pytest.approx(
            np.pi * (1 - percent) / 2, abs=0.001
        )
        assert goal_validator.current_error.tolist()[1] == pytest.approx(
            np.pi * (1 - percent) / 2,
            abs=0.001,
        )


def test_list_of_revolute_joint_positions_goal_generic(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = GoalValidator(
        RevoluteJointPositionErrorChecker(is_iterable=True),
        lambda x: [
            world.state[world.get_degree_of_freedom_by_name(name).id].position
            for name in x
        ],
    )
    validate_list_of_revolute_joint_positions_goal(goal_validator, world=world)


def test_list_of_revolute_joint_positions_goal(goal_validator_world):
    world, robot_view, context = goal_validator_world
    goal_validator = MultiJointPositionGoalValidator(
        lambda x: [
            world.state[world.get_degree_of_freedom_by_name(name).id].position
            for name in x
        ]
    )
    validate_list_of_revolute_joint_positions_goal(
        goal_validator, [JointType.REVOLUTE, JointType.REVOLUTE], world
    )


def validate_list_of_revolute_joint_positions_goal(
    goal_validator, joint_types: Optional[List[JointType]] = None, world=None
):
    goal_joint_position = -np.pi / 4
    goal_joint_positions = np.array([goal_joint_position, goal_joint_position])
    joint_names = ["l_shoulder_lift_joint", "r_shoulder_lift_joint"]
    if joint_types is not None:
        goal_validator.register_goal(goal_joint_positions, joint_types, joint_names)
    else:
        goal_validator.register_goal(goal_joint_positions, joint_names)
    assert not (goal_validator.goal_achieved)
    assert goal_validator.actual_percentage_of_goal_achieved == 0
    assert np.allclose(
        goal_validator.current_error,
        np.array([abs(goal_joint_position), abs(goal_joint_position)]),
        atol=0.001,
    )

    for percent in [0.5, 1]:
        current_joint_position = goal_joint_positions * percent
        for joint_name, joint_position in zip(joint_names, current_joint_position):
            world.state[world.get_degree_of_freedom_by_name(joint_name).id].position = (
                joint_position
            )
        world.notify_state_change()
        for joint_name, joint_position in zip(joint_names, current_joint_position):
            assert np.allclose(
                world.state[
                    world.get_degree_of_freedom_by_name(joint_name).id
                ].position,
                joint_position,
                atol=0.001,
            )
        if percent == 1:
            assert goal_validator.goal_achieved
        else:
            assert not (goal_validator.goal_achieved)
        assert goal_validator.actual_percentage_of_goal_achieved == pytest.approx(
            percent, abs=0.001
        )
        assert goal_validator.current_error.tolist()[0] == pytest.approx(
            abs(goal_joint_position) * (1 - percent),
            abs=0.001,
        )
        assert goal_validator.current_error.tolist()[1] == pytest.approx(
            abs(goal_joint_position) * (1 - percent), abs=0.001
        )

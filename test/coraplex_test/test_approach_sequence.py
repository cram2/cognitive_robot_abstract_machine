import numpy as np
import pytest

from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import (
    Pose,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures

BOX_SCALE = Scale(0.1, 0.2, 0.3)
"""
Extents of the box the approach sequence has to clear.
"""


@pytest.fixture
def boxed_pr2_world(simple_pr2_context):
    """
    A PR2 next to a graspable box of known extents, a meter above the world root.
    """
    world, robot, context = simple_pr2_context
    with world.modify_world():
        box = Body(
            name=PrefixedName("approach_box"),
            collision=ShapeCollection([Box(scale=BOX_SCALE)]),
        )
        connection = Connection6DoF.create_with_dofs(world, world.root, box)
        world.add_connection(connection)
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            1, 0, 1, reference_frame=world.root
        )
        graspable = Milk(root=box)
        world.add_semantic_annotation(graspable)
    return world, robot, graspable


def grasp_at_origin(graspable) -> GraspCandidate:
    """
    :param graspable: The annotation of the object to grasp.
    :return: A grasp at the object's origin, approaching along the body's x-axis.
    """
    return GraspCandidate.from_body_origin(graspable)


def grasp_from_above(graspable) -> GraspCandidate:
    """
    :param graspable: The annotation of the object to grasp.
    :return: A grasp on the top face of the box, reached straight down with the fingers
        closing along the body's x-axis, so its own z-axis lies flat.
    """
    return GraspCandidate(
        graspable,
        Pose(
            position=Vector3(0, 0, BOX_SCALE.z / 2).to_point3(),
            orientation=RotationMatrix.from_vectors(
                x=Vector3.NEGATIVE_Z(), y=Vector3.X()
            ).to_quaternion(),
            reference_frame=graspable.root,
        ),
    )


# %% approach sequences


def test_pre_grasp_pose_clears_the_body_it_grasps(boxed_pr2_world):
    """
    A grasp at the body's own origin has to be approached from outside the body, so the
    pre-grasp pose stands off by half of it plus the clearance.
    """
    _, robot, graspable = boxed_pr2_world
    action = HasApproachesGraspPoses()

    origin_grasp = grasp_at_origin(graspable)
    poses = action.grasp_pose_sequence(
        origin_grasp.grasp_pose,
        robot.left_arm.end_effector,
        origin_grasp,
    )

    expected_standoff = BOX_SCALE.x / 2 + action.approach_clearance
    np.testing.assert_allclose(
        poses.pre_grasp.to_np()[:3, 3], [-expected_standoff, 0, 0], atol=1e-9
    )


def test_pre_grasp_pose_of_a_surface_grasp_only_adds_the_clearance(boxed_pr2_world):
    """
    A grasp already on the body's surface, approached from outside it, needs nothing
    beyond the clearance -- this is what lets a bowl be grasped at its rim.
    """
    _, robot, graspable = boxed_pr2_world
    action = HasApproachesGraspPoses()
    surface_grasp = grasp_from_above(graspable)

    poses = action.grasp_pose_sequence(
        surface_grasp.grasp_pose,
        robot.left_arm.end_effector,
        surface_grasp,
    )

    np.testing.assert_allclose(
        poses.pre_grasp.to_np()[:3, 3],
        [0, 0, BOX_SCALE.z / 2 + action.approach_clearance],
        atol=1e-9,
    )


def test_a_body_without_collision_is_approached_from_the_clearance_alone(
    boxed_pr2_world,
):
    """
    A body without collision geometry has no boundary for the gripper to stay out of.
    """
    world, _, _ = boxed_pr2_world
    with world.modify_world():
        body = Body(name=PrefixedName("no_collision_box"))
        world.add_connection(FixedConnection(parent=world.root, child=body))
        graspable = Milk(root=body)
        world.add_semantic_annotation(graspable)
    action = HasApproachesGraspPoses()

    assert action._approach_distance(grasp_at_origin(graspable)) == pytest.approx(
        action.approach_clearance
    )


def test_a_grasp_outside_the_body_is_approached_from_the_clearance_alone(
    boxed_pr2_world,
):
    """
    A grasp in front of the body's face is already outside it, so there is nothing to
    retrace before the clearance.
    """
    _, _, graspable = boxed_pr2_world
    action = HasApproachesGraspPoses()
    in_front = GraspCandidate(
        graspable,
        Pose(
            position=Vector3(-BOX_SCALE.x, 0, 0).to_point3(),
            reference_frame=graspable.root,
        ),
    )

    assert action._approach_distance(in_front) == pytest.approx(
        action.approach_clearance
    )


def test_a_diagonal_approach_leaves_the_body_through_its_nearest_face(
    boxed_pr2_world,
):
    """
    Approached at an angle, the gripper leaves the box through whichever face it meets
    first, here the narrow one along x.
    """
    _, _, graspable = boxed_pr2_world
    action = HasApproachesGraspPoses()
    diagonal = GraspCandidate(
        graspable,
        Pose(
            orientation=RotationMatrix.from_rpy(yaw=np.pi / 4).to_quaternion(),
            reference_frame=graspable.root,
        ),
    )

    assert action._approach_distance(diagonal) == pytest.approx(
        BOX_SCALE.x / 2 * np.sqrt(2) + action.approach_clearance
    )


def test_grasp_pose_is_the_middle_of_the_sequence(boxed_pr2_world):
    _, robot, graspable = boxed_pr2_world
    end_effector = robot.left_arm.end_effector
    grasp = grasp_at_origin(graspable)

    approach = HasApproachesGraspPoses()
    poses = approach.grasp_pose_sequence(grasp.grasp_pose, end_effector, grasp)

    np.testing.assert_allclose(
        poses.grasp.to_np(),
        end_effector.tool_frame_goal(grasp.grasp_pose).to_np(),
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "grasp_on",
    [grasp_at_origin, grasp_from_above],
    ids=["approached-sideways", "approached-from-above"],
)
def test_retreat_pose_rises_along_the_world_z_axis(boxed_pr2_world, grasp_on):
    """
    The object is lifted straight up off its support, whichever way the grasp frame is
    turned, so a grasp taken from above does not drag it sideways.
    """
    world, robot, graspable = boxed_pr2_world
    action = HasApproachesGraspPoses()
    grasp = grasp_on(graspable)

    poses = action.grasp_pose_sequence(
        grasp.grasp_pose,
        robot.left_arm.end_effector,
        grasp,
    )

    world_P_grasp = world.transform(
        grasp.grasp_pose.to_homogeneous_matrix(), world.root
    ).to_np()
    world_P_retreat = world.transform(
        poses.retreat.to_homogeneous_matrix(), world.root
    ).to_np()
    np.testing.assert_allclose(
        world_P_retreat[:3, 3] - world_P_grasp[:3, 3],
        [0, 0, action.retreat_distance],
        atol=1e-9,
    )


def test_retreat_pose_keeps_the_grasp_orientation(boxed_pr2_world):
    _, robot, graspable = boxed_pr2_world

    approach = HasApproachesGraspPoses()
    origin_grasp = grasp_at_origin(graspable)
    poses = approach.grasp_pose_sequence(
        origin_grasp.grasp_pose,
        robot.left_arm.end_effector,
        origin_grasp,
    )

    np.testing.assert_allclose(
        poses.retreat.to_rotation_matrix().to_np(),
        poses.grasp.to_rotation_matrix().to_np(),
        atol=1e-9,
    )


def test_sequence_without_a_body_stands_off_by_the_clearance_alone(boxed_pr2_world):
    _, robot, graspable = boxed_pr2_world
    action = HasApproachesGraspPoses()

    poses = action.grasp_pose_sequence(
        grasp_at_origin(graspable).grasp_pose, robot.left_arm.end_effector
    )

    np.testing.assert_allclose(
        poses.pre_grasp.to_np()[:3, 3], [-action.approach_clearance, 0, 0], atol=1e-9
    )


# %% the same sequence for grippers of different conventions


def _assert_the_gripper_approaches_along_the_grasp(world, end_effector) -> None:
    """
    Assert that the grasp goal turns `end_effector`'s own approach axis onto the grasp
    frame's x-axis, and that the pre-grasp pose lies back along it.
    """
    graspable = Milk(root=world.root)
    grasp = grasp_at_origin(graspable)
    action = HasApproachesGraspPoses()

    poses = action.grasp_pose_sequence(grasp.grasp_pose, end_effector, grasp)

    root_R_tool = poses.grasp.to_rotation_matrix().to_np()[:3, :3]
    np.testing.assert_allclose(
        root_R_tool @ end_effector.approach_axis.to_np()[:3], [1, 0, 0], atol=1e-9
    )
    np.testing.assert_allclose(
        poses.pre_grasp.to_np()[:3, 3],
        [-action._approach_distance(grasp), 0, 0],
        atol=1e-9,
    )


def test_a_pr2_gripper_approaches_along_the_grasp(boxed_pr2_world):
    world, robot, _ = boxed_pr2_world
    _assert_the_gripper_approaches_along_the_grasp(world, robot.left_arm.end_effector)


def test_a_tracy_gripper_approaches_along_the_grasp(tracy_world):
    """
    Tracy's gripper points along its tool frame's z-axis rather than its x-axis, and the
    sequence has to follow it all the same.
    """
    tracy = tracy_world.get_semantic_annotations_by_type(Tracy)[0]
    _assert_the_gripper_approaches_along_the_grasp(
        tracy_world, tracy.left_arm.end_effector
    )

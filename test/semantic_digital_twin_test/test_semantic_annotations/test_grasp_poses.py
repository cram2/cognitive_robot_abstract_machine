import numpy as np
import pytest
import trimesh

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import (
    HasGraspPoses,
    HasRootBody,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Cabinet,
    Dishwasher,
    Floor,
    Handle,
    Milk,
    Spoon,
    Table,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures

BOWL_INNER_RADIUS = 0.09
"""
Radius of the synthetic bowl's inner wall.
"""

BOWL_OUTER_RADIUS = 0.10
"""
Radius of the synthetic bowl's outer wall.
"""

BOWL_HEIGHT = 0.06
"""
Height of the synthetic bowl's wall.
"""

BOX_SCALE = Scale(0.1, 0.2, 0.3)
"""
Extents of the box body used by the default grasp pose tests.
"""


@pytest.fixture
def bowl(tmp_path) -> Bowl:
    """
    A bowl whose wall is an exact tube, so its rim radius is known by construction.
    """
    mesh = trimesh.creation.annulus(
        r_min=BOWL_INNER_RADIUS, r_max=BOWL_OUTER_RADIUS, height=BOWL_HEIGHT
    )
    mesh_path = tmp_path / "bowl.stl"
    mesh.export(mesh_path)
    shape = Mesh(origin=HomogeneousTransformationMatrix(), filename=str(mesh_path))
    body = Body(
        name=PrefixedName("bowl", prefix="grasp_poses"),
        collision=ShapeCollection([shape]),
    )
    annotation = Bowl(root=body)
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(annotation)
    return annotation


@pytest.fixture
def milk() -> Milk:
    """
    A box-shaped body, which the default implementation grasps at its origin.
    """
    body = Body(
        name=PrefixedName("milk", prefix="grasp_poses"),
        collision=ShapeCollection(
            [Box(origin=HomogeneousTransformationMatrix(), scale=BOX_SCALE)]
        ),
    )
    annotation = Milk(root=body)
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(annotation)
    return annotation


def axes_of(pose) -> np.ndarray:
    """
    :param pose: The pose to read the frame axes of.
    :return: The pose's x, y and z axis as the columns of a 3x3 array.
    """
    return pose.to_np()[:3, :3]


# %% default grasp poses


def test_default_grasp_poses_are_in_the_root_frame(milk):
    for pose in milk.grasp_poses():
        assert pose.reference_frame is milk.root


def test_default_grasp_poses_are_at_the_root_origin(milk):
    for pose in milk.grasp_poses():
        np.testing.assert_allclose(pose.to_np()[:3, 3], np.zeros(3), atol=1e-9)


def test_default_grasp_pose_count_follows_the_field(milk):
    milk.grasp_pose_count = 7
    assert len(list(milk.grasp_poses())) == 7


def test_default_grasp_poses_differ_only_in_yaw(milk):
    for pose in milk.grasp_poses():
        # A pure yaw keeps the frame's z-axis on the body's z-axis.
        np.testing.assert_allclose(axes_of(pose)[:, 2], [0, 0, 1], atol=1e-9)


def test_default_grasp_poses_approach_along_evenly_spaced_yaws(milk):
    approach_yaws = sorted(
        np.arctan2(axes_of(pose)[1, 0], axes_of(pose)[0, 0])
        for pose in milk.grasp_poses()
    )
    expected = np.linspace(0, 2 * np.pi, milk.grasp_pose_count, endpoint=False)
    np.testing.assert_allclose(
        approach_yaws, np.sort(np.arctan2(np.sin(expected), np.cos(expected)))
    )


# %% rim grasp poses


def test_bowl_grasps_sit_on_the_rim_wall(bowl):
    wall_center_radius = (BOWL_INNER_RADIUS + BOWL_OUTER_RADIUS) / 2
    for pose in bowl.grasp_poses():
        position = pose.to_np()[:3, 3]
        assert np.linalg.norm(position[:2]) == pytest.approx(
            wall_center_radius, abs=1e-3
        )


def test_bowl_grasps_sit_below_the_rim_by_the_configured_depth(bowl):
    rim_height = BOWL_HEIGHT / 2 - bowl.rim_grasp_depth
    for pose in bowl.grasp_poses():
        assert pose.to_np()[2, 3] == pytest.approx(rim_height)


def test_bowl_grasps_approach_straight_down(bowl):
    for pose in bowl.grasp_poses():
        np.testing.assert_allclose(axes_of(pose)[:, 0], [0, 0, -1], atol=1e-9)


def test_bowl_grasp_fingers_close_across_the_rim_wall(bowl):
    """
    The finger axis must be radial, so the fingers straddle the wall rather than
    pinching along it.
    """
    for pose in bowl.grasp_poses():
        position = pose.to_np()[:3, 3]
        radial = position / np.linalg.norm(position[:2])
        radial[2] = 0
        finger_axis = axes_of(pose)[:, 1]
        assert abs(float(np.dot(finger_axis, radial))) == pytest.approx(1.0, abs=1e-6)


# %% the contract itself


def test_only_annotations_that_can_be_held_offer_grasps():
    """
    A root body is not enough to be graspable: furniture has one and is not picked up.

    The mixin sits below :class:`HasRootBody` rather than above it precisely so that a
    dishwasher cannot be asked where to grasp it.
    """
    for graspable in (Bowl, Milk, Spoon, Handle):
        assert issubclass(graspable, HasGraspPoses)
    for fixed in (Dishwasher, Cabinet, Table, Floor):
        assert issubclass(fixed, HasRootBody)
        assert not issubclass(fixed, HasGraspPoses)

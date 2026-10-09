import numpy as np
from numpy.typing import NDArray
import pytest
import trimesh

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    MissingReferenceFrameError,
    NoGraspGeometry,
    ReferenceFrameMismatchError,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.grasping.grasp_candidates import (
    GraspCandidate,
    HasGraspCandidates,
)
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
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
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
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
        name=PrefixedName("bowl", prefix="grasp_candidates"),
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
        name=PrefixedName("milk", prefix="grasp_candidates"),
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


def axes_of(pose: Pose) -> NDArray[np.float64]:
    """
    :param pose: The pose to read the frame axes of.
    :return: The pose's x, y and z axis as the columns of a 3x3 array.
    """
    return pose.to_np()[:3, :3]


# %% default grasp poses


def test_default_grasp_candidates_are_in_the_root_frame(milk):
    for grasp in milk.grasp_candidates():
        assert grasp.grasp_pose.reference_frame is milk.root


def test_default_grasp_candidates_belong_to_the_annotation_that_offers_them(milk):
    for grasp in milk.grasp_candidates():
        assert grasp.graspable is milk


def test_default_grasp_candidates_are_at_the_root_origin(milk):
    for grasp in milk.grasp_candidates():
        np.testing.assert_allclose(
            grasp.grasp_pose.to_np()[:3, 3], np.zeros(3), atol=1e-9
        )


def test_default_grasp_candidate_count_follows_the_field(milk):
    milk.grasp_candidate_count = 7
    assert len(milk.grasp_candidates()) == 7


def test_default_grasp_candidates_differ_only_in_yaw(milk):
    for grasp in milk.grasp_candidates():
        # A pure yaw keeps the frame's z-axis on the body's z-axis.
        np.testing.assert_allclose(
            axes_of(grasp.grasp_pose)[:, 2], [0, 0, 1], atol=1e-9
        )


def test_default_grasp_candidates_approach_along_evenly_spaced_yaws(milk):
    approach_yaws = sorted(
        np.arctan2(axes_of(grasp.grasp_pose)[1, 0], axes_of(grasp.grasp_pose)[0, 0])
        for grasp in milk.grasp_candidates()
    )
    expected = np.linspace(0, 2 * np.pi, milk.grasp_candidate_count, endpoint=False)
    np.testing.assert_allclose(
        approach_yaws, np.sort(np.arctan2(np.sin(expected), np.cos(expected)))
    )


# %% rim grasp poses


def test_bowl_grasps_sit_on_the_rim_wall(bowl):
    wall_center_radius = (BOWL_INNER_RADIUS + BOWL_OUTER_RADIUS) / 2
    for grasp in bowl.grasp_candidates():
        position = grasp.grasp_pose.to_np()[:3, 3]
        assert np.linalg.norm(position[:2]) == pytest.approx(
            wall_center_radius, abs=1e-3
        )


def test_bowl_grasps_sit_below_the_rim_by_the_configured_depth(bowl):
    rim_height = BOWL_HEIGHT / 2 - bowl.rim_grasp_depth
    for grasp in bowl.grasp_candidates():
        assert grasp.grasp_pose.to_np()[2, 3] == pytest.approx(rim_height)


def test_bowl_grasps_approach_straight_down(bowl):
    for grasp in bowl.grasp_candidates():
        np.testing.assert_allclose(
            axes_of(grasp.grasp_pose)[:, 0], [0, 0, -1], atol=1e-9
        )


def test_bowl_grasp_fingers_close_across_the_rim_wall(bowl):
    """
    The finger axis must be radial, so the fingers straddle the wall rather than
    pinching along it.
    """
    for grasp in bowl.grasp_candidates():
        position = grasp.grasp_pose.to_np()[:3, 3]
        radial = position / np.linalg.norm(position[:2])
        radial[2] = 0
        finger_axis = axes_of(grasp.grasp_pose)[:, 1]
        assert abs(float(np.dot(finger_axis, radial))) == pytest.approx(1.0, abs=1e-6)


# %% cutlery grasp poses


def _spoon_lying_along(length_axis: int) -> Spoon:
    """
    :param length_axis: The axis of its own frame the spoon lies along, 0 for x and 1
        for y.
    :return: A spoon whose collision box is long along that axis.
    """
    extents = [0.02, 0.02, 0.01]
    extents[length_axis] = 0.15
    body = Body(
        name=PrefixedName(f"spoon_along_{length_axis}", prefix="grasp_candidates"),
        collision=ShapeCollection(
            [Box(origin=HomogeneousTransformationMatrix(), scale=Scale(*extents))]
        ),
    )
    annotation = Spoon(root=body)
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(annotation)
    return annotation


@pytest.mark.parametrize("length_axis", [0, 1], ids=["along-x", "along-y"])
def test_cutlery_is_grasped_from_above_across_its_length(length_axis):
    """
    A piece of cutlery lies flat, so the fingers come down onto it and close across it,
    never along it.
    """
    [grasp] = _spoon_lying_along(length_axis).grasp_candidates()
    approach, closing = (
        axes_of(grasp.grasp_pose)[:, 0],
        axes_of(grasp.grasp_pose)[:, 1],
    )
    length_direction = np.eye(3)[length_axis]

    np.testing.assert_allclose(approach, [0, 0, -1], atol=1e-9)
    assert float(np.dot(closing, length_direction)) == pytest.approx(0, abs=1e-9)
    assert closing[2] == pytest.approx(0, abs=1e-9)


def test_cutlery_without_a_shape_offers_no_grasp():
    body = Body(name=PrefixedName("shapeless_spoon", prefix="grasp_candidates"))
    spoon = Spoon(root=body)
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(spoon)

    with pytest.raises(NoGraspGeometry):
        spoon.grasp_candidates()


# %% the frame a grasp is expressed in


def test_a_grasp_in_a_foreign_frame_is_refused(milk, bowl):
    """
    A grasp written in another frame moves with the wrong body, so the approach would
    clear the wrong geometry. Nothing downstream can tell, so it is refused here.
    """
    with pytest.raises(ReferenceFrameMismatchError):
        GraspCandidate(milk, Pose(reference_frame=bowl.root))


def test_a_grasp_without_a_frame_is_refused(milk):
    """
    A frameless pose names no body at all, so it cannot be a grasp on one.
    """
    with pytest.raises(MissingReferenceFrameError):
        GraspCandidate(milk, Pose())


def test_a_grasp_from_the_body_origin_takes_the_object_at_its_own_origin(milk):
    grasp = GraspCandidate.from_body_origin(milk)

    assert grasp.graspable is milk
    assert grasp.grasp_pose.reference_frame is milk.root
    np.testing.assert_allclose(grasp.grasp_pose.to_np(), np.eye(4), atol=1e-9)


def test_a_grasp_in_the_world_frame_follows_where_the_object_stands():
    world = World()
    world_root = Body(name=PrefixedName("map", prefix="grasp_candidates"))
    body = Body(name=PrefixedName("milk", prefix="grasp_candidates"))
    milk = Milk(root=body)
    world_T_milk = HomogeneousTransformationMatrix.from_xyz_rpy(1.0, 2.0, 0.5, yaw=0.3)
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world_root,
                child=body,
                parent_T_connection_expression=world_T_milk,
            )
        )
        world.add_semantic_annotation(milk)
    grasp = milk.grasp_candidates()[1]

    world_T_grasp = grasp.world_T_grasp

    assert world_T_grasp.reference_frame is world_root
    np.testing.assert_allclose(
        world_T_grasp.to_np(),
        world_T_milk.to_np() @ grasp.grasp_pose.to_np(),
        atol=1e-9,
    )


# %% the contract itself


def test_only_annotations_that_can_be_held_offer_grasps():
    """
    A root body is not enough to be graspable: furniture has one and is not picked up.

    The mixin sits below :class:`HasRootBody` rather than above it precisely so that a
    dishwasher cannot be asked where to grasp it.
    """
    for graspable in (Bowl, Milk, Spoon, Handle):
        assert issubclass(graspable, HasGraspCandidates)
    for fixed in (Dishwasher, Cabinet, Table, Floor):
        assert issubclass(fixed, HasRootBody)
        assert not issubclass(fixed, HasGraspCandidates)


def test_an_object_described_with_its_type_can_be_grasped(milk):
    """
    A typed description stands for an object a robot is asked to pick up, so it offers
    the default grasps of any graspable object.
    """
    described = NaturalLanguageWithTypeDescription(
        root=milk.root, description="a carton of milk", type_description="milk"
    )

    assert [
        grasp.grasp_pose.to_np().tolist() for grasp in described.grasp_candidates()
    ] == [grasp.grasp_pose.to_np().tolist() for grasp in milk.grasp_candidates()]

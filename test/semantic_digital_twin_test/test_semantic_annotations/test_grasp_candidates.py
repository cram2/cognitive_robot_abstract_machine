from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import pytest
import trimesh

from krrood.bounds import Bounds
from krrood.adapters.json_serializer import from_json, to_json
from krrood.ormatic.utils import classproperty

from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    GraspNotOnObjectError,
    NoGraspGeometry,
    ObjectHasNoWidthWithoutCollision,
)
from krrood_spatial_math.exceptions import (
    MissingReferenceFrameError,
    ReferenceFrameMismatchError,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    IsGraspable,
    HasRootBody,
    HasDrawnGraspsByDefault,
)
from semantic_digital_twin.grasping.axial_grasp import AxialGrasp
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.grasping.grasp_strategies import (
    GraspABar,
    GraspAcrossTheLength,
    GraspFromAnySide,
    GraspStrategy,
    PinchARim,
    PinchTheTopOfAWall,
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
    Mug,
    Rim,
    ShapeSortingPiece,
    Spoon,
    Table,
)
from semantic_digital_twin.part_finding.locating import LocatedFacesFinder
from semantic_digital_twin.part_finding.splitting import PartSplitter
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from krrood_spatial_math.spatial_types import (
    LineSegment3,
    Point3,
    Pose,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from ._statements import allowed_intervals

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

BOWL_RIM_DEPTH = 0.02
"""
How far below the top of the synthetic bowl's wall its rim reaches.
"""

BOX_SCALE = Scale(0.1, 0.2, 0.3)
"""
Extents of the box body used by the default grasp pose tests.
"""

PIECE_SCALE = Scale(0.03, 0.04, 0.03)
"""
Extents of the shape-sorting piece, a rectangular prism.
"""

PIECE_GRASP_HEIGHT = 0.0185
"""
How far above its bottom face the shape-sorting piece is gripped.
"""


@pytest.fixture
def bowl(tmp_path) -> Bowl:
    """
    A bowl whose wall is an exact tube, so its rim radius is known by construction,
    with its rim split off at the faces a part locator would have found it at.
    """
    wall = trimesh.creation.annulus(
        r_min=BOWL_INNER_RADIUS,
        r_max=BOWL_OUTER_RADIUS,
        height=BOWL_HEIGHT - BOWL_RIM_DEPTH,
    )
    wall.apply_translation([0.0, 0.0, -BOWL_RIM_DEPTH / 2])
    rim = trimesh.creation.annulus(
        r_min=BOWL_INNER_RADIUS, r_max=BOWL_OUTER_RADIUS, height=BOWL_RIM_DEPTH
    )
    rim.apply_translation([0.0, 0.0, (BOWL_HEIGHT - BOWL_RIM_DEPTH) / 2])
    mesh = trimesh.util.concatenate([wall, rim])
    # The rim's bottom cap lies on the wall's top cap; it stays with the wall.
    rim_faces = ~np.isclose(rim.triangles_center[:, 2], rim.bounds[0][2])
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
    PartSplitter(
        part_type=Rim,
        finder=LocatedFacesFinder(face_centers=rim.triangles_center[rim_faces]),
    ).split(annotation)
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


# %% changing how an annotation is grasped


@dataclass(eq=False)
class FixedlyGraspedHandle(Handle):
    """
    A handle that offers one grasp at its origin, whatever its shape.
    """

    def grasp_candidates(self, model_registry=None, result_pattern=None):
        return [GraspCandidate.from_body_origin(self)]


def test_an_annotation_can_offer_its_grasps_its_own_way():
    mug = _mug(with_handle=True, handle_type=FixedlyGraspedHandle)

    grasps = mug.grasp_candidates()

    assert len(grasps) == 1
    assert grasps[0].graspable is mug.handle
    np.testing.assert_allclose(grasps[0].grasp_pose.to_np(), np.eye(4), atol=1e-9)


@dataclass(eq=False)
class MilkGraspedLow(Milk):
    """
    A carton that may be grasped only at its lower half.
    """

    @classproperty
    def grasp_strategy(cls) -> GraspStrategy:
        return GraspFromAnySide(height=Bounds(0.05, 0.5))


def test_an_annotation_can_state_where_it_may_be_grasped(milk):
    carton = MilkGraspedLow(root=milk.root)
    with milk._world.modify_world():
        milk._world.add_semantic_annotation(carton)

    grasps = carton.drawn_grasp_candidates()

    assert grasps
    for grasp in grasps:
        assert grasp.grasp_pose.to_np()[2, 3] < 0.0


@dataclass(eq=False)
class MilkDrawnUniformly(HasDrawnGraspsByDefault, Milk):
    """
    A carton that offers grasps drawn uniformly from its statement while no model is
    given.
    """


def test_an_annotation_can_offer_other_grasps_while_no_model_is_given(milk):
    carton = MilkDrawnUniformly(root=milk.root)
    with milk._world.modify_world():
        milk._world.add_semantic_annotation(carton)
    half_extents = BOX_SCALE.to_np()[:3] / 2

    grasps = carton.grasp_candidates()

    assert len(grasps) == carton.grasp_candidate_count
    for grasp in grasps:
        position = grasp.grasp_pose.to_np()[:3, 3]
        assert np.all(np.abs(position) <= half_extents + 1e-9)
        assert not np.allclose(position, 0.0)


# %% rim grasp poses


def test_bowl_grasps_sit_in_the_rim_wall(bowl):
    for grasp in bowl.grasp_candidates():
        position = grasp.grasp_pose.to_np()[:3, 3]
        assert BOWL_INNER_RADIUS < np.linalg.norm(position[:2]) < BOWL_OUTER_RADIUS


def test_bowl_grasps_sit_in_the_middle_of_the_rims_height(bowl):
    lowest, highest = bowl.rim.grasp_surface().bounds
    rim_height = highest[2] - lowest[2]

    assert rim_height == pytest.approx(BOWL_RIM_DEPTH)
    for grasp in bowl.grasp_candidates():
        height_on_rim = grasp.grasp_pose.to_np()[2, 3] - lowest[2]
        assert 0.25 * rim_height - 1e-9 <= height_on_rim <= 0.75 * rim_height + 1e-9


def test_bowl_grasps_approach_from_above(bowl):
    allowed = allowed_intervals(bowl.rim.grasp_strategy.statement(bowl.rim))
    for grasp in bowl.grasp_candidates():
        approach = axes_of(grasp.grasp_pose)[:, 0]
        assert -approach[2] >= np.cos(allowed["pitch"].simple_sets[-1].upper)


def test_bowl_grasp_fingers_close_across_the_rim_wall(bowl):
    """
    The finger axis must be close to radial, so the fingers straddle the wall rather
    than pinching along it.
    """
    allowed = allowed_intervals(bowl.rim.grasp_strategy.statement(bowl.rim))
    for grasp in bowl.grasp_candidates():
        position = grasp.grasp_pose.to_np()[:3, 3]
        radial = position / np.linalg.norm(position[:2])
        radial[2] = 0
        finger_axis = axes_of(grasp.grasp_pose)[:, 1]
        assert abs(float(np.dot(finger_axis, radial))) >= np.cos(
            allowed["roll"].simple_sets[-1].upper
        ) * np.cos(allowed["pitch"].simple_sets[-1].upper)


def test_bowl_grasps_close_on_the_middle_of_the_wall(bowl):
    allowed = allowed_intervals(bowl.rim.grasp_strategy.statement(bowl.rim))
    wall_thickness = BOWL_OUTER_RADIUS - BOWL_INNER_RADIUS

    assert PinchARim().depth_unit(bowl.rim, bowl.rim.grasp_surface()) == pytest.approx(
        wall_thickness, rel=0.02
    )
    assert allowed["depth"].simple_sets[0].lower == pytest.approx(
        0.25 * wall_thickness, rel=0.02
    )
    assert allowed["depth"].simple_sets[-1].upper == pytest.approx(
        0.75 * wall_thickness, rel=0.02
    )


def test_a_wall_is_pinched_as_deep_as_asked(bowl):
    strategy = PinchARim(depth=Bounds(0.1, 0.9))
    allowed = allowed_intervals(strategy.statement(bowl.rim))
    wall_thickness = strategy.depth_unit(bowl.rim, bowl.rim.grasp_surface())

    assert allowed["depth"].simple_sets[0].lower == pytest.approx(
        strategy.depth.lower * wall_thickness
    )
    assert allowed["depth"].simple_sets[-1].upper == pytest.approx(
        strategy.depth.upper * wall_thickness
    )


def test_a_wall_is_pinched_as_tilted_and_turned_as_asked(bowl):
    pitch, roll = Bounds(0.0, 0.3), Bounds(-0.2, 0.2)
    allowed = allowed_intervals(PinchARim(pitch=pitch, roll=roll).statement(bowl.rim))

    assert allowed["pitch"].simple_sets[-1].upper == pytest.approx(pitch.upper)
    assert allowed["roll"].simple_sets[0].lower == pytest.approx(roll.lower)
    assert allowed["roll"].simple_sets[-1].upper == pytest.approx(roll.upper)


def test_a_rim_is_pinched_between_the_heights_asked(bowl):
    height = Bounds(0.1, 0.9)
    allowed = allowed_intervals(PinchARim(height=height).statement(bowl.rim))

    assert allowed["height"].simple_sets[0].lower == pytest.approx(height.lower)
    assert allowed["height"].simple_sets[-1].upper == pytest.approx(height.upper)


def test_a_container_without_a_rim_is_pinched_as_far_into_its_rim_depth_as_asked(
    bowl,
):
    bowl.rim = None
    rim_depth, rim_fraction = 0.02, Bounds(0.1, 0.9)
    allowed = allowed_intervals(
        PinchTheTopOfAWall(rim_depth=rim_depth, rim_fraction=rim_fraction).statement(
            bowl
        )
    )
    lowest, highest = bowl.grasp_surface().bounds
    rim_depth_fraction = rim_depth / float(highest[2] - lowest[2])

    assert allowed["height"].simple_sets[0].lower == pytest.approx(
        1 - rim_fraction.upper * rim_depth_fraction
    )
    assert allowed["height"].simple_sets[-1].upper == pytest.approx(
        1 - rim_fraction.lower * rim_depth_fraction
    )


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


@pytest.mark.parametrize("length_axis", [0, 1], ids=["along-x", "along-y"])
def test_drawn_cutlery_grasps_come_from_above_across_its_length(length_axis):
    """
    Grasps drawn for a piece of cutlery come straight down onto its length, close
    across it, and keep the gripper on its length.
    """
    spoon = _spoon_lying_along(length_axis)
    length_direction = np.eye(3)[length_axis]

    grasps = spoon.drawn_grasp_candidates()

    assert grasps
    for grasp in grasps:
        approach, closing = (
            axes_of(grasp.grasp_pose)[:, 0],
            axes_of(grasp.grasp_pose)[:, 1],
        )
        across_length = np.delete(grasp.grasp_pose.to_np()[:3, 3], length_axis)
        np.testing.assert_allclose(approach, [0, 0, -1], atol=1e-9)
        assert float(np.dot(closing, length_direction)) == pytest.approx(0, abs=1e-9)
        np.testing.assert_allclose(across_length, [0, 0], atol=1e-9)


def test_cutlery_is_grasped_along_its_length_as_far_as_asked():
    spoon = _spoon_lying_along(0)
    strategy = GraspAcrossTheLength(position=Bounds(0.3, 0.7))
    allowed = allowed_intervals(strategy.statement(spoon))

    assert allowed["position"].simple_sets[0].lower == pytest.approx(
        strategy.position.lower
    )
    assert allowed["position"].simple_sets[-1].upper == pytest.approx(
        strategy.position.upper
    )
    assert allowed["angle"].simple_sets[0].lower == pytest.approx(strategy.angle.lower)
    assert allowed["angle"].simple_sets[-1].upper == pytest.approx(strategy.angle.upper)


def test_cutlery_without_a_shape_offers_no_grasp():
    body = Body(name=PrefixedName("shapeless_spoon", prefix="grasp_candidates"))
    spoon = Spoon(root=body)
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(spoon)

    with pytest.raises(NoGraspGeometry):
        spoon.grasp_candidates()


# %% grasping at a handle


def _mug(with_handle: bool, handle_type: type = Handle) -> Mug:
    """
    :param with_handle: Whether the mug is given its handle.
    :param handle_type: The annotation its handle is given.
    :return: A mug whose round body is a tube and whose handle is a box sticking out
        towards positive x.
    """
    body_shape = trimesh.creation.annulus(
        r_min=BOWL_INNER_RADIUS, r_max=BOWL_OUTER_RADIUS, height=BOWL_HEIGHT
    )
    handle_shape = trimesh.creation.box(extents=[0.03, 0.01, BOWL_HEIGHT / 2])
    handle_shape.apply_translation([BOWL_OUTER_RADIUS + 0.015, 0.0, 0.0])
    body = Body(
        name=PrefixedName(f"mug_{with_handle}", prefix="grasp_candidates"),
        inertial=Inertial(),
    )
    body.collision = ShapeCollection(
        [
            Mesh.from_trimesh(
                mesh=trimesh.util.concatenate([body_shape, handle_shape]),
                origin=HomogeneousTransformationMatrix(reference_frame=body),
            )
        ],
        reference_frame=body,
    )
    mug = Mug(root=body)
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(mug)
    if with_handle:
        PartSplitter(
            part_type=handle_type,
            finder=LocatedFacesFinder(face_centers=handle_shape.triangles_center),
        ).split(mug)
    return mug


def test_an_object_with_a_handle_is_grasped_at_its_handle():
    mug = _mug(with_handle=True)

    grasps = mug.grasp_candidates()

    assert grasps
    for grasp in grasps:
        assert grasp.graspable is mug.handle
        assert grasp.grasp_pose.reference_frame is mug.handle.root


def test_the_grasped_part_of_an_object_with_a_handle_is_its_handle():
    mug = _mug(with_handle=True)

    assert mug.grasped_part() is mug.handle


def test_an_object_without_its_handle_is_grasped_as_it_offers_otherwise():
    mug = _mug(with_handle=False)

    grasps = mug.grasp_candidates()

    assert grasps
    for grasp in grasps:
        assert grasp.graspable is mug
    assert mug.grasped_part() is mug


def test_a_handle_is_grasped_along_its_bar_and_around_it_as_far_as_asked():
    handle = _mug(with_handle=True).handle
    strategy = GraspABar(position=Bounds(0.3, 0.7), angle=Bounds(0.5, 1.5))
    allowed = allowed_intervals(strategy.statement(handle))

    assert allowed["position"].simple_sets[0].lower == pytest.approx(
        strategy.position.lower
    )
    assert allowed["position"].simple_sets[-1].upper == pytest.approx(
        strategy.position.upper
    )
    assert allowed["angle"].simple_sets[0].lower == pytest.approx(strategy.angle.lower)
    assert allowed["angle"].simple_sets[-1].upper == pytest.approx(strategy.angle.upper)


def test_a_handle_split_off_a_shape_is_fixed_to_its_whole():
    mug = _mug(with_handle=True)

    connection = mug.handle.root.parent_connection

    assert connection.parent is mug.root
    assert mug.handle.root.collision


def test_a_handle_split_off_a_shape_leaves_the_whole_its_inertia():
    whole = _mug(with_handle=False)
    split = _mug(with_handle=True)

    assert split.mass == pytest.approx(whole.mass, abs=1e-5)
    assert split.root.inertial.mass == whole.root.inertial.mass
    np.testing.assert_array_equal(
        split.root.inertial.inertia.data, whole.root.inertial.inertia.data
    )
    assert split.handle.root.inertial.mass == Inertial.negligible().mass


def test_the_surface_of_an_object_includes_its_split_off_parts():
    whole = _mug(with_handle=False)
    split = _mug(with_handle=True)

    np.testing.assert_allclose(
        split.grasp_surface().bounds, whole.grasp_surface().bounds, atol=1e-9
    )
    assert split.handle.grasp_surface().bounds[0][0] >= BOWL_OUTER_RADIUS - 1e-9


# %% shape-sorting piece grasp poses


def _shape_sorting_piece(collision: ShapeCollection) -> ShapeSortingPiece:
    """
    :param collision: The piece's collision geometry.
    :return: A shape-sorting piece standing alone in a world of its own.
    """
    body = Body(
        name=PrefixedName("shape_sorting_piece", prefix="grasp_candidates"),
        collision=collision,
    )
    annotation = ShapeSortingPiece(root=body, grasp_height=PIECE_GRASP_HEIGHT)
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(annotation)
    return annotation


@pytest.fixture
def shape_sorting_piece() -> ShapeSortingPiece:
    """
    A shape-sorting piece whose box is centred on its origin, so its bottom face lies half
    its height below the origin.
    """
    return _shape_sorting_piece(
        ShapeCollection(
            [Box(origin=HomogeneousTransformationMatrix(), scale=PIECE_SCALE)]
        )
    )


def test_a_shape_sorting_piece_is_grasped_from_above_closing_along_its_own_x(
    shape_sorting_piece,
):
    """
    The fingers come down onto the piece and close along its own x, so a triangular
    prism turned with a face across x is held by that face and the opposite edge.
    """
    [grasp] = shape_sorting_piece.grasp_candidates()
    approach, closing = (
        axes_of(grasp.grasp_pose)[:, 0],
        axes_of(grasp.grasp_pose)[:, 1],
    )

    np.testing.assert_allclose(approach, [0, 0, -1], atol=1e-9)
    np.testing.assert_allclose(closing, [1, 0, 0], atol=1e-9)


def test_a_shape_sorting_piece_is_gripped_its_grasp_height_above_its_bottom_face(
    shape_sorting_piece,
):
    [grasp] = shape_sorting_piece.grasp_candidates()
    bottom_face = -PIECE_SCALE.z / 2

    np.testing.assert_allclose(
        grasp.grasp_pose.to_np()[:3, 3],
        [0, 0, bottom_face + shape_sorting_piece.grasp_height],
        atol=1e-9,
    )


def test_a_shape_sorting_piece_reaches_another_world_with_its_grasp_height(
    shape_sorting_piece,
):
    """
    A world kept in step with another one is sent its annotations as JSON, so a piece has
    to arrive able to say where it is gripped.
    """
    [block] = (
        shape_sorting_piece.root._world.get_world_model_manager().model_modification_blocks
    )
    other_world = World()
    sent = from_json(to_json(block), **WorldEntityWithIDKwargsTracker().create_kwargs())
    with other_world.modify_world():
        sent.apply(other_world)

    [arrived] = other_world.get_semantic_annotations_by_type(ShapeSortingPiece)
    assert arrived.grasp_height == shape_sorting_piece.grasp_height


def test_a_shape_sorting_piece_without_a_shape_offers_no_grasp():
    with pytest.raises(NoGraspGeometry):
        _shape_sorting_piece(ShapeCollection()).grasp_candidates()


# %% handle grasp poses


def _handle_with_bar_along(
    bar_axis: int, bar_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Handle:
    """
    :param bar_axis: The axis of its own frame the handle's bar runs along.
    :param bar_center: Where the bar's centre lies in the handle's own frame.
    :return: A handle whose collision box is long along that axis.
    """
    extents = [0.03, 0.03, 0.02]
    extents[bar_axis] = 0.2
    body = Body(
        name=PrefixedName(f"handle_along_{bar_axis}", prefix="grasp_candidates"),
        collision=ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(*bar_center),
                    scale=Scale(*extents),
                )
            ]
        ),
    )
    annotation = Handle(root=body)
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(annotation)
    return annotation


@pytest.mark.parametrize("bar_axis", [0, 1, 2], ids=["along-x", "along-y", "along-z"])
def test_a_handle_is_approached_from_every_side_across_its_bar(bar_axis) -> None:
    """
    The fingers close around the bar, so neither the approach nor the closing runs
    along it, and the bar can be approached from any of the four sides around it.
    """
    bar_direction = np.eye(3)[bar_axis]
    grasps = _handle_with_bar_along(bar_axis).grasp_candidates()

    approaches = [axes_of(grasp.grasp_pose)[:, 0] for grasp in grasps]
    closings = [axes_of(grasp.grasp_pose)[:, 1] for grasp in grasps]

    expected_approaches = [
        sign * np.eye(3)[axis]
        for axis in range(3)
        if axis != bar_axis
        for sign in (1, -1)
    ]
    assert sorted(map(tuple, np.round(approaches, 9))) == sorted(
        map(tuple, np.round(expected_approaches, 9))
    )
    for closing in closings:
        assert float(np.dot(closing, bar_direction)) == pytest.approx(0, abs=1e-9)


def test_a_handle_is_grasped_at_the_centre_of_its_bar() -> None:
    """
    A handle's frame may sit at its mount rather than on the bar, so the grasps are
    placed where the bar is.
    """
    bar_center = (0.04, 0.0, -0.01)

    for grasp in _handle_with_bar_along(1, bar_center).grasp_candidates():
        np.testing.assert_allclose(
            grasp.grasp_pose.to_np()[:3, 3], bar_center, atol=1e-9
        )


def test_a_kitchen_handle_can_be_approached_into_the_furniture(kitchen_world) -> None:
    """
    The kitchen's handle links point out of the furniture with the bar along their
    y-axis, so one grasp has to approach along their negative x-axis and close along
    their z-axis.
    """
    body = kitchen_world.get_body_by_name("sink_area_left_middle_drawer_handle")
    handle = Handle(root=body)
    with kitchen_world.modify_world():
        kitchen_world.add_semantic_annotation(handle)

    assert any(
        np.allclose(axes_of(grasp.grasp_pose)[:, 0], [-1, 0, 0], atol=1e-9)
        and np.allclose(np.abs(axes_of(grasp.grasp_pose)[:, 1]), [0, 0, 1], atol=1e-9)
        for grasp in handle.grasp_candidates()
    )


def test_a_handle_without_a_shape_offers_no_grasp() -> None:
    body = Body(name=PrefixedName("shapeless_handle", prefix="grasp_candidates"))
    handle = Handle(root=body)
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(handle)

    with pytest.raises(NoGraspGeometry):
        handle.grasp_candidates()


@pytest.mark.parametrize("bar_axis", [0, 1, 2], ids=["along-x", "along-y", "along-z"])
def test_drawn_handle_grasps_hold_the_bar_on_its_axis_and_close_across_it(bar_axis):
    """
    The gripper's tool centre point stays on the bar, only further along it or not, and
    neither the approach nor the closing runs along the bar.
    """
    bar_center = np.array([0.04, 0.0, -0.01])
    handle = _handle_with_bar_along(bar_axis, tuple(bar_center))
    bar_direction = np.eye(3)[bar_axis]
    half_length = float(max(handle.grasp_surface().extents)) / 2
    position = GraspABar().position

    grasps = handle.drawn_grasp_candidates()

    assert grasps
    for grasp in grasps:
        offset = grasp.grasp_pose.to_np()[:3, 3] - bar_center
        along_bar = float(np.dot(offset, bar_direction))
        np.testing.assert_allclose(offset - along_bar * bar_direction, 0, atol=1e-9)
        assert (2 * position.lower - 1) * half_length - 1e-9 <= along_bar
        assert along_bar <= (2 * position.upper - 1) * half_length + 1e-9
        for axis in axes_of(grasp.grasp_pose)[:, :2].T:
            assert float(np.dot(axis, bar_direction)) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("position", [0.0, 0.5, 1.0])
def test_an_axial_grasp_closes_where_along_the_axis_it_names(position):
    handle = _handle_with_bar_along(1)
    lowest, highest = handle.grasp_surface().bounds

    grasp = AxialGrasp(position=position, angle=0.0).grasp_candidate(handle)

    np.testing.assert_allclose(
        grasp.grasp_pose.to_np()[:3, 3],
        [0.0, lowest[1] + position * (highest[1] - lowest[1]), 0.0],
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "bar_axis, approach_without_turn",
    [(0, [0, 0, -1]), (1, [0, 0, -1]), (2, [0, -1, 0])],
    ids=["along-x", "along-y", "along-z"],
)
def test_an_axial_grasp_turns_its_approach_about_the_axis(
    bar_axis, approach_without_turn
):
    """
    Without a turn a horizontal bar is approached from above and a vertical one along
    the negative y-axis; a quarter turn follows the right hand about the bar.
    """
    handle = _handle_with_bar_along(bar_axis)
    bar_direction = np.eye(3)[bar_axis]

    unturned, quarter_turned = (
        AxialGrasp(position=0.5, angle=angle).grasp_candidate(handle)
        for angle in (0.0, np.pi / 2)
    )

    np.testing.assert_allclose(
        axes_of(unturned.grasp_pose)[:, 0], approach_without_turn, atol=1e-9
    )
    np.testing.assert_allclose(
        axes_of(quarter_turned.grasp_pose)[:, 0],
        np.cross(bar_direction, approach_without_turn),
        atol=1e-9,
    )


@pytest.mark.parametrize("bar_axis", [0, 1, 2], ids=["along-x", "along-y", "along-z"])
def test_the_grasp_axis_of_a_bar_runs_through_its_centre_along_its_length(bar_axis):
    bar_center = np.array([0.04, 0.0, -0.01])
    handle = _handle_with_bar_along(bar_axis, tuple(bar_center))
    half_length = float(max(handle.grasp_surface().extents)) / 2
    bar_direction = np.eye(3)[bar_axis]

    axis = handle.grasp_axis()

    np.testing.assert_allclose(
        axis.start.to_np()[:3], bar_center - half_length * bar_direction, atol=1e-9
    )
    np.testing.assert_allclose(
        axis.end.to_np()[:3], bar_center + half_length * bar_direction, atol=1e-9
    )


@dataclass(eq=False)
class HandleWithDiagonalBar(Handle):
    """
    A handle that states its grasp axis instead of leaving it to its bounding box.
    """

    def grasp_axis(self) -> LineSegment3:
        return LineSegment3(
            start=Point3(0.0, 0.0, 0.0, reference_frame=self.root),
            end=Point3(0.1, 0.1, 0.1, reference_frame=self.root),
        )


def test_an_axial_grasp_follows_the_axis_its_object_states():
    handle = _handle_with_bar_along(0)
    diagonal = HandleWithDiagonalBar(root=handle.root)
    with handle._world.modify_world():
        handle._world.add_semantic_annotation(diagonal)
    axis = diagonal.grasp_axis()
    start, end = axis.start.to_np()[:3], axis.end.to_np()[:3]
    axis_direction = axis.direction.to_np()[:3]

    grasp = AxialGrasp(position=0.25, angle=1.0).grasp_candidate(diagonal)

    np.testing.assert_allclose(
        grasp.grasp_pose.to_np()[:3, 3], start + 0.25 * (end - start), atol=1e-9
    )
    for gripper_axis in axes_of(grasp.grasp_pose)[:, :2].T:
        assert float(np.dot(gripper_axis, axis_direction)) == pytest.approx(0, abs=1e-9)


def test_an_axial_grasp_beyond_the_ends_of_the_axis_does_not_lie_on_the_object():
    handle = _handle_with_bar_along(1)
    beyond = AxialGrasp(position=1.5, angle=0.0)

    assert not beyond.reaches_surface_of(handle)
    with pytest.raises(GraspNotOnObjectError):
        beyond.grasp_candidate(handle)


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
    for graspable in (Bowl, Milk, Spoon, Handle, Mug, ShapeSortingPiece):
        assert issubclass(graspable, IsGraspable)
    for fixed in (Dishwasher, Cabinet, Table, Floor):
        assert issubclass(fixed, HasRootBody)
        assert not issubclass(fixed, IsGraspable)


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


def test_a_bowl_is_grasped_at_its_rim(bowl):
    assert bowl.grasped_part() is bowl.rim
    assert isinstance(bowl.rim, Rim)


def test_a_bowl_without_a_rim_is_grasped_as_a_whole(bowl):
    bowl.rim = None

    assert bowl.grasped_part() is bowl


# %% how wide the object stands between the fingers


def test_the_width_is_the_objects_extent_along_the_closing_axis(milk):
    """
    A grasp whose closing axis lies along one of the box's own axes meets the box across
    exactly that extent.
    """
    grasp = GraspCandidate(
        milk,
        Pose(
            orientation=RotationMatrix.from_vectors(
                x=Vector3.NEGATIVE_Z(), y=Vector3.Y()
            ).quaternion,
            reference_frame=milk.root,
        ),
    )

    assert grasp.width_along_closing_axis == pytest.approx(BOX_SCALE.y)


def test_turning_the_grasp_changes_the_width_it_meets(milk):
    """
    The same box offers a different width to fingers closing along its other axis, so
    the width follows the grasp rather than the object alone.
    """
    along_y, along_x = (
        GraspCandidate(
            milk,
            Pose(
                orientation=RotationMatrix.from_vectors(
                    x=Vector3.NEGATIVE_Z(), y=closing_axis
                ).quaternion,
                reference_frame=milk.root,
            ),
        )
        for closing_axis in (Vector3.Y(), Vector3.X())
    )

    assert along_y.width_along_closing_axis == pytest.approx(BOX_SCALE.y)
    assert along_x.width_along_closing_axis == pytest.approx(BOX_SCALE.x)


def test_a_diagonal_grasp_meets_the_box_across_its_shadow(milk):
    """
    Closing at an angle meets the box across the width of its projection onto that
    axis, since that is the widest the fingers can run into.
    """
    diagonal = GraspCandidate(
        milk,
        Pose(
            orientation=RotationMatrix.from_vectors(
                x=Vector3.NEGATIVE_Z(),
                y=Vector3(x=1 / np.sqrt(2), y=1 / np.sqrt(2), z=0),
            ).quaternion,
            reference_frame=milk.root,
        ),
    )

    expected = (BOX_SCALE.x + BOX_SCALE.y) / np.sqrt(2)
    assert diagonal.width_along_closing_axis == pytest.approx(expected)


def test_an_object_without_collision_has_no_width_to_measure():
    body = Body(name=PrefixedName("shapeless", prefix="grasp_candidates"))
    annotation = Milk(root=body)
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(annotation)

    with pytest.raises(ObjectHasNoWidthWithoutCollision):
        GraspCandidate.from_body_origin(annotation).width_along_closing_axis

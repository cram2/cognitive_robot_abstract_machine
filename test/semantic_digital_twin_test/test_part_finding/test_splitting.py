"""
Splitting a part off an object's shape, at the faces a vision-language model located it
at, which are given here.
"""

import numpy as np
import pytest
import trimesh
from numpy.typing import NDArray

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import CollisionDiffersFromVisualError
from semantic_digital_twin.pipeline.mesh_decomposition.base import MeshDecomposer
from semantic_digital_twin.part_finding.locating import LocatedFacesFinder
from semantic_digital_twin.part_finding.splitting import MeshSplit, PartSplitter
from semantic_digital_twin.pipeline.part_finding import SplitPartFromShape
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Mug,
    Pot,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Mesh, Shape
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% shapes

BODY_OUTER_RADIUS = 0.04
"""
Outer radius of the round body of the synthetic mug.
"""

HANDLE_REACH = 0.02
"""
How far the synthetic mug's handle sticks out of its body.
"""


def cup_without_handle() -> trimesh.Trimesh:
    """
    :return: A round body with a wall, open at the top.
    """
    return trimesh.creation.annulus(
        r_min=BODY_OUTER_RADIUS - 0.005, r_max=BODY_OUTER_RADIUS, height=0.1
    )


def mug() -> trimesh.Trimesh:
    """
    :return: The cup with a box-shaped handle sticking out towards positive x.
    """
    handle = trimesh.creation.box(extents=[HANDLE_REACH, 0.01, 0.06])
    handle.apply_translation([BODY_OUTER_RADIUS + HANDLE_REACH / 2, 0.0, 0.0])
    return trimesh.util.concatenate([cup_without_handle(), handle])


def handle_faces() -> NDArray[np.bool_]:
    """
    :return: For each face of the mug, whether it is a face of its handle, as a model
        would have located them.
    """
    return np.arange(len(mug().faces)) >= len(cup_without_handle().faces)


def located(belongs_to_part_mask: NDArray[np.bool_]) -> LocatedFacesFinder:
    """
    :param belongs_to_part_mask: For each face of the mug, whether it belongs to the part.
    :return: Finds the part at those faces.
    """
    return LocatedFacesFinder(face_centers=mug().triangles_center[belongs_to_part_mask])


# %% splitting a shape at located faces


def test_a_shape_is_split_at_the_located_faces():
    split = located(handle_faces()).split(mug())

    lowest, highest = split.part.bounds
    assert lowest[0] == pytest.approx(BODY_OUTER_RADIUS)
    assert highest[0] == pytest.approx(BODY_OUTER_RADIUS + HANDLE_REACH)
    assert split.rest.bounds[1][0] == pytest.approx(BODY_OUTER_RADIUS)
    assert len(split.part.faces) + len(split.rest.faces) == len(mug().faces)


def test_a_part_with_no_located_faces_is_not_split_off():
    assert located(np.zeros(len(mug().faces), dtype=bool)).split(mug()) is None


def test_a_part_that_is_the_whole_shape_is_not_split_off():
    assert located(np.ones(len(mug().faces), dtype=bool)).split(mug()) is None


# %% splitting a part off an annotation in a world


def annotated_mug(world: World | None = None) -> Mug:
    """
    :param world: The empty world to put the mug in; a new one if not given.
    :return: A mug, alone in its world, whose handle is still part of its body's shape.
    """
    world = world or World()
    body = Body(name=PrefixedName("mug"))
    shapes = ShapeCollection(
        [
            Mesh.from_trimesh(
                mesh=mug(), origin=HomogeneousTransformationMatrix(reference_frame=body)
            )
        ],
        reference_frame=body,
    )
    body.visual = shapes
    body.collision = shapes
    annotation = Mug(root=body)
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(annotation)
    return annotation


def split_handle(annotation: Mug) -> Handle:
    """
    :return: The handle split off the mug's shape.
    """
    return handle_splitter().split(annotation)


def handle_splitter() -> PartSplitter:
    """
    :return: Splits the handle off the shape of a mug.
    """
    return PartSplitter(part_type=Handle, finder=located(handle_faces()))


def test_a_part_is_split_off_only_the_first_time_it_is_asked_for():
    annotation = annotated_mug()
    splitter = handle_splitter()

    handle = splitter.part_of(annotation)

    assert annotation.handle is handle
    assert splitter.part_of(annotation) is handle
    assert len(annotation._world.bodies) == 2


def test_a_set_handle_field_is_not_split_again():
    annotation = annotated_mug()
    handle = split_handle(annotation)
    faces = len(annotation.root.visual.combined_mesh.faces)

    assert split_handle(annotation) is None
    assert annotation.handle is handle
    assert len(annotation._world.bodies) == 2
    assert len(annotation.root.visual.combined_mesh.faces) == faces


def test_a_field_of_many_handles_gets_every_handle_split_off():
    annotation = annotated_mug()
    world = annotation._world
    pot = Pot(root=annotation.root)
    with world.modify_world():
        world.remove_semantic_annotation(annotation)
        world.add_semantic_annotation(pot)
    bottom = mug().triangles_center[:, 2] < 0.0

    first = handle_splitter().split(pot)
    second = PartSplitter(
        part_type=Handle, finder=located(bottom & ~handle_faces())
    ).split(pot)

    assert pot.handles == [first, second]


def test_applying_the_split_again_leaves_the_world_as_it_is():
    annotation = annotated_mug()
    world = annotation._world
    step = SplitPartFromShape(whole_type=Mug, splitter=handle_splitter())

    step.apply(world)
    assert annotation.handle is not None
    faces = len(annotation.root.collision.combined_mesh.faces)
    step.apply(world)

    assert len(world.bodies) == 2
    assert len(annotation.root.collision.combined_mesh.faces) == faces


def test_rolling_back_a_split_restores_the_whole_shape():
    annotation = annotated_mug()
    world = annotation._world
    split_handle(annotation)

    world.rollback_modification_blocks()

    assert len(world.bodies) == 1
    assert annotation.handle is None
    assert len(annotation.root.visual.combined_mesh.faces) == len(mug().faces)
    assert len(annotation.root.collision.combined_mesh.faces) == len(mug().faces)
    assert annotation.root.collision.reference_frame is annotation.root


def test_a_split_is_reproduced_in_a_synchronized_world():
    annotation = annotated_mug()
    world = annotation._world
    split_handle(annotation)

    copy = World()
    for block in world.modification_history:
        kwargs = WorldEntityWithIDKwargsTracker.from_world(copy).create_kwargs()
        with copy.modify_world():
            from_json(to_json(block), **kwargs).apply(copy)

    root = copy.get_kinematic_structure_entity_by_id(annotation.root.id)
    assert len(copy.bodies) == 2
    assert root.collision.reference_frame is root
    assert len(root.collision.combined_mesh.faces) == len(
        annotation.root.collision.combined_mesh.faces
    )
    assert root.collision.combined_mesh.bounds[1][0] == pytest.approx(BODY_OUTER_RADIUS)


def test_a_part_is_split_off_a_single_body_without_annotating_it():
    annotation = annotated_mug()
    world = annotation._world

    part = handle_splitter().split_body(annotation.root)

    assert part.parent_kinematic_structure_entity is annotation.root
    assert annotation.handle is None
    assert len(world.semantic_annotations) == 1
    assert part.collision.combined_mesh.bounds[1][0] == pytest.approx(
        BODY_OUTER_RADIUS + HANDLE_REACH
    )
    assert annotation.root.collision.combined_mesh.bounds[1][0] == pytest.approx(
        BODY_OUTER_RADIUS
    )


def test_the_pieces_keep_the_color_of_the_shape():
    annotation = annotated_mug()
    red = Color(1.0, 0.0, 0.0, 1.0)
    annotation.root.visual.dye_shapes(red)

    handle = split_handle(annotation)

    for shapes in (handle.root.visual, annotation.root.visual):
        colors = shapes.combined_mesh.visual.face_colors
        assert (colors == [255, 0, 0, 255]).all()


def test_the_pieces_keep_the_colors_of_the_faces_they_were_cut_from():
    annotation = annotated_mug()
    split = MeshSplit.from_mesh(mug(), handle_faces())
    split.part.visual.face_colors = [0, 255, 0, 255]
    split.rest.visual.face_colors = [0, 0, 255, 255]
    mesh = trimesh.util.concatenate([split.part, split.rest])
    shapes = ShapeCollection(
        [
            Mesh.from_trimesh(
                mesh=mesh,
                origin=HomogeneousTransformationMatrix(reference_frame=annotation.root),
            )
        ],
        reference_frame=annotation.root,
    )
    with annotation._world.modify_world():
        annotation.root.update_shapes(visual=shapes, collision=shapes)

    handle = split_handle(annotation)

    handle_colors = handle.root.visual.combined_mesh.visual.vertex_colors
    body_colors = annotation.root.visual.combined_mesh.visual.vertex_colors
    assert (handle_colors == [0, 255, 0, 255]).all()
    assert (body_colors == [0, 0, 255, 255]).all()


def test_parts_of_the_same_type_split_off_the_same_body_share_a_name():
    annotation = annotated_mug()
    bottom = mug().triangles_center[:, 2] < 0.0
    first = handle_splitter().split_body(annotation.root)
    second = PartSplitter(
        part_type=Handle, finder=located(bottom & ~handle_faces())
    ).split_body(annotation.root)

    assert str(first.name) == str(second.name) == "mug/handle"
    assert len(annotation._world.bodies) == 3


def test_shapes_without_a_reference_frame_are_taken_relative_to_the_body():
    annotation = annotated_mug()
    shapes = ShapeCollection([Mesh.from_trimesh(mesh=mug())])

    with annotation._world.modify_world():
        annotation.root.update_shapes(visual=shapes, collision=shapes)

    assert all(
        shape.origin.reference_frame is annotation.root
        for shape in annotation.root.collision
    )


# %% bodies whose collision shapes differ from their visual shapes


class ConvexHullDecomposer(MeshDecomposer):
    """
    Decomposes a mesh into its convex hull: a stand-in for a real decomposer.
    """

    def apply_to_mesh(self, mesh: Mesh) -> list[Shape]:
        return [Mesh.from_trimesh(mesh=mesh.mesh.convex_hull, origin=mesh.origin)]

    def apply_to_mesh_and_save(self, mesh: Mesh, output_path: str) -> str:
        self.apply_to_mesh(mesh)[0].mesh.export(output_path, file_type="obj")
        return output_path


def mug_with_own_collision() -> Mug:
    """
    :return: The mug, colliding as its convex hull instead of as it looks.
    """
    annotation = annotated_mug()
    body = annotation.root
    hull = ShapeCollection(
        [
            Mesh.from_trimesh(
                mesh=mug().convex_hull,
                origin=HomogeneousTransformationMatrix(reference_frame=body),
            )
        ],
        reference_frame=body,
    )
    with annotation._world.modify_world():
        body.update_shapes(visual=body.visual, collision=hull)
    return annotation


def test_a_body_whose_collision_differs_from_how_it_looks_is_not_split():
    annotation = mug_with_own_collision()

    with pytest.raises(CollisionDiffersFromVisualError):
        split_handle(annotation)

    assert annotation.handle is None
    assert len(annotation._world.bodies) == 1


def test_a_decomposer_gives_the_pieces_new_collision_shapes():
    annotation = mug_with_own_collision()
    splitter = PartSplitter(
        part_type=Handle,
        finder=located(handle_faces()),
        decomposer=ConvexHullDecomposer(),
    )

    handle = splitter.split(annotation)

    for body in (handle.root, annotation.root):
        assert body.collision.combined_mesh.is_convex
    assert annotation.root.collision.combined_mesh.bounds[1][0] == pytest.approx(
        BODY_OUTER_RADIUS
    )


def test_splitting_before_decomposing_decomposes_each_piece():
    annotation = annotated_mug()
    world = annotation._world

    SplitPartFromShape(whole_type=Mug, splitter=handle_splitter()).apply(world)
    ConvexHullDecomposer().apply(world)

    assert annotation.handle.root.collision.combined_mesh.is_convex
    assert annotation.root.collision.combined_mesh.bounds[1][0] == pytest.approx(
        BODY_OUTER_RADIUS
    )

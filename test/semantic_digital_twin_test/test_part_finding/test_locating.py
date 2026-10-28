"""
Locating the parts an annotation's part-whole relationships call for in its mesh, and
giving the annotation the parts found, with stand-ins for the vision-language models
that box and mask exactly the faces they are told.
"""

from __future__ import annotations

import json

import jinja2
import numpy as np
import pytest
import trimesh
from numpy.typing import NDArray

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import UnknownPartWholeRelationshipField
from semantic_digital_twin.part_finding.annotation_part_finder import (
    AnnotationPartFinder,
)
from semantic_digital_twin.pipeline.part_finding import FindParts
from semantic_digital_twin.part_finding.relationships import PartRelationshipInspector
from semantic_digital_twin.part_finding.locating import (
    KnownPartLocator,
    LocatedFacesFinder,
    LocatedPart,
    PartQuery,
)
from semantic_digital_twin.part_finding.vision_language import (
    FilledBoxSegmenter,
    KnownPartBoxDetector,
    PartBoxPrompt,
    VisionLanguagePartLocator,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasHandle,
    HasHandles,
    HasRootBody,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    DiningTable,
    Handle,
    Leg,
    Mug,
    Pot,
    Rim,
)
from semantic_digital_twin.spatial_computations.mesh_rendering import (
    CameraRing,
    ViewDirection,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% shapes whose parts every camera of a full ring sees


def mug_with_handle() -> tuple[trimesh.Trimesh, NDArray[np.int64]]:
    """
    :return: A cup and, two centimeters apart from it so that none of its faces is hidden, a
        box-shaped handle; and the indices of the handle's faces.
    """
    cup = trimesh.creation.annulus(r_min=0.035, r_max=0.04, height=0.1)
    handle = trimesh.creation.box(extents=[0.02, 0.01, 0.06])
    handle.apply_translation([0.07, 0.0, 0.0])
    mesh = trimesh.util.concatenate([cup, handle])
    return mesh, np.arange(len(cup.faces), len(mesh.faces))


def pot_with_handles() -> tuple[trimesh.Trimesh, list[NDArray[np.int64]]]:
    """
    :return: A pot with a large handle on one side and a small one on the other, both
        two centimeters apart from it so that none of its faces is hidden; and the
        indices of each handle's faces, the large one's first.
    """
    pot = trimesh.creation.annulus(r_min=0.035, r_max=0.04, height=0.1)
    large = trimesh.creation.box(extents=[0.02, 0.01, 0.06])
    large.apply_translation([0.07, 0.0, 0.0])
    small = trimesh.creation.box(extents=[0.01, 0.01, 0.03])
    small.apply_translation([-0.065, 0.0, 0.0])
    mesh = trimesh.util.concatenate([pot, large, small])
    first = len(pot.faces)
    second = first + len(large.faces)
    return mesh, [np.arange(first, second), np.arange(second, len(mesh.faces))]


def table_with_legs() -> tuple[trimesh.Trimesh, list[NDArray[np.int64]]]:
    """
    :return: A table top with two legs a little below it, and the indices of each
        leg's faces.
    """
    top = trimesh.creation.box(extents=[0.6, 0.4, 0.02])
    top.apply_translation([0.0, 0.0, 0.7])
    legs = []
    for x in (-0.25, 0.25):
        leg = trimesh.creation.box(extents=[0.04, 0.04, 0.6])
        leg.apply_translation([x, 0.0, 0.35])
        legs.append(leg)
    mesh = trimesh.util.concatenate([top, *legs])
    first = len(top.faces)
    second = first + len(legs[0].faces)
    return mesh, [np.arange(first, second), np.arange(second, len(mesh.faces))]


def all_around() -> CameraRing:
    """
    :return: Cameras that see every face of the shapes above, from above and below.
    """
    return CameraRing(
        directions=[
            direction
            for elevation in (-40.0, 25.0, 65.0)
            for direction in ViewDirection.around(8, np.radians(elevation))
        ],
        resolution=CameraResolution(width=128, height=128),
    )


def stand_in_locator(
    part_faces: dict[str, list[NDArray[np.int64]]],
) -> VisionLanguagePartLocator:
    """
    :param part_faces: For each part name, the faces of each part of that name.
    :return: A locator whose detector boxes exactly those faces.
    """
    return VisionLanguagePartLocator(
        detector=KnownPartBoxDetector(part_faces=part_faces),
        segmenter=FilledBoxSegmenter(),
        cameras=all_around(),
    )


# %% the parts an annotation calls for


def test_a_mug_calls_for_one_optional_handle_and_one_optional_rim():
    relationships = {
        relationship.name: relationship
        for relationship in PartRelationshipInspector.all_of_type(Mug)
    }

    assert relationships.keys() == {"handle", "rim"}
    for name, part_type in (("handle", Handle), ("rim", Rim)):
        relationship = relationships[name]
        assert relationship.part_type is part_type
        assert relationship.part_name == name
        assert relationship.may_be_empty
        assert not relationship.holds_many


def test_has_handle_inherits_has_handles():
    assert issubclass(HasHandle, HasHandles)


def test_a_mug_exposes_its_handle_as_an_optional_singular_relationship():
    [relationship] = [
        relationship
        for relationship in PartRelationshipInspector.all_of_type(Mug)
        if relationship.name == "handle"
    ]

    assert relationship.part_type is Handle
    assert relationship.may_be_empty
    assert not relationship.holds_many


def test_a_dining_table_calls_for_any_number_of_legs():
    [relationship] = PartRelationshipInspector.all_of_type(DiningTable)

    assert relationship.part_type is Leg
    assert relationship.holds_many
    assert relationship.may_be_empty


def test_asking_for_a_relationship_the_annotation_does_not_have_fails():
    with pytest.raises(UnknownPartWholeRelationshipField):
        PartRelationshipInspector.named(Mug, "spout")


def test_a_query_survives_being_sent_as_json():
    mesh, _ = mug_with_handle()
    query = PartQuery.from_mesh(mesh, Mug, ["handle"])

    received = from_json(json.loads(json.dumps(to_json(query))))

    assert np.array_equal(received.vertices, query.vertices)
    assert np.array_equal(received.faces, query.faces)
    assert received.whole_type is Mug
    assert received.relationship_names == ["handle"]


# %% locating parts in views


def test_the_faces_of_a_handle_are_located():
    mesh, handle_faces = mug_with_handle()
    locator = stand_in_locator({"handle": [handle_faces]})

    [located] = locator.locate(PartQuery.from_mesh(mesh, Mug, ["handle"]))

    assert located.relationship_name == "handle"
    assert set(located.faces) == set(handle_faces)
    assert located.confidence == 1.0


def test_a_part_no_view_shows_is_not_located():
    mesh, _ = mug_with_handle()
    locator = stand_in_locator({})

    assert locator.locate(PartQuery.from_mesh(mesh, Mug, ["handle"])) == []


def test_of_two_handles_found_for_one_the_most_likely_is_located():
    mesh, handle_faces = pot_with_handles()
    locator = stand_in_locator({"handle": handle_faces})

    [located] = locator.locate(PartQuery.from_mesh(mesh, Mug, ["handle"]))

    assert set(located.faces) == set(handle_faces[0])


def test_each_handle_of_a_pot_is_located_on_its_own():
    mesh, handle_faces = pot_with_handles()
    locator = stand_in_locator({"handle": handle_faces})

    located = locator.locate(PartQuery.from_mesh(mesh, Pot, ["handles"]))

    assert sorted(sorted(part.faces) for part in located) == sorted(
        sorted(faces) for faces in handle_faces
    )


def test_each_of_many_parts_is_located_on_its_own():
    mesh, leg_faces = table_with_legs()
    locator = stand_in_locator({"leg": leg_faces})

    located = locator.locate(PartQuery.from_mesh(mesh, DiningTable, ["legs"]))

    assert sorted(sorted(part.faces) for part in located) == sorted(
        sorted(faces) for faces in leg_faces
    )


def test_located_faces_are_found_again_by_where_they_are():
    mesh, handle_faces = mug_with_handle()
    finder = LocatedFacesFinder(face_centers=mesh.triangles_center[handle_faces])
    shuffled = mesh.submesh([np.random.default_rng(0).permutation(len(mesh.faces))])[0]

    split = finder.split(shuffled)

    assert len(split.part.faces) == len(handle_faces)
    assert np.allclose(split.part.bounds, mesh.submesh([handle_faces])[0].bounds)


# %% giving an annotation its parts


def annotate(mesh: trimesh.Trimesh, annotation_type: type[HasRootBody]) -> HasRootBody:
    """
    :param mesh: The mesh of the annotation's root body.
    :param annotation_type: The type of the annotation.
    :return: The annotation, alone in a new world.
    """
    world = World()
    body = Body(name=PrefixedName(annotation_type.__name__.lower()))
    shapes = ShapeCollection(
        [
            Mesh.from_trimesh(
                mesh=mesh, origin=HomogeneousTransformationMatrix(reference_frame=body)
            )
        ],
        reference_frame=body,
    )
    body.visual = shapes
    body.collision = shapes
    annotation = annotation_type(root=body)
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_semantic_annotation(annotation)
    return annotation


def test_a_found_handle_becomes_the_mugs_handle():
    mesh, handle_faces = mug_with_handle()
    mug = annotate(mesh, Mug)
    locator = stand_in_locator({"handle": [handle_faces]})

    [handle] = AnnotationPartFinder(locator=locator).find_parts(mug)

    assert mug.handle is handle
    assert isinstance(handle, Handle)
    assert len(handle.root.collision.combined_mesh.faces) == len(handle_faces)
    assert len(mug.root.collision.combined_mesh.faces) == len(mesh.faces) - len(
        handle_faces
    )


def test_an_optional_part_the_locator_is_unsure_of_is_left_out():
    mesh, handle_faces = mug_with_handle()
    mug = annotate(mesh, Mug)
    finder = AnnotationPartFinder(
        locator=KnownPartLocator(
            parts=[LocatedPart("handle", handle_faces, confidence=0.3)]
        ),
        minimum_confidence=0.5,
    )

    assert finder.find_parts(mug) == []
    assert mug.handle is None
    assert len(mug._world.bodies) == 1


def test_a_part_the_annotation_has_already_is_not_looked_for():
    mesh, handle_faces = mug_with_handle()
    mug = annotate(mesh, Mug)
    AnnotationPartFinder(
        locator=stand_in_locator({"handle": [handle_faces]})
    ).find_parts(mug)
    locator = KnownPartLocator()

    assert AnnotationPartFinder(locator=locator).find_parts(mug) == []
    [query] = locator.queries
    assert query.relationship_names == ["rim"]


def test_each_located_leg_becomes_a_leg_of_the_table():
    mesh, leg_faces = table_with_legs()
    table = annotate(mesh, DiningTable)
    locator = stand_in_locator({"leg": leg_faces})

    legs = AnnotationPartFinder(locator=locator).find_parts(table)

    assert table.legs == legs
    assert len(legs) == 2
    assert len(table._world.bodies) == 3


def test_the_pipeline_step_gives_every_mug_its_handle():
    mesh, handle_faces = mug_with_handle()
    mug = annotate(mesh, Mug)
    step = FindParts(
        whole_type=Mug,
        finder=AnnotationPartFinder(
            locator=stand_in_locator({"handle": [handle_faces]})
        ),
    )

    step.apply(mug._world)

    assert isinstance(mug.handle, Handle)
    assert len(mug._world.bodies) == 2


# %% the question asked about each view


def test_the_question_names_the_part_and_the_whole():
    question = PartBoxPrompt().question("handle", "mug")
    assert question == (
        "Locate the handle of the mug in the image. Output its bounding boxes in JSON "
        'format as [{"bbox_2d": [x1, y1, x2, y2], "label": "handle"}]. If no handle is '
        "visible, output []."
    )


def test_a_question_with_an_unknown_placeholder_fails():
    prompt = PartBoxPrompt(template="Where is the {{ prat }} of the {{ whole }}?")
    with pytest.raises(jinja2.UndefinedError):
        prompt.question("handle", "mug")

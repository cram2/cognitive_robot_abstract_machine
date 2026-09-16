from __future__ import annotations

import dataclasses
import math
import shutil
from importlib.resources import files
from pathlib import Path

import pytest
import trimesh
from sqlalchemy.orm import Session

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from sqlalchemy import select
from experiments.orm.ormatic_interface import (
    Base,
    PreprocessedObjectDAO,
    RelationalCircuitExperimentShelfDAO,
)
from semantic_digital_twin.orm.ormatic_interface import (
    Sage10kObjectDAO,
    Sage10kPhysicallyBasedRenderingDAO,
    Sage10kPositionDAO,
    Sage10kRotationDAO,
    Sage10kSizeDAO,
)
from experiments.scene_generation_experiments.preprocessing.preprocess_sage10k import (
    PreprocessedObject,
    MeshBounds,
    MeshMeasurements,
    ObjectTypeClassifier,
    Sage10kPreprocessingRun,
    ShelfContents,
    ShelfMembershipClassifier,
)
from experiments.scene_generation_experiments.utils import MeshCandidate, ObjectType
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from experiments.scene_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
    RelationalCircuitExperimentShelfLayer,
)
from semantic_digital_twin.spatial_types import Point2, Pose, Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body


def _empty_world() -> tuple[World, Body]:
    """
    A fresh world with a single root body, for spawning a shelf into.
    """
    world = World()
    root = Body(name=PrefixedName(name="map"))
    with world.modify_world():
        world.add_body(root)
    return world, root


def _eg_object(
    object_id: str,
    place_id: str,
    object_type: ObjectType,
    x: float,
    y: float,
    z: float = 0.5,
    yaw: float = 0.0,
    width: float = 0.1,
    length: float = 0.1,
    height: float = 0.2,
    source_id: str | None = None,
) -> PreprocessedObject:
    return PreprocessedObject(
        id=object_id,
        room_id="room_1",
        place_id=place_id,
        object_type=object_type,
        scale=Scale(x=length, y=width, z=height),
        pose=Pose.from_xyz_rpy(
            x=x, y=y, z=z, roll=0.0, pitch=0.0, yaw=math.radians(yaw)
        ),
        source_id=source_id or f"{object_id}_src",
    )


def _shelf(
    width: float = 2.0, length: float = 2.0, yaw: float = 0.0, height: float = 2.0
) -> PreprocessedObject:
    """
    A shelf whose origin sits at z=1.0, so a matching ``MeshBounds(bottom=-1.0,
    top=1.0)`` places its base at 0.0 and its top at 2.0.
    """
    return _eg_object(
        "room_1_shelf_1",
        place_id="floor",
        object_type=ObjectType.SHELF,
        x=0.0,
        y=0.0,
        z=1.0,
        yaw=yaw,
        width=width,
        length=length,
        height=height,
    )


def _object_2d(
    object_type: ObjectType, object_id: str, x: float = 0.0, y: float = 0.0
) -> RelationalCircuitExperimentObject2D:
    return RelationalCircuitExperimentObject2D(
        object_type=object_type,
        scale=Scale(x=0.1, y=0.1, z=0.1),
        pose=Pose2D(x=x, y=y, yaw=0.0),
        source_id=object_id,
        name=object_id,
    )


def _layers_by_shelf(
    objects: list[PreprocessedObject],
) -> list[list[RelationalCircuitExperimentShelfLayer]]:
    """
    The extracted shelves' layers, one list per shelf.
    """
    return [
        shelf.layers
        for shelf in Sage10kPreprocessingRun._shelves_with_layers(
            objects,
            {
                "room_1_shelf_1_src": MeshBounds(
                    footprint_center_x=0.0, footprint_center_y=0.0, bottom=-1.0, top=1.0
                )
            },
            {"room_1_shelf_1"},
            MeshMeasurements(source_id_to_path={}),
        )
    ]


def _cache_off_center_mesh(tmp_path: Path, source_id: str) -> Path:
    """
    Write a box mesh, cached under *source_id*, whose local origin sits at a
    corner rather than at its bounding-box centre: it spans x in [0, 0.4] and
    y in [-0.1, 0.1], so its true centre is at local (0.2, 0.0).
    """
    objects_directory = tmp_path / "scene_1" / "objects"
    objects_directory.mkdir(parents=True)
    box = trimesh.creation.box(extents=[0.4, 0.2, 0.2])
    box.apply_translation([0.2, 0.0, 0.1])
    box.export(str(objects_directory / f"{source_id}.ply"))
    return objects_directory.parent


# %% ObjectTypeClassifier -- mapping raw sage10k type strings onto ObjectType


@pytest.fixture
def object_type_classifier() -> ObjectTypeClassifier:
    return ObjectTypeClassifier()


@pytest.mark.parametrize(
    "raw_type, expected_object_type",
    [
        ("book2", ObjectType.BOOK),
        ("Book2", ObjectType.BOOK),
        ("BOOKCHAIR6", ObjectType.CHAIR),
        ("bookshelf", ObjectType.SHELF),
        ("pottedplant", ObjectType.PLANT),
        ("floorlamp", ObjectType.LAMP),
        ("printer", ObjectType.PRINTER),
        ("smartphone", ObjectType.PHONE),
        ("showcase", ObjectType.DISPLAYCASE),
        ("dishwasher", ObjectType.DISHWASHER),
        ("chair_1", ObjectType.CHAIR),
    ],
)
def test_classify_maps_raw_type_to_expected_object_type(
    object_type_classifier: ObjectTypeClassifier,
    raw_type: str,
    expected_object_type: ObjectType,
) -> None:
    assert object_type_classifier.classify(raw_type) == expected_object_type


def test_classify_falls_back_to_other_for_unrecognized_type(
    object_type_classifier: ObjectTypeClassifier,
) -> None:
    assert (
        object_type_classifier.classify("xyzzy_totally_unknown_object")
        == ObjectType.OTHER
    )


# %% ShelfMembershipClassifier -- deciding which furniture names are shelf-like


@pytest.fixture
def shelf_membership_classifier() -> ShelfMembershipClassifier:
    return ShelfMembershipClassifier()


@pytest.mark.parametrize(
    "raw_type",
    [
        "shelf",
        "wallshelf",
        "rack",
        "shelving",
        "bookshelf",
        "bookcase",
        "cabinet",
        "storagecabinet",
        "sideboard",
        "console",
        "credenza",
        "SHELF",
        "BookShelf",
    ],
)
def test_shelf_like_furniture_is_recognized(
    shelf_membership_classifier: ShelfMembershipClassifier, raw_type: str
) -> None:
    """
    Any of the modelled keyword groups, in any casing, is recognized as shelf-like.
    """
    assert shelf_membership_classifier.is_shelf_like(raw_type)


@pytest.mark.parametrize(
    "raw_type",
    ["dresser", "wardrobe", "closet", "displaycase", "table", "chair", "bed", "sofa"],
)
def test_furniture_outside_the_modelled_types_is_not_classified(
    shelf_membership_classifier: ShelfMembershipClassifier, raw_type: str
) -> None:
    """
    ``is_shelf_like`` doubles as the membership gate deciding what enters training, so
    it answers ``False`` rather than a catch-all match.

    Dressers, wardrobes and display cases are storage but were deliberately left out of
    the modelled keywords, and a catch-all would pull them plus every table and chair
    back in.
    """
    assert not shelf_membership_classifier.is_shelf_like(raw_type)


# %% MeshMeasurements -- correcting a recorded position to the mesh's centre


def test_position_is_corrected_to_the_meshs_bounding_box_center(
    tmp_path: Path,
) -> None:
    """
    A sage10k object's recorded position is its mesh's local origin, which is not
    guaranteed to be that mesh's bounding-box centre, so the correction has to shift it
    there.
    """
    scene_directory = _cache_off_center_mesh(tmp_path, "book_src")
    measurements = MeshMeasurements(source_id_to_path={"book_src": scene_directory})

    corrected = measurements.corrected_position(
        source_id="book_src", position=Point2(x=1.0, y=2.0), yaw_degrees=0.0
    )

    assert corrected.is_mesh_corrected
    assert float(corrected.position.x) == pytest.approx(1.2)
    assert float(corrected.position.y) == pytest.approx(2.0)


def test_measured_footprint_center_fields_are_ordinary_floats(tmp_path: Path) -> None:
    """
    A mesh is measured with numpy, whose scalars pass for floats everywhere except at the
    database driver: PostgreSQL is handed their repr and rejects the statement, while
    SQLite accepts them, so nothing short of an explicit check catches the leak.
    """
    scene_directory = _cache_off_center_mesh(tmp_path, "book_src")
    measurements = MeshMeasurements(source_id_to_path={"book_src": scene_directory})

    bounds = measurements.bounds("book_src")

    assert type(bounds.footprint_center_x) is float
    assert type(bounds.footprint_center_y) is float


def test_position_correction_is_rotated_by_the_objects_own_yaw(
    tmp_path: Path,
) -> None:
    """
    The mesh-local offset has to be rotated into world axes by the object's own yaw, or
    a rotated object is corrected along the wrong axis.
    """
    scene_directory = _cache_off_center_mesh(tmp_path, "book_src")
    measurements = MeshMeasurements(source_id_to_path={"book_src": scene_directory})

    corrected = measurements.corrected_position(
        source_id="book_src", position=Point2(x=0.0, y=0.0), yaw_degrees=90.0
    )

    assert float(corrected.position.x) == pytest.approx(0.0, abs=1e-9)
    assert float(corrected.position.y) == pytest.approx(0.2)


def test_position_falls_back_to_the_recorded_one_without_a_cached_mesh() -> None:
    """
    An object whose mesh is not cached locally keeps its recorded position and is
    flagged as uncorrected, so the gap stays visible instead of silently mixing
    corrected and uncorrected data.
    """
    measurements = MeshMeasurements(source_id_to_path={})

    corrected = measurements.corrected_position(
        source_id="missing_src", position=Point2(x=0.3, y=0.1), yaw_degrees=0.0
    )

    assert not corrected.is_mesh_corrected
    assert float(corrected.position.x) == pytest.approx(0.3)
    assert float(corrected.position.y) == pytest.approx(0.1)


def test_mesh_is_measured_once_per_source_id(tmp_path: Path) -> None:
    """
    Many objects share one mesh asset, so measuring a mesh again for every object that
    uses it would dominate the pipeline's runtime.
    """
    scene_directory = _cache_off_center_mesh(tmp_path, "book_src")
    measurements = MeshMeasurements(source_id_to_path={"book_src": scene_directory})

    for _ in range(3):
        measurements.corrected_position(
            source_id="book_src", position=Point2(x=0.0, y=0.0), yaw_degrees=0.0
        )

    assert measurements.measured_mesh_count == 1


# %% eg_object_from_sage10k_object -- unified types and carried-through text


def _sage10k_object(
    raw_type: str = "book2",
    description: str = "A worn hardcover novel",
    place_guidance: str = "on the middle shelf",
    x: float = 0.0,
    y: float = 0.0,
    object_id: str = "book_1",
    room_id: str = "room_1",
    source_id: str = "book_src",
    place_id: str = "room_1_shelf_1",
) -> Sage10kObjectDAO:
    return Sage10kObjectDAO(
        id=object_id,
        room_id=room_id,
        type=raw_type,
        description=description,
        source="generation",
        source_id=source_id,
        place_id=place_id,
        place_guidance=place_guidance,
        mass=0.4,
        position=Sage10kPositionDAO(x=x, y=y, z=0.5),
        rotation=Sage10kRotationDAO(x=0.0, y=0.0, z=0.0),
        dimensions=Sage10kSizeDAO(height=0.2, length=0.1, width=0.05),
        pbr_parameters=Sage10kPhysicallyBasedRenderingDAO(metallic=0.0, roughness=0.5),
    )


def test_conversion_maps_the_raw_type_onto_a_unified_object_type() -> None:
    converted = PreprocessedObject.from_sage10k_object(
        _sage10k_object(raw_type="bookshelf1"),
        ObjectTypeClassifier(),
        MeshMeasurements(source_id_to_path={}),
    )

    assert converted.object_type == ObjectType.SHELF


def test_conversion_carries_the_datasets_free_text_through() -> None:
    """
    The dataset's ``description`` and ``place_guidance`` are the only natural language
    available for placement reasoning, and were previously dropped on conversion.
    """
    converted = PreprocessedObject.from_sage10k_object(
        _sage10k_object(),
        ObjectTypeClassifier(),
        MeshMeasurements(source_id_to_path={}),
    )

    assert converted.description == "A worn hardcover novel"
    assert converted.place_guidance == "on the middle shelf"


def test_conversion_corrects_the_position_and_records_that_it_did(
    tmp_path: Path,
) -> None:
    scene_directory = _cache_off_center_mesh(tmp_path, "book_src")

    converted = PreprocessedObject.from_sage10k_object(
        _sage10k_object(),
        ObjectTypeClassifier(),
        MeshMeasurements(source_id_to_path={"book_src": scene_directory}),
    )

    assert converted.position_is_mesh_corrected
    assert float(converted.pose.x) == pytest.approx(0.2)
    assert float(converted.pose.z) == pytest.approx(0.5)


def test_conversion_flags_an_object_whose_mesh_was_unavailable() -> None:
    converted = PreprocessedObject.from_sage10k_object(
        _sage10k_object(x=0.3),
        ObjectTypeClassifier(),
        MeshMeasurements(source_id_to_path={}),
    )

    assert not converted.position_is_mesh_corrected
    assert float(converted.pose.x) == pytest.approx(0.3)


# %% Sage10kPreprocessingRun._shelves_with_layers -- grouping corrected objects into ordered layers


def _rotated_shelf_and_book(local_x: float, local_y: float) -> list[PreprocessedObject]:
    """
    A shelf rotated 45 degrees in the room (wide 1.0, shallow 0.4), with one book placed
    at ``(local_x, local_y)`` in the *shelf's own* frame.
    """
    yaw_degrees = 45.0
    theta = math.radians(yaw_degrees)
    shelf = _shelf(width=1.0, length=0.4, yaw=yaw_degrees, height=0.02)
    book = _eg_object(
        "book_1",
        place_id="room_1_shelf_1",
        object_type=ObjectType.BOOK,
        x=local_x * math.cos(theta) - local_y * math.sin(theta),
        y=local_x * math.sin(theta) + local_y * math.cos(theta),
    )
    return [shelf, book]


def test_within_bounds_filter_accounts_for_shelf_rotation() -> None:
    """
    A book at the legitimate edge of a rotated shelf's wide axis, centred on its shallow
    axis, must be kept -- comparing the raw world-frame offset against the shelf's width
    and length tests the wrong axes.
    """
    layers_by_shelf = _layers_by_shelf(_rotated_shelf_and_book(0.45, 0.0))

    assert len(layers_by_shelf) == 1
    source_ids = {
        object_.source_id for layer in layers_by_shelf[0] for object_ in layer.objects
    }
    assert source_ids == {"book_1_src"}


def test_within_bounds_filter_excludes_object_outside_rotated_footprint() -> None:
    assert _layers_by_shelf(_rotated_shelf_and_book(0.0, 0.35)) == []


def test_layers_are_ordered_from_the_bottom_up() -> None:
    """
    Layer order has to follow height, since a caller reasoning about where on a shelf
    something belongs reads meaning into a layer's position in the list.

    Grouping by cluster label alone leaves the order an accident of which object
    happened to be encountered first.
    """
    objects = [
        _shelf(),
        _eg_object("top", "room_1_shelf_1", ObjectType.BOOK, x=0.0, y=0.0, z=1.5),
        _eg_object("bottom", "room_1_shelf_1", ObjectType.BOOK, x=0.1, y=0.0, z=0.2),
        _eg_object("middle", "room_1_shelf_1", ObjectType.BOOK, x=0.2, y=0.0, z=0.9),
    ]

    [layers] = _layers_by_shelf(objects)

    assert [layer.objects[0].source_id for layer in layers] == [
        "bottom_src",
        "middle_src",
        "top_src",
    ]


def test_objects_at_a_similar_height_share_one_layer() -> None:
    objects = [
        _shelf(),
        _eg_object("left", "room_1_shelf_1", ObjectType.BOOK, x=-0.2, y=0.0, z=0.50),
        _eg_object("right", "room_1_shelf_1", ObjectType.BOOK, x=0.2, y=0.0, z=0.51),
    ]

    [layers] = _layers_by_shelf(objects)

    assert len(layers) == 1
    assert {object_.source_id for object_ in layers[0].objects} == {
        "left_src",
        "right_src",
    }


def test_a_shelf_whose_own_position_was_not_corrected_yields_no_layers() -> None:
    """
    A layer records its objects' offsets from the shelf's origin, so an uncentred shelf
    position shifts every one of them.

    Such a layer would teach a circuit an arrangement nobody ever built.
    """
    shelf = dataclasses.replace(_shelf(), position_is_mesh_corrected=False)
    book = _eg_object("book_1", "room_1_shelf_1", ObjectType.BOOK, x=0.0, y=0.0)

    assert _layers_by_shelf([shelf, book]) == []


def test_objects_whose_position_was_not_corrected_are_left_out_of_layers() -> None:
    objects = [
        _shelf(),
        _eg_object("centred", "room_1_shelf_1", ObjectType.BOOK, x=0.0, y=0.0),
        dataclasses.replace(
            _eg_object("uncentred", "room_1_shelf_1", ObjectType.BOOK, x=0.2, y=0.0),
            position_is_mesh_corrected=False,
        ),
    ]

    [layers] = _layers_by_shelf(objects)

    assert [object_.source_id for layer in layers for object_ in layer.objects] == [
        "centred_src"
    ]


def test_content_orientation_is_stored_relative_to_the_shelfs_content_frame() -> None:
    """
    Contents are spawned inside a corpus built in the shelf's content frame, so a yaw
    stored in absolute terms is double-counted for every rotated shelf.
    """
    objects = [
        _shelf(yaw=90.0),
        _eg_object(
            "book_1", "room_1_shelf_1", ObjectType.BOOK, x=0.3, y=0.0, yaw=110.0
        ),
    ]

    [layers] = _layers_by_shelf(objects)

    [stored] = layers[0].objects
    assert math.degrees(float(stored.pose.yaw)) == pytest.approx(-70.0)


def test_every_object_type_is_extracted() -> None:
    objects = [
        _shelf(),
        _eg_object("book_1", "room_1_shelf_1", ObjectType.BOOK, x=0.0, y=0.0),
        _eg_object("cup_1", "room_1_shelf_1", ObjectType.CUP, x=0.1, y=0.0),
    ]

    [layers] = _layers_by_shelf(objects)

    assert {object_.source_id for layer in layers for object_ in layer.objects} == {
        "book_1_src",
        "cup_1_src",
    }


# %% Layer vertical context -- where a layer sits in its shelf


def _three_layer_shelf() -> list[PreprocessedObject]:
    """
    A shelf whose base is at 0.0 and top at 2.0, holding one book on each of three
    layers at heights 0.2, 0.8 and 1.4, listed top-down.
    """
    return [_shelf()] + [
        _eg_object(name, "room_1_shelf_1", ObjectType.BOOK, x=0.0, y=0.0, z=height)
        for name, height in [("top", 1.4), ("bottom", 0.2), ("middle", 0.8)]
    ]


def test_layer_height_is_measured_from_the_shelf_meshs_own_base() -> None:
    """
    The shelf's recorded position is its mesh's origin, not its base, so the height a
    layer sits at only follows once the mesh has been measured.
    """
    [layers] = _layers_by_shelf(_three_layer_shelf())

    assert [layer.height_above_shelf_base for layer in layers] == [
        pytest.approx(0.2),
        pytest.approx(0.8),
        pytest.approx(1.4),
    ]


def test_relative_height_places_layers_between_the_shelfs_base_and_top() -> None:
    """
    The fraction is what transfers across shelves of different sizes, so it has to run
    from zero at the base to one at the top.
    """
    [layers] = _layers_by_shelf(_three_layer_shelf())

    assert [layer.relative_height for layer in layers] == [
        pytest.approx(0.1),
        pytest.approx(0.4),
        pytest.approx(0.7),
    ]


def test_vertical_clearance_reaches_the_next_layer_up() -> None:
    [layers] = _layers_by_shelf(_three_layer_shelf())

    assert layers[0].vertical_clearance == pytest.approx(0.6)
    assert layers[1].vertical_clearance == pytest.approx(0.6)


def test_the_topmost_layers_clearance_reaches_the_shelfs_top() -> None:
    """
    Nothing stands above the topmost layer, so its clearance is the room left under the
    shelf's own ceiling -- which is what decides what still fits.
    """
    [layers] = _layers_by_shelf(_three_layer_shelf())

    assert layers[-1].vertical_clearance == pytest.approx(0.6)


def test_a_shelf_of_no_measurable_height_reads_as_sitting_at_its_base() -> None:
    """
    A degenerate mesh must not divide by zero; the absolute height stays meaningful even
    when the fraction cannot be.
    """
    objects = [
        _shelf(),
        _eg_object("book_1", "room_1_shelf_1", ObjectType.BOOK, x=0.0, y=0.0, z=1.0),
    ]

    shelves = Sage10kPreprocessingRun._shelves_with_layers(
        objects,
        {
            "room_1_shelf_1_src": MeshBounds(
                footprint_center_x=0.0, footprint_center_y=0.0, bottom=0.0, top=0.0
            )
        },
        {"room_1_shelf_1"},
        MeshMeasurements(source_id_to_path={}),
    )

    [layer] = shelves[0].layers
    assert layer.relative_height == pytest.approx(0.0)
    assert layer.height_above_shelf_base == pytest.approx(0.0)


def test_a_shelf_whose_mesh_was_never_measured_is_skipped() -> None:
    """
    Without the shelf's real base and top its layers' heights would be guesswork, so it
    contributes nothing rather than something invented.
    """
    objects = [
        _shelf(),
        _eg_object("book_1", "room_1_shelf_1", ObjectType.BOOK, x=0.0, y=0.0),
    ]

    assert (
        Sage10kPreprocessingRun._shelves_with_layers(
            objects,
            {},
            {"room_1_shelf_1"},
            MeshMeasurements(source_id_to_path={}),
        )
        == []
    )


def test_a_shelf_keeps_its_own_pose_and_measured_height() -> None:
    """
    Keeping the shelf, not just loose layers, is what preserves which layers belong
    together and in what order.
    """
    [shelf] = Sage10kPreprocessingRun._shelves_with_layers(
        _three_layer_shelf(),
        {
            "room_1_shelf_1_src": MeshBounds(
                footprint_center_x=0.0, footprint_center_y=0.0, bottom=-1.0, top=1.0
            )
        },
        {"room_1_shelf_1"},
        MeshMeasurements(source_id_to_path={}),
    )

    assert shelf.scale.z == pytest.approx(2.0)
    assert len(shelf.layers) == 3
    assert [
        object_.source_id for layer in shelf.layers for object_ in layer.objects
    ] == [
        "bottom_src",
        "middle_src",
        "top_src",
    ]


def test_a_shelfs_theme_is_the_object_type_its_objects_have_the_most_of() -> None:
    """
    A shelf's theme is derived from what is actually placed on it, so two books and one
    bottle must make the shelf book-themed -- and every layer on it must carry that same
    theme, since it is denormalized onto both.
    """
    objects = [_shelf()] + [
        _eg_object("book_1", "room_1_shelf_1", ObjectType.BOOK, x=0.0, y=0.0, z=0.5),
        _eg_object("book_2", "room_1_shelf_1", ObjectType.BOOK, x=0.2, y=0.0, z=0.5),
        _eg_object(
            "bottle_1", "room_1_shelf_1", ObjectType.BOTTLE, x=0.4, y=0.0, z=0.5
        ),
    ]

    [shelf] = Sage10kPreprocessingRun._shelves_with_layers(
        objects,
        {
            "room_1_shelf_1_src": MeshBounds(
                footprint_center_x=0.0, footprint_center_y=0.0, bottom=-1.0, top=1.0
            )
        },
        {"room_1_shelf_1"},
        MeshMeasurements(source_id_to_path={}),
    )

    assert shelf.theme_dominant_type is ObjectType.BOOK
    assert {layer.theme_dominant_type for layer in shelf.layers} == {ObjectType.BOOK}


def test_a_tied_theme_breaks_alphabetically_by_type_value() -> None:
    """
    A tie between equally-frequent types must resolve the same way every time rather
    than depend on iteration order -- broken here by the type's own, ascending value
    ("book" < "bottle").
    """
    objects = [_shelf()] + [
        _eg_object(
            "bottle_1", "room_1_shelf_1", ObjectType.BOTTLE, x=0.0, y=0.0, z=0.5
        ),
        _eg_object("book_1", "room_1_shelf_1", ObjectType.BOOK, x=0.2, y=0.0, z=0.5),
    ]

    [shelf] = Sage10kPreprocessingRun._shelves_with_layers(
        objects,
        {
            "room_1_shelf_1_src": MeshBounds(
                footprint_center_x=0.0, footprint_center_y=0.0, bottom=-1.0, top=1.0
            )
        },
        {"room_1_shelf_1"},
        MeshMeasurements(source_id_to_path={}),
    )

    assert shelf.theme_dominant_type is ObjectType.BOOK


# %% ShelfContents -- keeping only what layer extraction reads


def test_shelf_ids_come_from_the_classified_types_of_the_raw_objects(
    tmp_path: Path,
) -> None:
    """
    Which raw objects are shelves has to be settled before the dataset is read in full,
    since that is what tells the main pass whether an object may be dropped the moment
    it has been written.
    """
    objects = [
        _sage10k_object(object_id="shelf_1", raw_type="bookshelf1"),
        _sage10k_object(object_id="book_1", raw_type="book2"),
    ]
    engine = _populated_sqlite_engine(tmp_path, objects)

    with Session(engine) as session:
        contents = ShelfContents.from_raw_objects(session, ShelfMembershipClassifier())

    assert contents.shelf_ids == {"shelf_1"}


def test_relevant_source_ids_are_shelves_and_their_members(tmp_path: Path) -> None:
    """
    Only a shelf's own mesh and the meshes of whatever stands on it ever feed layer
    extraction, so an unrelated object's mesh must be left out of the source ids worth
    measuring even though it sits in the same raw dataset.
    """
    objects = [
        _sage10k_object(
            object_id="shelf_1",
            raw_type="bookshelf1",
            source_id="shelf_1_src",
            place_id="floor",
        ),
        _sage10k_object(
            object_id="book_1",
            raw_type="book2",
            source_id="book_1_src",
            place_id="shelf_1",
        ),
        _sage10k_object(
            object_id="chair_1",
            raw_type="chair2",
            source_id="chair_1_src",
            place_id="floor",
        ),
    ]
    engine = _populated_sqlite_engine(tmp_path, objects)

    with Session(engine) as session:
        contents = ShelfContents.from_raw_objects(session, ShelfMembershipClassifier())

    assert contents.relevant_source_ids == {"shelf_1_src", "book_1_src"}


def test_shelves_and_the_objects_standing_on_them_are_kept() -> None:
    shelf = _shelf()
    on_shelf = _eg_object("book_1", "room_1_shelf_1", ObjectType.BOOK, x=0.0, y=0.0)
    elsewhere = _eg_object("chair_1", "floor", ObjectType.CHAIR, x=5.0, y=5.0)
    contents = ShelfContents(shelf_ids={"room_1_shelf_1"})

    for processed_object in [shelf, on_shelf, elsewhere]:
        contents.collect(processed_object)

    assert contents.objects == [shelf, on_shelf]


def test_a_shelf_like_object_is_not_counted_as_another_shelfs_content() -> None:
    """
    An object that is itself classified as a shelf-like parent must not also be
    counted as ordinary content standing on a different shelf: the raw dataset
    records a small piece of shelf-like furniture placed on a bigger one this way,
    and treating it as passive content teaches the circuit that shelves commonly
    hold other shelves.
    """
    shelf = _shelf()
    nested_shelf_like_object = _eg_object(
        "small_shelf_1", "room_1_shelf_1", ObjectType.SHELF, x=0.0, y=0.0
    )
    book = _eg_object("book_1", "room_1_shelf_1", ObjectType.BOOK, x=0.3, y=0.0)
    shelf_ids = {"room_1_shelf_1", "small_shelf_1"}

    [extracted_shelf] = Sage10kPreprocessingRun._shelves_with_layers(
        [shelf, nested_shelf_like_object, book],
        {
            "room_1_shelf_1_src": MeshBounds(
                footprint_center_x=0.0, footprint_center_y=0.0, bottom=-1.0, top=1.0
            )
        },
        shelf_ids,
        MeshMeasurements(source_id_to_path={}),
    )

    content_source_ids = {
        object_.source_id
        for layer in extracted_shelf.layers
        for object_ in layer.objects
    }
    assert content_source_ids == {"book_1_src"}


@dataclasses.dataclass(frozen=True)
class _ObjectSnapshot:
    """
    The comparable content of an :class:`RelationalCircuitExperimentObject2D`, standing
    in for it in an equality check.

    :attr:`RelationalCircuitExperimentObject2D.pose` is a :class:`Pose2D`, whose casadi-
    backed equality falls back to identity, so two independently built shelf trees never
    compare equal even when numerically identical -- this snapshot picks out only the
    plain values that matter instead.
    """

    object_type: ObjectType
    scale: tuple[float, float, float]
    position: tuple[float, float, float]
    source_id: str


@dataclasses.dataclass(frozen=True)
class _LayerSnapshot:
    """
    The comparable content of an :class:`RelationalCircuitExperimentShelfLayer`.
    """

    theme_dominant_type: ObjectType
    height_above_shelf_base: float
    relative_height: float
    vertical_clearance: float
    objects: tuple[_ObjectSnapshot, ...]


@dataclasses.dataclass(frozen=True)
class _ShelfSnapshot:
    """
    The comparable content of an :class:`RelationalCircuitExperimentShelf`.
    """

    scale: tuple[float, float, float]
    theme_dominant_type: ObjectType
    layers: tuple[_LayerSnapshot, ...]


def _shelf_snapshot(shelf: RelationalCircuitExperimentShelf) -> _ShelfSnapshot:
    return _ShelfSnapshot(
        scale=(shelf.scale.x, shelf.scale.y, shelf.scale.z),
        theme_dominant_type=shelf.theme_dominant_type,
        layers=tuple(_layer_snapshot(layer) for layer in shelf.layers),
    )


def _layer_snapshot(layer: RelationalCircuitExperimentShelfLayer) -> _LayerSnapshot:
    return _LayerSnapshot(
        theme_dominant_type=layer.theme_dominant_type,
        height_above_shelf_base=layer.height_above_shelf_base,
        relative_height=layer.relative_height,
        vertical_clearance=layer.vertical_clearance,
        objects=tuple(_object_snapshot(object_) for object_ in layer.objects),
    )


def _object_snapshot(object_: RelationalCircuitExperimentObject2D) -> _ObjectSnapshot:
    return _ObjectSnapshot(
        object_type=object_.object_type,
        scale=(object_.scale.x, object_.scale.y, object_.scale.z),
        position=(
            float(object_.pose.x),
            float(object_.pose.y),
            float(object_.pose.yaw),
        ),
        source_id=object_.source_id,
    )


def test_extraction_from_the_kept_objects_matches_extraction_from_all_of_them() -> None:
    """
    Holding back only shelves and their contents is what keeps the pipeline's memory
    bounded, so it must leave the extracted shelves exactly as they would have been had
    the whole dataset been kept.
    """
    every_object = _three_layer_shelf() + [
        _eg_object("chair_1", "floor", ObjectType.CHAIR, x=5.0, y=5.0),
        _eg_object("cup_1", "table_1", ObjectType.CUP, x=6.0, y=6.0),
    ]
    contents = ShelfContents(shelf_ids={"room_1_shelf_1"})
    for processed_object in every_object:
        contents.collect(processed_object)

    bounds_by_source_id = {
        "room_1_shelf_1_src": MeshBounds(
            footprint_center_x=0.0, footprint_center_y=0.0, bottom=-1.0, top=1.0
        )
    }
    measurements = MeshMeasurements(source_id_to_path={})
    assert [
        _shelf_snapshot(shelf)
        for shelf in Sage10kPreprocessingRun._shelves_with_layers(
            contents.objects,
            bounds_by_source_id,
            contents.shelf_ids,
            measurements,
        )
    ] == [
        _shelf_snapshot(shelf)
        for shelf in Sage10kPreprocessingRun._shelves_with_layers(
            every_object,
            bounds_by_source_id,
            contents.shelf_ids,
            measurements,
        )
    ]


def test_kept_shelves_supply_the_bounds_their_layers_need(tmp_path: Path) -> None:
    """
    Only the shelves' own meshes have to be measured; measuring every object's would
    load meshes whose reach nothing reads.
    """
    scene_directory = _cache_off_center_mesh(tmp_path, "room_1_shelf_1_src")
    contents = ShelfContents(shelf_ids={"room_1_shelf_1"})
    for processed_object in _three_layer_shelf():
        contents.collect(processed_object)

    bounds_by_source_id = contents.shelf_bounds(
        MeshMeasurements(source_id_to_path={"room_1_shelf_1_src": scene_directory})
    )

    assert set(bounds_by_source_id) == {"room_1_shelf_1_src"}


# %% Extraction and spawning must be inverses


def _cache_book_mesh(tmp_path: Path) -> None:
    """
    Copy a real textured mesh into *tmp_path* under the ``book_src`` source id, so a
    spawned shelf has geometry to place.
    """
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    objects_directory = tmp_path / "objects"
    objects_directory.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_directory / "book_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png",
        objects_directory / "book_src_texture.png",
    )


def test_extracted_contents_spawn_within_the_layer_footprint(tmp_path: Path) -> None:
    """
    An object offset along the shelf's wide face must spawn inside the corpus footprint
    on both axes.

    Were that face offset mapped onto the corpus's shallow depth axis, the object would
    protrude front and back.
    """
    _cache_book_mesh(tmp_path)
    shelf_depth, shelf_face = 0.3, 1.0
    shelf = _eg_object(
        "room_1_shelf_1",
        place_id="floor",
        object_type=ObjectType.SHELF,
        x=0.0,
        y=0.0,
        z=1.0,
        width=shelf_face,
        length=shelf_depth,
    )
    # At the shelf's zero yaw its wide face lies along world x, so a world-x
    # offset is a face offset: well within the face, far outside the depth.
    book = _eg_object(
        "book_1",
        place_id="room_1_shelf_1",
        object_type=ObjectType.BOOK,
        x=0.4,
        y=0.0,
        source_id="book_src",
    )

    [layers] = _layers_by_shelf([shelf, book])
    spawned = RelationalCircuitExperimentShelf(
        scale=Scale(x=shelf_depth, y=shelf_face, z=2.0),
        layers=layers,
        source_ids=[
            MeshCandidate(
                scene_directory=tmp_path,
                source_id="book_src",
                object_type=ObjectType.BOOK,
            )
        ],
        theme_dominant_type=ObjectType.BOOK,
    )
    world, root = _empty_world()
    spawned.spawn(world, parent=root)

    body = spawned.layers[0].objects[0].annotation
    corpus_x, corpus_y = body.parent_connection.origin.to_position().to_np()[:2]
    assert abs(corpus_x) <= shelf_depth / 2
    assert abs(corpus_y) <= shelf_face / 2


# %% Sage10kPreprocessingRun._measure_meshes_in_parallel -- mesh measurement split across processes


def test_parallel_measurement_matches_sequential_measurement(tmp_path: Path) -> None:
    """
    Measuring across worker processes must find the same bounds a single sequential pass
    would, for both a cached mesh and one that is missing.
    """
    scene_directory_a = _cache_off_center_mesh(tmp_path / "a", "mesh_a")
    scene_directory_b = _cache_off_center_mesh(tmp_path / "b", "mesh_b")
    source_id_to_path = {"mesh_a": scene_directory_a, "mesh_b": scene_directory_b}
    sequential = MeshMeasurements(source_id_to_path=source_id_to_path)

    parallel_bounds = Sage10kPreprocessingRun._measure_meshes_in_parallel(
        source_id_to_path, ["mesh_a", "mesh_b", "missing"], worker_count=2
    )

    assert parallel_bounds["mesh_a"] == sequential.bounds("mesh_a")
    assert parallel_bounds["mesh_b"] == sequential.bounds("mesh_b")
    assert parallel_bounds["missing"] is None


# %% Sage10kPreprocessingRun._partition_round_robin -- splitting rooms into shards


def test_partition_round_robin_splits_evenly_and_preserves_relative_order() -> None:
    partitions = Sage10kPreprocessingRun._partition_round_robin(
        ["a", "b", "c", "d", "e"], 2
    )

    assert partitions == [["a", "c", "e"], ["b", "d"]]


def test_partition_round_robin_leaves_extra_partitions_empty() -> None:
    partitions = Sage10kPreprocessingRun._partition_round_robin(["a", "b"], 5)

    assert partitions == [["a"], ["b"], [], [], []]


# %% Sage10kPreprocessingRun._process_objects_in_parallel -- the sharded read-convert-write pass


def _populated_sqlite_engine(tmp_path: Path, objects: list[Sage10kObjectDAO]):
    engine = create_engine(f"sqlite:///{tmp_path}/raw.db")
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        session.add_all(objects)
        session.commit()
    return engine


def test_sharded_object_pass_writes_every_object_exactly_once(tmp_path: Path) -> None:
    """
    Splitting the object pass across shards must not drop or duplicate objects: the
    processed database must end up with exactly the rows a single-shard pass would have
    written.
    """
    objects = [
        _sage10k_object(object_id="book_1", room_id="room_1", source_id="book_1_src"),
        _sage10k_object(object_id="book_2", room_id="room_2", source_id="book_2_src"),
        _sage10k_object(object_id="book_3", room_id="room_3", source_id="book_3_src"),
    ]
    _populated_sqlite_engine(tmp_path, objects)
    raw_uri = f"sqlite:///{tmp_path}/raw.db"
    processed_uri = f"sqlite:///{tmp_path}/processed.db"
    processed_engine = create_engine(processed_uri)
    Base.metadata.create_all(bind=processed_engine)

    run = Sage10kPreprocessingRun(
        sage10k_database_uri=raw_uri, processed_database_uri=processed_uri
    )
    results = run._process_objects_in_parallel(
        room_ids=["room_1", "room_2", "room_3"],
        source_id_to_path={},
        bounds_by_source_id={},
        shelf_ids=set(),
        worker_count=2,
    )

    assert sum(result.stored_count for result in results) == 3
    with Session(processed_engine) as session:
        stored_ids = set(session.execute(select(PreprocessedObjectDAO.id)).scalars())
    assert stored_ids == {"book_1", "book_2", "book_3"}


def test_sharded_object_pass_keeps_shelf_contents_regardless_of_shard(
    tmp_path: Path,
) -> None:
    """
    A shelf and everything standing on it always share a room, so no shard should ever
    lose track of shelf membership by having a shelf and its contents split apart: the
    shelf must be extracted and stored with its book on it, regardless of which of the
    two shards happened to own its room.
    """
    objects = [
        _sage10k_object(
            object_id="shelf_1",
            raw_type="bookshelf1",
            room_id="room_1",
            source_id="shelf_1_src",
            place_id="floor",
        ),
        _sage10k_object(
            object_id="book_1",
            room_id="room_1",
            source_id="book_1_src",
            place_id="shelf_1",
        ),
        _sage10k_object(
            object_id="chair_1",
            raw_type="chair",
            room_id="room_2",
            source_id="chair_1_src",
            place_id="floor",
        ),
    ]
    _populated_sqlite_engine(tmp_path, objects)
    raw_uri = f"sqlite:///{tmp_path}/raw.db"
    processed_uri = f"sqlite:///{tmp_path}/processed.db"
    processed_engine = create_engine(processed_uri)
    Base.metadata.create_all(bind=processed_engine)
    bounds_by_source_id = {
        "shelf_1_src": MeshBounds(
            footprint_center_x=0.0,
            footprint_center_y=0.0,
            bottom=-1.0,
            top=1.0,
        ),
        "book_1_src": MeshBounds(
            footprint_center_x=0.0,
            footprint_center_y=0.0,
            bottom=0.0,
            top=0.1,
        ),
    }

    run = Sage10kPreprocessingRun(
        sage10k_database_uri=raw_uri, processed_database_uri=processed_uri
    )
    results = run._process_objects_in_parallel(
        room_ids=["room_1", "room_2"],
        source_id_to_path={},
        bounds_by_source_id=bounds_by_source_id,
        shelf_ids={"shelf_1"},
        worker_count=2,
    )

    assert sum(result.shelf_count for result in results) == 1
    assert sum(result.layer_count for result in results) == 1
    with Session(processed_engine) as session:
        [shelf_dao] = session.query(RelationalCircuitExperimentShelfDAO).all()
        stored_shelf = shelf_dao.from_dao()
    content_source_ids = {
        object_.source_id for layer in stored_shelf.layers for object_ in layer.objects
    }
    assert content_source_ids == {"book_1_src"}


def test_only_shelf_relevant_meshes_are_measured(tmp_path: Path) -> None:
    """
    An object that neither is a shelf nor stands on one must be stored uncorrected even
    when its mesh is fully cached and measurable, since mesh measurement is scoped to
    :attr:`ShelfContents.relevant_source_ids` -- shelves and their own contents -- and
    nothing downstream of layer extraction reads any other object's correction.
    """
    objects = [
        _sage10k_object(
            object_id="shelf_1",
            raw_type="bookshelf1",
            room_id="room_1",
            source_id="shelf_1_src",
            place_id="floor",
        ),
        _sage10k_object(
            object_id="book_1",
            room_id="room_1",
            source_id="book_1_src",
            place_id="shelf_1",
        ),
        _sage10k_object(
            object_id="chair_1",
            raw_type="chair",
            room_id="room_2",
            source_id="chair_1_src",
            place_id="floor",
        ),
    ]
    engine = _populated_sqlite_engine(tmp_path, objects)
    raw_uri = f"sqlite:///{tmp_path}/raw.db"
    processed_uri = f"sqlite:///{tmp_path}/processed.db"
    processed_engine = create_engine(processed_uri)
    Base.metadata.create_all(bind=processed_engine)

    # Every mesh is cached, chair_1's included, so it is skipped by scope rather than
    # by being unmeasurable.
    full_source_id_to_path = {
        "shelf_1_src": _cache_off_center_mesh(tmp_path / "shelf", "shelf_1_src"),
        "book_1_src": _cache_off_center_mesh(tmp_path / "book", "book_1_src"),
        "chair_1_src": _cache_off_center_mesh(tmp_path / "chair", "chair_1_src"),
    }
    with Session(engine) as session:
        shelf_contents = ShelfContents.from_raw_objects(
            session, ShelfMembershipClassifier()
        )
    assert shelf_contents.relevant_source_ids == {"shelf_1_src", "book_1_src"}
    # Mirrors Sage10kPreprocessingRun.run(): narrowing bounds_by_source_id alone is not
    # enough, since MeshMeasurements.bounds() lazily re-measures anything it finds a
    # path for -- so source_id_to_path itself must be narrowed before either the
    # measurement pass or the object pass sees it.
    relevant_source_id_to_path = {
        source_id: path
        for source_id, path in full_source_id_to_path.items()
        if source_id in shelf_contents.relevant_source_ids
    }
    bounds_by_source_id = Sage10kPreprocessingRun._measure_meshes_in_parallel(
        relevant_source_id_to_path, shelf_contents.relevant_source_ids, worker_count=1
    )

    run = Sage10kPreprocessingRun(
        sage10k_database_uri=raw_uri, processed_database_uri=processed_uri
    )
    run._process_objects_in_parallel(
        room_ids=["room_1", "room_2"],
        source_id_to_path=relevant_source_id_to_path,
        bounds_by_source_id=bounds_by_source_id,
        shelf_ids=shelf_contents.shelf_ids,
        worker_count=2,
    )

    with Session(processed_engine) as session:
        corrected_by_id = dict(
            session.execute(
                select(
                    PreprocessedObjectDAO.id,
                    PreprocessedObjectDAO.position_is_mesh_corrected,
                )
            ).all()
        )
    assert corrected_by_id == {
        "shelf_1": True,
        "book_1": True,
        "chair_1": False,
    }

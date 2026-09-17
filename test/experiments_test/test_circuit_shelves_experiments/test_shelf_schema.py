from __future__ import annotations

import math
import shutil
from importlib.resources import files
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from experiments.orm.ormatic_interface import Base, RelationalCircuitExperimentShelfDAO
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
    RelationalCircuitExperimentShelfLayer,
)
from experiments.shelf_generation_experiments.utils import MeshCandidate, ObjectType
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def chair_mesh_directory(tmp_path: Path) -> Path:
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    objects_directory = tmp_path / "objects"
    objects_directory.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_directory / "chair_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png",
        objects_directory / "chair_src_texture.png",
    )
    return tmp_path


@pytest.fixture
def close_and_oversized_book_candidates(
    tmp_path: Path,
) -> tuple[MeshCandidate, MeshCandidate]:
    """
    Two mesh candidates of the same ``ObjectType``, sharing the same underlying asset
    but claiming very different real-world sizes -- one close to the book object
    :func:`_make_layer` samples, one bookcase-sized.

    The geometry itself is irrelevant here; only the claimed ``native_extents`` matter
    for the size scoring under test.
    """
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    objects_directory = tmp_path / "objects"
    objects_directory.mkdir()
    for source_id in ("close_match", "oversized"):
        shutil.copy(
            resources_root / "chair.ply", objects_directory / f"{source_id}.ply"
        )
        shutil.copy(
            resources_root / "chair_texture.png",
            objects_directory / f"{source_id}_texture.png",
        )
    return (
        MeshCandidate(
            tmp_path, "close_match", ObjectType.BOOK, native_extents=(0.1, 0.05, 0.2)
        ),
        MeshCandidate(
            tmp_path, "oversized", ObjectType.BOOK, native_extents=(0.6, 0.25, 1.5)
        ),
    )


def _make_layer(relative_height: float = 0.0) -> RelationalCircuitExperimentShelfLayer:
    return RelationalCircuitExperimentShelfLayer(
        objects=[
            RelationalCircuitExperimentObject2D(
                object_type=ObjectType.BOOK,
                scale=Scale(x=0.05, y=0.1, z=0.2),
                pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
                source_id="chair_src",
            )
        ],
        theme_dominant_type=ObjectType.BOOK,
        relative_height=relative_height,
    )


def _make_shelf(
    relative_heights: tuple[float, ...] = (0.0,),
    scale: Scale | None = None,
) -> RelationalCircuitExperimentShelf:
    return RelationalCircuitExperimentShelf(
        scale=scale or Scale(x=0.4, y=0.8, z=2.0),
        layers=[_make_layer(height) for height in relative_heights],
        theme_dominant_type=ObjectType.BOOK,
        source_ids=[],
    )


def _object_bodies(layer: RelationalCircuitExperimentShelfLayer) -> dict[int, Body]:
    """
    The bodies spawned for *layer*'s objects, keyed by their index in
    :attr:`RelationalCircuitExperimentShelfLayer.objects`; objects with no spawned body
    are omitted.
    """
    return {
        index: object_.annotation
        for index, object_ in enumerate(layer.objects)
        if object_.annotation is not None
    }


def _slab_footprints(
    shelf: RelationalCircuitExperimentShelf,
) -> set[tuple[float, float]]:
    """
    Every spawned slab's x/y extents, rounded so float noise does not split them.
    """
    shelf.spawn(World.create_with_root_body())
    return {
        (
            round(float(layer.annotation.root.collision.shapes[0].scale.x), 6),
            round(float(layer.annotation.root.collision.shapes[0].scale.y), 6),
        )
        for layer in shelf.layers
    }


def _slab_heights(shelf: RelationalCircuitExperimentShelf) -> list[float]:
    """
    Height of every spawned slab above the shelf's base, lowest first.

    Slabs are reparented onto the corpus so the shelf moves as one unit, which puts
    their origins in the corpus frame, centred half a shelf up.
    """
    shelf.spawn(World.create_with_root_body())
    return sorted(
        float(
            layer.annotation.root.parent_connection.origin.to_position()
            .to_np()[2]
            .item()
        )
        + shelf.scale.z / 2
        for layer in shelf.layers
    )


def test_shelf_mounts_at_the_origin_of_the_parent_it_is_given(
    chair_mesh_directory: Path,
) -> None:
    """
    A shelf defines the frame its contents are expressed in and sits at that frame's
    origin, so a caller positions it by choosing the parent rather than by a pose on the
    shelf.

    The corpus is still built in the content frame, so its yaw carries
    :meth:`RelationalCircuitExperimentShelf.content_frame_yaw`'s offset, which extraction has to invert.
    """
    shelf = _make_shelf()
    shelf.source_ids = [
        MeshCandidate(chair_mesh_directory, "chair_src", ObjectType.BOOK)
    ]

    world = World()
    parent = Body(name=PrefixedName(name="room_parent"))
    with world.modify_world():
        world.add_body(parent)

    shelf.spawn(world, parent=parent)

    [corpus_body] = [body for body in world.bodies if body.name.name == "shelf_corpus"]
    assert corpus_body.parent_connection.parent is parent

    translation = corpus_body.parent_connection.origin.to_position().to_np()
    assert translation[0] == pytest.approx(0.0, abs=1e-6)
    assert translation[1] == pytest.approx(0.0, abs=1e-6)

    yaw = corpus_body.parent_connection.origin.to_rotation_matrix().to_rpy()[2]
    assert float(yaw.to_np().item()) == pytest.approx(
        RelationalCircuitExperimentShelf.content_frame_yaw(), abs=1e-6
    )


def test_content_frame_yaw_adds_the_offset_to_the_shelfs_own_yaw() -> None:
    """
    Extraction rotates a raw object's offset by its shelf's own yaw plus the content
    frame offset; :meth:`RelationalCircuitExperimentShelf.spawn` builds the corpus at
    that same combination (with a shelf yaw of zero, since a spawned shelf carries none
    of its own) -- both read this one method so they cannot drift apart from each other.
    """
    shelf_yaw_radians = math.radians(37.0)

    content_frame_yaw = RelationalCircuitExperimentShelf.content_frame_yaw(
        shelf_yaw_radians
    )

    assert content_frame_yaw == pytest.approx(
        shelf_yaw_radians
        + math.radians(
            RelationalCircuitExperimentShelf.content_frame_yaw_offset_degrees
        )
    )


def test_content_frame_yaw_defaults_to_the_offset_alone() -> None:
    """
    :meth:`RelationalCircuitExperimentShelf.spawn` calls this with no shelf yaw, since a
    spawned shelf carries no pose of its own.
    """
    assert RelationalCircuitExperimentShelf.content_frame_yaw() == pytest.approx(
        math.radians(RelationalCircuitExperimentShelf.content_frame_yaw_offset_degrees)
    )


def test_theme_dominant_type_survives_a_database_round_trip() -> None:
    """
    The theme is what a sampled shelf is conditioned on, so a stored shelf that lost it
    would come back as a shelf of no particular theme.
    """
    shelf = _make_shelf()
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = Session(engine)
    session.add(to_dao(shelf))
    session.commit()
    session.expunge_all()

    [stored] = session.query(RelationalCircuitExperimentShelfDAO).all()
    restored = stored.from_dao()

    assert restored.theme_dominant_type is ObjectType.BOOK
    assert [layer.theme_dominant_type for layer in restored.layers] == [ObjectType.BOOK]


def test_every_slab_spawns_at_the_shelfs_own_footprint() -> None:
    """
    A layer carries no footprint of its own, so every slab spawns at its shelf's width
    and length.
    """
    shelf = _make_shelf(relative_heights=(0.2, 0.5, 0.8))

    footprints = _slab_footprints(shelf)

    assert footprints == {(shelf.scale.x, shelf.scale.y)}


def test_the_corpus_interior_matches_the_shelfs_learned_dimensions() -> None:
    """
    The corpus used to be sized from the layers, so the shelf's own learned width and
    depth had no effect and every type spawned the same box.

    They are what distinguishes a narrow bookcase from a wide cabinet.
    """
    narrow = _make_shelf(scale=Scale(x=0.26, y=0.62, z=1.5))
    wide = _make_shelf(scale=Scale(x=0.40, y=1.43, z=1.5))

    corpus_wall_thickness = 0.03
    assert narrow.corpus_footprint.y == pytest.approx(
        0.62 + 2 * corpus_wall_thickness, abs=1e-6
    )
    assert wide.corpus_footprint.y == pytest.approx(
        1.43 + 2 * corpus_wall_thickness, abs=1e-6
    )
    assert narrow.corpus_footprint.y < wide.corpus_footprint.y


def test_slabs_are_evenly_spaced_whatever_heights_were_drawn() -> None:
    """
    ``relative_height`` records where objects were *found*, not where slabs are:

    an empty shelf level leaves no trace, so a measured gap is the distance to the next
    occupied level. Real shelves are evenly spaced, so the drawn heights must not become
    slab positions.
    """
    shelf = _make_shelf(relative_heights=(0.05, 0.06, 0.9))

    heights = _slab_heights(shelf)

    gaps = [second - first for first, second in zip(heights, heights[1:])]
    assert gaps == pytest.approx([gaps[0]] * len(gaps), abs=1e-6)


def test_each_slab_is_placed_at_its_own_layers_height_rank() -> None:
    """
    ``_layer_heights`` computes the evenly-spaced grid by walking ``layers`` sorted by
    ``relative_height``, but a shelf's ``layers`` are drawn from an exchangeable RSPN
    template and come back in no particular order.

    A slab must land at the height rank of the layer it was built for, not at whichever
    grid slot the sorted pass happened to produce it in.
    """
    shelf = _make_shelf(relative_heights=(0.9, 0.1, 0.5))
    shelf.spawn(World.create_with_root_body())

    spawned_heights = [
        float(
            layer.annotation.root.parent_connection.origin.to_position()
            .to_np()[2]
            .item()
        )
        + shelf.scale.z / 2
        for layer in shelf.layers
    ]

    ranks = sorted(
        range(len(spawned_heights)), key=lambda index: spawned_heights[index]
    )
    expected_ranks = sorted(
        range(len(shelf.layers)),
        key=lambda index: shelf.layers[index].relative_height,
    )
    assert ranks == expected_ranks


def test_every_slab_gap_leaves_room_for_a_typical_object() -> None:
    """
    Layers drawn close together used to be pushed apart by 3 cm, which is far less than
    anything stands on a shelf, so the layer spawned empty.

    The median object in the dataset is 0.077 m tall.
    """
    shelf = _make_shelf(relative_heights=(0.5, 0.5, 0.5))

    heights = _slab_heights(shelf)

    gaps = [second - first for first, second in zip(heights, heights[1:])]
    assert all(gap > 0.077 for gap in gaps)


def test_objects_recorded_on_top_of_low_furniture_spawn_above_it() -> None:
    """
    A quarter of the recorded layers describe things standing *on* a piece of furniture,
    not on a shelf inside it.

    On a low cabinet that is where they belong, and spawning them inside crushes them
    against the corpus ceiling.
    """
    shelf = _make_shelf(
        relative_heights=(0.3, 1.0),
        scale=Scale(x=0.4, y=0.8, z=1.0),
    )

    heights = _slab_heights(shelf)

    assert heights[-1] == pytest.approx(shelf.scale.z, abs=1e-6)


def test_objects_recorded_on_top_of_tall_furniture_spawn_inside_it() -> None:
    """
    Nobody leaves things on top of a shelf they cannot reach, so above the reach
    threshold the recorded top layer is an ordinary level.

    Its objects are kept rather than discarded -- they were observed, only their height
    is implausible.
    """
    shelf = _make_shelf(
        relative_heights=(0.3, 1.0),
        scale=Scale(x=0.4, y=0.8, z=2.4),
    )

    heights = _slab_heights(shelf)

    assert all(height < shelf.scale.z for height in heights)
    assert len(heights) == 2


def test_an_object_on_the_shelfs_top_is_not_rejected_for_lack_of_headroom(
    chair_mesh_directory: Path,
) -> None:
    """
    Nothing stands above the shelf's top, so an object placed there has open air over
    it.

    Measuring its headroom against the corpus ceiling -- which lies *below* the top
    surface -- makes every such object appear too tall and drops the whole layer.
    """
    shelf = _make_shelf(relative_heights=(1.0,), scale=Scale(x=0.4, y=0.8, z=1.0))
    shelf.source_ids = [
        MeshCandidate(
            chair_mesh_directory,
            "chair_src",
            ObjectType.BOOK,
            native_extents=(0.1, 0.05, 0.2),
        )
    ]

    shelf.spawn(World.create_with_root_body())

    assert sum(len(_object_bodies(layer)) for layer in shelf.layers) == 1


def test_object_mesh_is_matched_to_its_sampled_size_not_just_its_type(
    close_and_oversized_book_candidates: tuple[MeshCandidate, MeshCandidate],
) -> None:
    """
    A mesh candidate's real size must be weighed against the size the circuit actually
    sampled for that object, not only its ``ObjectType`` -- otherwise a bookcase-sized
    mesh tagged the same type as a 0.2 m book is just as eligible as one the right size,
    purely because both fit the layer's footprint (and, on the shelf's top, its
    unbounded headroom).
    """
    close_match, oversized = close_and_oversized_book_candidates
    shelf = _make_shelf(relative_heights=(1.0,), scale=Scale(x=0.4, y=0.8, z=1.0))
    shelf.source_ids = [oversized, close_match]

    shelf.spawn(World.create_with_root_body())

    assert sum(len(_object_bodies(layer)) for layer in shelf.layers) == 1
    assert shelf.layers[0].objects[0].source_id == "close_match"


def test_only_one_layer_can_occupy_the_shelfs_top() -> None:
    """
    Layers are drawn independently, so several can come back recorded at the shelf's
    top.

    A shelf has one top, and placing them all there stacks slabs at the same height with
    no room between them for anything to stand.
    """
    shelf = _make_shelf(
        relative_heights=(0.3, 1.0, 1.0),
        scale=Scale(x=0.4, y=0.8, z=1.4),
    )

    heights = _slab_heights(shelf)

    gaps = [second - first for first, second in zip(heights, heights[1:])]
    assert all(gap > 0.077 for gap in gaps)


# %% layer geometry


def test_layer_geometry_reports_the_height_its_slab_spawns_at() -> None:
    """
    A caller placing something on an already-spawned shelf reads its heights from the
    geometry, so a geometry that disagreed with the spawn would aim at nothing.
    """
    shelf = _make_shelf(relative_heights=(0.9, 0.1, 0.5))

    layers = shelf.layers_with_geometry()

    # Reverses layers_with_geometry's own
    # slab_top_height = (height - corpus_height / 2) + slab_thickness / 2, since
    # corpus_height == shelf.scale.z (RelationalCircuitExperimentShelf.corpus_footprint pads x/y but not z).
    heights = [
        layer.slab_top_height
        - RelationalCircuitExperimentShelfLayer.slab_thickness / 2
        + shelf.scale.z / 2
        for layer in layers
    ]
    assert sorted(heights) == pytest.approx(_slab_heights(shelf))


def test_a_layer_accepts_an_object_that_reaches_just_under_the_next_slab() -> None:
    """
    An object taller than the room above its slab pierces the shelf above, which no in-
    plane repair can fix, so that room is exactly what the layer accepts.
    """
    shelf = _make_shelf(relative_heights=(0.2, 0.4, 0.6))

    layers = sorted(
        shelf.layers_with_geometry(), key=lambda layer: layer.slab_top_height
    )

    lowest, next_up = layers[0], layers[1]
    slab_underside = next_up.slab_top_height - 0.02
    assert lowest.maximum_object_extents.z == pytest.approx(
        slab_underside - lowest.slab_top_height - 0.01  # margin
    )


def test_a_layer_on_the_shelfs_top_accepts_an_object_of_any_height() -> None:
    """
    Nothing stands above the shelf's top, so a layer resting there is bounded by nothing
    rather than by the corpus ceiling below it.
    """
    shelf = _make_shelf(relative_heights=(0.3, 1.0), scale=Scale(x=0.4, y=0.8, z=1.0))

    heights = [layer.maximum_object_extents.z for layer in shelf.layers_with_geometry()]

    assert heights[1] == math.inf
    assert heights[0] < math.inf


def test_every_layer_accepts_an_object_as_wide_as_the_shelf() -> None:
    """
    A slab spans its shelf's own footprint, so nothing narrower than the shelf is
    rejected for its width or length.
    """
    shelf = _make_shelf(relative_heights=(0.2, 0.8))

    for layer in shelf.layers_with_geometry():
        assert layer.maximum_object_extents.y == shelf.scale.y
        assert layer.maximum_object_extents.x == shelf.scale.x


def test_identical_layers_still_get_their_own_ceilings() -> None:
    """
    Layers compare equal whenever they hold equal objects, so a layer looked up by value
    finds the first one every time and every slab is measured against the bottom slab's
    ceiling.
    """
    shelf = _make_shelf(relative_heights=(0.5, 0.5, 0.5))

    layers = shelf.layers_with_geometry()

    rooms = [layer.maximum_object_extents.z for layer in layers]
    assert all(room > 0 for room in rooms)

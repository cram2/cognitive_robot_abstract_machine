"""
Native shape collections remain authoritative for live object geometry.
"""

from pathlib import Path
import urllib.parse
from dataclasses import replace

import numpy
import pytest
import trimesh
from PIL import Image
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Mesh,
    Scale,
    Sphere,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live.bridge import Bridge, ObjectCatalogEntry
from cramera.live.recording_bundle import _object_entry

from .dataset.mesh_geometry import resolved_textured_mesh


# %% catalog geometry
def test_mesh_urls_reuse_native_sources_across_bodies_and_shape_order(
    resolved_textured_mesh: Mesh, tmp_path: Path
) -> None:
    """
    Keep source identity stable while preserving different material assets.

    :param resolved_textured_mesh: First native mesh with its texture assets.
    :param tmp_path: Directory receiving a second material-bearing native export.
    """
    second_geometry = resolved_textured_mesh.unscaled_mesh.copy()
    second_geometry.visual.material.image = Image.new("RGB", (2, 2), (220, 30, 50))
    second_mesh = Mesh.from_trimesh(second_geometry, directory=tmp_path)
    repeated_mesh = replace(resolved_textured_mesh, scale=Scale(2, 3, 4))
    first_body = Body(
        name=PrefixedName("first"),
        visual=ShapeCollection(shapes=[resolved_textured_mesh, second_mesh]),
    )
    second_body = Body(
        name=PrefixedName("second"),
        visual=ShapeCollection(shapes=[repeated_mesh]),
    )
    bridge = Bridge()
    bodies = {str(body.name): body for body in [first_body, second_body]}
    bridge.publish_bodies(bodies)

    first_entry, second_entry = bridge.object_catalog()
    first_url, different_url = [shape["mesh"] for shape in first_entry["shapes"]]
    repeated_shape = second_entry["shapes"][0]
    assert repeated_shape["mesh"] == first_url
    assert repeated_shape["scale"] == repeated_mesh.scale.to_np().tolist()
    assert different_url != first_url
    numpy.testing.assert_allclose(
        second_mesh.unscaled_mesh.vertices,
        resolved_textured_mesh.unscaled_mesh.vertices,
    )
    for mesh, url in [
        (resolved_textured_mesh, first_url),
        (second_mesh, different_url),
    ]:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        assert query["key"] == [mesh.filename]
        assert bridge.mesh_path(mesh.filename) == str(mesh.local_file)

    first_body.visual = ShapeCollection(shapes=[second_mesh, resolved_textured_mesh])
    bridge.publish_bodies(bodies)
    reordered_urls = [shape["mesh"] for shape in bridge.object_catalog()[0]["shapes"]]
    assert reordered_urls == [different_url, first_url]


def test_catalog_resolves_the_native_mesh_and_its_declared_material(
    resolved_textured_mesh: Mesh,
) -> None:
    """
    Serve the resolved file and the material declared by its native export.

    :param resolved_textured_mesh: Native mesh with material and texture side assets.
    """
    body = Body(
        name=PrefixedName("textured"),
        visual=ShapeCollection(shapes=[resolved_textured_mesh]),
    )
    bridge = Bridge()
    bridge.publish_bodies({str(body.name): body})

    [shape] = bridge.object_catalog()[0]["shapes"]
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(shape["mesh"]).query)
    material_query = urllib.parse.parse_qs(urllib.parse.urlsplit(shape["mtl"]).query)
    [material] = resolved_textured_mesh.local_file.parent.glob("*.mtl")

    assert bridge.mesh_path(query["key"][0]) == str(resolved_textured_mesh.local_file)
    assert material_query["side"] == [material.name]
    assert material_query["key"] == query["key"]


def test_recording_copies_the_resolved_mesh_with_its_original_materials(
    resolved_textured_mesh: Mesh, tmp_path: Path
) -> None:
    """
    Keep original file bytes and texture assets when the native pose is unchanged.

    :param resolved_textured_mesh: Native exported mesh addressed through a file URI.
    :param tmp_path: Directory receiving the copied recording assets.
    """
    entry = ObjectCatalogEntry(
        key="textured", shapes=ShapeCollection(shapes=[resolved_textured_mesh])
    )
    destination = tmp_path / "recorded"

    payload = _object_entry(entry, [0, 0, 0, 0, 0, 0, 1], destination)

    recorded_file = destination / payload["mesh"]
    assert recorded_file.read_bytes() == resolved_textured_mesh.local_file.read_bytes()
    for source in resolved_textured_mesh.local_file.parent.iterdir():
        if source == resolved_textured_mesh.local_file:
            continue
        assert (recorded_file.parent / source.name).read_bytes() == source.read_bytes()


def test_catalog_serializes_native_meshes_without_an_external_file_map(
    tmp_path: Path,
) -> None:
    """
    Use the entry's native meshes as the complete geometry input.

    :param tmp_path: Directory holding the native mesh file.
    """
    mesh_path = tmp_path / "native.obj"
    Box(scale=Scale(0.2, 0.3, 0.4)).mesh.export(mesh_path)
    mesh = Mesh(filename=str(mesh_path), scale=Scale(2, 3, 4))
    entry = ObjectCatalogEntry(key="native", shapes=ShapeCollection(shapes=[mesh]))

    payload = entry.to_payload()

    [shape] = payload["shapes"]
    assert shape["format"] == mesh_path.suffix.lstrip(".")
    assert shape["scale"] == mesh.scale.to_np().tolist()
    assert shape["mesh"].split("?")[0] == "/mesh"


@pytest.mark.parametrize("color", [Color(), Color.WHITE(), Color(1, 1, 1, 0.4)])
def test_white_geometry_keeps_its_native_color(color: Color) -> None:
    """
    Keep white and translucent white instead of assigning a palette color.

    :param color: The white appearance assigned to native geometry.
    """
    body = Body(
        name=PrefixedName("white"),
        visual=ShapeCollection(shapes=[Sphere(radius=0.2, color=color)]),
    )
    bridge = Bridge()
    bridge.publish_bodies({str(body.name): body})

    [entry] = bridge.object_catalog()
    [shape] = entry["shapes"]

    assert bridge.object_metadata[0].color is color
    assert entry["color"] == color.to_hex()
    assert shape["color"] == color.to_hex()
    assert shape["opacity"] == color.A


@pytest.mark.parametrize("color", [Color(0.8, 0.2, 0.7), Color(0.1, 0.6, 0.9, 0.4)])
def test_catalog_uses_native_color_conversion(color: Color) -> None:
    """
    Publish native RGB conversion while retaining opacity separately.

    :param color: The native appearance of the published shape.
    """
    body = Body(
        name=PrefixedName("colored"),
        visual=ShapeCollection(shapes=[Sphere(radius=0.2, color=color)]),
    )
    bridge = Bridge()
    bridge.publish_bodies({str(body.name): body})

    [entry] = bridge.object_catalog()
    [shape] = entry["shapes"]

    assert entry["color"] == color.to_hex()
    assert shape["color"] == color.to_hex()
    assert shape["opacity"] == color.A


def test_catalog_reuses_the_visual_shape_collection() -> None:
    """
    Publish the body's existing visual geometry without a second classification.
    """
    body = Body(
        name=PrefixedName("object"),
        visual=ShapeCollection(shapes=[Sphere(radius=0.2)]),
        collision=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
    )
    bridge = Bridge()

    bridge.publish_bodies({str(body.name): body})

    assert bridge.object_metadata[0].shapes is body.visual


def test_catalog_reuses_collision_geometry_without_visual_shapes() -> None:
    """
    Select the native collision collection when the visual collection is empty.
    """
    body = Body(
        name=PrefixedName("object"),
        collision=ShapeCollection(shapes=[Sphere(radius=0.2)]),
    )
    bridge = Bridge()

    bridge.publish_bodies({str(body.name): body})

    assert bridge.object_metadata[0].shapes is body.collision


def test_shapeless_catalog_preserves_its_empty_native_geometry() -> None:
    """
    Keep a body addressable without inventing a solid where it has no geometry.
    """
    body = Body(name=PrefixedName("object"))
    bridge = Bridge()

    bridge.publish_bodies({str(body.name): body})

    assert bridge.object_metadata[0].shapes is body.collision
    assert bridge.object_catalog()[0]["shapes"] == []
    assert bridge.object_body(str(body.name)) is body
    assert bridge.object_keys() == [str(body.name)]


def test_recording_keeps_a_shapeless_body_without_exporting_a_solid(
    tmp_path: Path,
) -> None:
    """
    Retain identity and spawn pose when no geometry is available to record.

    :param tmp_path: Directory receiving the object's recording assets.
    """
    body = Body(name=PrefixedName("empty"))
    bridge = Bridge()
    bridge.publish_bodies({str(body.name): body})
    pose = [1, 2, 3, 0, 0, 0, 1]

    payload = _object_entry(bridge.object_metadata[0], pose, tmp_path)

    assert payload["key"] == str(body.name)
    assert payload["spawn"] == pose
    assert payload["shapes"] == []
    assert "box" not in payload
    assert "mesh" not in payload
    assert list(tmp_path.iterdir()) == []


def test_a_shapeless_body_keeps_its_world_pose(
    world_with_two_bodies: tuple[World, Body, Body],
) -> None:
    """
    Snapshot a native body's pose independently of its empty geometry.

    :param world_with_two_bodies: Native world, parent and unshaped child fixtures.
    """
    world, parent, body = world_with_two_bodies
    parent_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3)
    with world.modify_world():
        world.add_body(parent)
        world.add_connection(
            FixedConnection(
                parent=parent,
                child=body,
                parent_T_connection_expression=parent_T_body,
            )
        )
    bridge = Bridge()
    bridge.attach(world)
    key = str(body.name)
    bridge.publish_bodies({key: body})

    bridge.snapshot()

    assert bridge.get_state()["objects"][key] == [1, 2, 3, 0, 0, 0, 1]
    assert bridge.object_catalog()[0]["shapes"] == []
    assert bridge.object_body(key) is body


def test_live_catalog_does_not_duplicate_the_shapes_classification() -> None:
    """
    The object payload's shape list is sufficient to select geometry rendering.
    """
    body = Body(
        name=PrefixedName("object"),
        visual=ShapeCollection(shapes=[Sphere(radius=0.2)]),
    )
    bridge = Bridge()

    bridge.publish_bodies({str(body.name): body})

    assert set(bridge.object_catalog()[0]) == {"key", "id", "color", "shapes"}


# %% recording geometry
def test_recording_exports_the_catalogs_native_visual_geometry(tmp_path: Path) -> None:
    """
    Export the selected visual collection with its local shape transform.

    :param tmp_path: Directory receiving the recorded mesh asset.
    """
    body = Body(
        name=PrefixedName("object"),
        visual=ShapeCollection(
            shapes=[
                Sphere(
                    radius=0.2,
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(0.1, 0.2, 0.3),
                )
            ]
        ),
        collision=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
    )
    bridge = Bridge()
    bridge.publish_bodies({str(body.name): body})

    payload = _object_entry(bridge.object_metadata[0], [0, 0, 0, 0, 0, 0, 1], tmp_path)

    recorded_mesh = trimesh.load_mesh(tmp_path / payload["mesh"])
    numpy.testing.assert_allclose(
        recorded_mesh.bounds, body.visual.combined_mesh.bounds
    )


@pytest.mark.parametrize(
    "origin",
    [
        HomogeneousTransformationMatrix.from_xyz_rpy(0.1, 0.2, 0.3),
        HomogeneousTransformationMatrix.from_xyz_rpy(roll=0.4, pitch=0.2, yaw=0.7),
    ],
    ids=["translated", "rotated"],
)
def test_recorded_box_preserves_its_local_transform(
    tmp_path: Path, origin: HomogeneousTransformationMatrix
) -> None:
    """
    Retain a box's body-relative transform in the exported recording geometry.

    :param tmp_path: Directory receiving the recorded mesh asset.
    :param origin: Translation or rotation of the box inside its body.
    """
    body = Body(
        name=PrefixedName("object"),
        visual=ShapeCollection(shapes=[Box(scale=Scale(0.2, 0.3, 0.4), origin=origin)]),
    )
    bridge = Bridge()
    bridge.publish_bodies({str(body.name): body})

    payload = _object_entry(bridge.object_metadata[0], [0, 0, 0, 0, 0, 0, 1], tmp_path)

    recorded_mesh = trimesh.load_mesh(tmp_path / payload["mesh"], process=False)
    numpy.testing.assert_allclose(
        recorded_mesh.vertices, body.visual.combined_mesh.vertices, atol=1e-8
    )


@pytest.mark.parametrize(
    "origin,scale",
    [
        (HomogeneousTransformationMatrix.from_xyz_rpy(0.1, 0.2, 0.3), Scale()),
        (
            HomogeneousTransformationMatrix.from_xyz_rpy(roll=0.4, pitch=0.2, yaw=0.7),
            Scale(),
        ),
        (HomogeneousTransformationMatrix(), Scale(2.0, 3.0, 4.0)),
    ],
    ids=["translated", "rotated", "scaled"],
)
def test_recorded_mesh_preserves_its_local_transform_and_scale(
    tmp_path: Path, origin: HomogeneousTransformationMatrix, scale: Scale
) -> None:
    """
    Export each mesh's transformed geometry when copying would lose its pose or size.

    :param tmp_path: Directory containing the source and recorded mesh assets.
    :param origin: Translation or rotation of the mesh inside its body.
    :param scale: Scale applied to the mesh's file geometry.
    """
    source = tmp_path / "shape.obj"
    Box(scale=Scale(0.2, 0.3, 0.4)).mesh.export(source)
    body = Body(
        name=PrefixedName("object"),
        visual=ShapeCollection(
            shapes=[Mesh(filename=str(source), origin=origin, scale=scale)]
        ),
    )
    bridge = Bridge()
    bridge.publish_bodies({str(body.name): body})

    payload = _object_entry(
        bridge.object_metadata[0], [0, 0, 0, 0, 0, 0, 1], tmp_path / "recording"
    )

    recorded_mesh = trimesh.load_mesh(
        tmp_path / "recording" / payload["mesh"], process=False
    )
    numpy.testing.assert_allclose(
        recorded_mesh.vertices, body.visual.combined_mesh.vertices, atol=1e-8
    )

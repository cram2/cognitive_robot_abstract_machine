"""
Native shape collections remain authoritative for live object geometry.
"""

from pathlib import Path

import numpy
import pytest
import trimesh
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Mesh,
    Scale,
    Sphere,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live.bridge import Bridge
from cramera.live.recording_bundle import _object_entry


# %% catalog geometry
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


def test_shapeless_catalog_uses_a_native_placeholder_box() -> None:
    """
    Represent missing geometry with the same native box used for real geometry.
    """
    body = Body(name=PrefixedName("object"))
    bridge = Bridge()

    bridge.publish_bodies({str(body.name): body})

    [shape] = bridge.object_metadata[0].shapes
    assert isinstance(shape, Box)
    assert shape.scale == bridge.configuration.default_object_size


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

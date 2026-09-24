"""
Native shape collections remain authoritative for live object geometry.
"""

from pathlib import Path

import numpy
import trimesh
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Box, Scale, Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live.bridge import Bridge
from cramera.live.recording_bundle import _object_entry


# %% catalog geometry
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
    assert shape.scale == Scale(*Bridge.DEFAULT_OBJECT_SIZE)


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

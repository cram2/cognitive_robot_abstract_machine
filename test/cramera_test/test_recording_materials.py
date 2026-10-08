"""
Recorded meshes retain their materials after local transforms are applied.
"""

from pathlib import Path

import numpy
import pytest
import trimesh
from PIL import Image
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live.bridge import Bridge
from cramera.live.recording_bundle import _object_entry
from cramera.mesh_format import MeshFormat
from cramera.recording_fields import SceneField
from cramera.onboard.bundle_urdf import companion_material_library


# %% native textured geometry
def write_textured_mesh(path: Path, image: Image.Image) -> Path:
    """
    Write a mesh with an embedded texture that survives native geometry loading.

    :param path: Basename used for the generated mesh file.
    :param image: Authored texture to embed in the mesh.
    :return: Path of the generated binary glTF mesh.
    """
    source = path.with_suffix(MeshFormat.GLB)
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=numpy.zeros((len(mesh.vertices), 2)), image=image
    )
    mesh.export(source)
    return source


@pytest.fixture
def textured_image() -> Image.Image:
    """
    A solid image with a distinct colour for checking recorded texture contents.
    """
    return Image.new("RGB", (2, 2), (40, 70, 110))


# %% recorded material references
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
def test_transformed_mesh_publishes_its_recorded_material_library(
    tmp_path: Path,
    textured_image: Image.Image,
    origin: HomogeneousTransformationMatrix,
    scale: Scale,
) -> None:
    """
    Publish the generated material library required to replay a textured mesh.

    :param tmp_path: Directory holding the source mesh and recording bundle.
    :param textured_image: Texture embedded in the source mesh.
    :param origin: Body-relative transform baked into the recorded geometry.
    :param scale: Scale baked into the recorded geometry.
    """
    source = write_textured_mesh(tmp_path / "source", textured_image)
    body = Body(
        name=PrefixedName("object"),
        visual=ShapeCollection(
            shapes=[Mesh(filename=str(source), origin=origin, scale=scale)]
        ),
    )
    bridge = Bridge()
    bridge.publish_bodies({str(body.name): body})
    bundle = tmp_path / "recording"

    payload = _object_entry(bridge.object_metadata[0], [0, 0, 0, 0, 0, 0, 1], bundle)

    material_library = companion_material_library(bundle, payload[SceneField.MESH])
    assert material_library is not None
    assert payload.get(SceneField.MATERIAL_LIBRARY) == material_library


def test_exported_meshes_keep_independent_textures(
    tmp_path: Path, textured_image: Image.Image
) -> None:
    """
    Keep each object's texture when separate source meshes share material names.

    :param tmp_path: Directory holding both source meshes and their recording bundle.
    :param textured_image: Texture of the first object, distinct from the second.
    """
    images = [textured_image, Image.new("RGB", textured_image.size, (190, 20, 60))]
    bodies = [
        Body(
            name=PrefixedName(str(index)),
            visual=ShapeCollection(
                shapes=[
                    Mesh(
                        filename=str(write_textured_mesh(tmp_path / str(index), image)),
                        scale=Scale(2.0, 3.0, 4.0),
                    )
                ]
            ),
        )
        for index, image in enumerate(images)
    ]
    bridge = Bridge()
    bridge.publish_bodies({str(body.name): body for body in bodies})
    bundle = tmp_path / "recording"

    payloads = [
        _object_entry(entry, [0, 0, 0, 0, 0, 0, 1], bundle)
        for entry in bridge.object_metadata
    ]

    for payload, image in zip(payloads, images):
        recorded_mesh = trimesh.load_mesh(bundle / payload[SceneField.MESH])
        numpy.testing.assert_array_equal(recorded_mesh.visual.material.image, image)

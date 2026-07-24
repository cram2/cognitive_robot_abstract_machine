import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import trimesh

from krrood.adapters.json_serializer import from_json, to_json

from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale, Texture


def test_shape():
    mesh = Mesh.from_ply_file(
        ply_file_path=os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "ply",
            "chair.ply",
        ),
        texture_file_path=os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "ply",
            "chair_texture.png",
        ),
    )
    assert mesh.filename.startswith("/tmp/")
    assert mesh.filename.endswith(".obj")
    assert len(mesh.mesh.visual.uv) == 8527


def test_mesh_color_survives_serialization(tmp_path):
    """
    Per-vertex mesh color survives the to_json/from_json round-trip.

    Color travels inside the serialized geometry (re-exported as OBJ, which the
    collision loader and visualizer can read), so a receiver renders it without needing
    the original mesh file.
    """
    source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    source.visual.vertex_colors = np.tile([200, 50, 50, 255], (len(source.vertices), 1))

    mesh = Mesh.from_trimesh(mesh=source, dirname=str(tmp_path), file_type="ply")
    restored = Mesh.from_json(mesh.to_json())

    assert restored.filename.endswith(".obj")
    assert (restored.mesh.visual.vertex_colors[:, :3] == [200, 50, 50]).all()


def test_mesh_scale_change_after_load_is_reflected_by_the_geometry(tmp_path):
    """
    Assigning a new scale after the mesh geometry was already loaded must be reflected
    by the geometry, not silently ignored by the cache.
    """
    source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh = Mesh.from_trimesh(mesh=source, dirname=str(tmp_path), file_type="stl")
    assert np.allclose(mesh.mesh.extents, [1.0, 1.0, 1.0])

    mesh.scale = Scale(1.6, 1.6, 1.6)

    assert np.allclose(mesh.mesh.extents, [1.6, 1.6, 1.6])


def test_mesh_color_is_lost_without_color_preserving_format(tmp_path):
    """
    A format that cannot store per-vertex color (STL) drops it on export.

    The contrast to the PLY round-trip: without a color-preserving format the color
    is lost.
    """
    source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    source.visual.vertex_colors = np.tile([200, 50, 50, 255], (len(source.vertices), 1))

    mesh = Mesh.from_trimesh(mesh=source, dirname=str(tmp_path), file_type="stl")

    assert not (mesh.mesh.visual.vertex_colors[:, :3] == [200, 50, 50]).all()


def test_texture_defaults():
    texture = Texture(file_path="/textures/wood.png")

    assert texture.repeat == (1.0, 1.0)
    assert texture.uniform is False


def test_texture_survives_serialization():
    """
    A texture's fields survive the to_json/from_json round-trip, so a receiver renders the
    same tiling as the sender without needing the original scene.
    """
    texture = Texture(file_path="/textures/wood.png", repeat=(2.0, 3.0), uniform=True)

    restored = from_json(to_json(texture))

    assert restored == texture


def test_textured_primitive_survives_serialization():
    """
    A primitive shape carrying a texture round-trips through serialization with the texture
    intact, rather than silently collapsing to its flat color.
    """
    box = Box(scale=Scale(1.0, 1.0, 1.0), texture=Texture(file_path="/textures/marble.png"))

    restored = Box.from_json(box.to_json())

    assert restored.texture == box.texture
    assert restored == box

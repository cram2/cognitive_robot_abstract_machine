import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import trimesh

from semantic_digital_twin.world_description.geometry import Mesh


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

    Color travels inside the serialized geometry (re-exported as OBJ,
    which the collision loader and visualizer can read), so a receiver
    renders it without needing the original mesh file.
    """
    source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    source.visual.vertex_colors = np.tile([200, 50, 50, 255], (len(source.vertices), 1))

    mesh = Mesh.from_trimesh(mesh=source, dirname=str(tmp_path), file_type="ply")
    restored = Mesh.from_json(mesh.to_json())

    assert restored.filename.endswith(".obj")
    assert (restored.mesh.visual.vertex_colors[:, :3] == [200, 50, 50]).all()


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

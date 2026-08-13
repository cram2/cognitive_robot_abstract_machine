import os
from importlib.resources import files
from pathlib import Path

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

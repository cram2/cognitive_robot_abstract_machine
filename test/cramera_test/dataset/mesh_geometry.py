"""
Native exported meshes with materials resolved through a registered file source.
"""

from pathlib import Path

import numpy
import pytest
import trimesh
from PIL import Image

from semantic_digital_twin.adapters.package_resolver import FileUriResolver
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.mesh_file_storage import MeshFileSources


# %% native textured geometry
@pytest.fixture()
def resolved_textured_mesh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Mesh:
    """
    Export native texture assets and address the mesh through a file URI.

    :param tmp_path: Directory receiving the mesh and its material files.
    :param monkeypatch: Restore the process's registered sources after the test.
    :return: A native mesh whose source reference requires local resolution.
    """
    geometry = trimesh.creation.box(extents=[0.2, 0.3, 0.4])
    geometry.visual = trimesh.visual.TextureVisuals(
        uv=numpy.zeros((len(geometry.vertices), 2)),
        image=Image.new("RGB", (2, 2), (30, 100, 220)),
    )
    mesh = Mesh.from_trimesh(geometry, directory=tmp_path)
    mesh.filename = Path(mesh.filename).as_uri()
    monkeypatch.setattr(MeshFileSources(), "sources", [FileUriResolver()])
    return mesh

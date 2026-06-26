import os
from importlib.resources import files
from pathlib import Path

import numpy as np

from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world_description.geometry import Mesh


def test_recenter_origin_centers_bounding_box():
    # A non-planar point cloud whose bounding box is offset from the origin.
    mesh = Mesh.from_3d_points(
        points_3d=[
            Point3(0, 0, 0),
            Point3(2, 0, 0),
            Point3(0, 4, 0),
            Point3(0, 0, 6),
        ]
    )
    bounding_box = mesh.local_frame_bounding_box
    expected_center = np.array(
        [
            (bounding_box.min_x + bounding_box.max_x) / 2,
            (bounding_box.min_y + bounding_box.max_y) / 2,
            (bounding_box.min_z + bounding_box.max_z) / 2,
        ]
    )

    mesh.recenter_origin()

    np.testing.assert_allclose(mesh.origin.to_position().to_np()[:3], -expected_center)


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

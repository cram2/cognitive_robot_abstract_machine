from __future__ import annotations

import shutil
from importlib.resources import files
from pathlib import Path

import pytest
from plyfile import PlyData

from experiments.scene_generation_experiments.utils import ObjectType
from experiments.scene_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def off_center_mesh_path(tmp_path: Path) -> Path:
    """
    A scene directory holding a PLY whose local origin sits away from its own footprint,
    mimicking a sage10k scan whose origin was not placed at the object's centre.
    """
    objects_directory = tmp_path / "objects"
    objects_directory.mkdir()
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    ply = PlyData.read(str(resources_root / "chair.ply"))
    vertex = ply["vertex"]
    vertex["x"] += 2.0
    vertex["y"] += 3.0
    vertex["z"] += 1.0
    ply.write(str(objects_directory / "test_object.ply"))
    shutil.copy(
        resources_root / "chair_texture.png",
        objects_directory / "test_object_texture.png",
    )
    return tmp_path


def _object_2d() -> RelationalCircuitExperimentObject2D:
    return RelationalCircuitExperimentObject2D(
        object_type=ObjectType.VASE,
        scale=Scale(x=0.3, y=0.3, z=0.5),
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        source_id="test_object",
    )


def _world_with_root() -> tuple[World, Body]:
    world = World()
    root = Body(name=PrefixedName(name="map"))
    with world.modify_world():
        world.add_body(root)
    return world, root


def test_shelf_object_mesh_is_centered_on_its_body_frame(
    off_center_mesh_path: Path,
) -> None:
    """
    A shelf-layer object's mesh, whose own local origin sits away from its footprint,
    must still render centred on its body's TF frame in x/y and resting on it in z, so
    the object's visual geometry lines up with its TF root regardless of where the
    source PLY's local origin happens to be.
    """
    world, root = _world_with_root()
    body = _object_2d().spawn(world, parent=root, mesh_path=off_center_mesh_path)

    minimum_bound, maximum_bound = body.collision.combined_mesh.bounds
    assert (minimum_bound[0] + maximum_bound[0]) / 2 == pytest.approx(0.0, abs=1e-6)
    assert (minimum_bound[1] + maximum_bound[1]) / 2 == pytest.approx(0.0, abs=1e-6)
    assert minimum_bound[2] == pytest.approx(0.0, abs=1e-6)

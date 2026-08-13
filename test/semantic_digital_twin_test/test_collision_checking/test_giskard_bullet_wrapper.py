import os

import giskardpy_bullet_bindings as pb
import pytest
from importlib.resources import files
from pathlib import Path

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.collision_checking.pybullet_collision_detector import (
    clear_cache,
    convert_to_decomposed_obj_and_save_in_tmp,
    create_cache_dir,
)
from semantic_digital_twin.pipeline.mesh_decomposition.bullet_vhacd import (
    BulletVHACDMeshDecomposer,
)
from semantic_digital_twin.pipeline.mesh_decomposition.coacd import COACDMeshDecomposer
from semantic_digital_twin.pipeline.mesh_decomposition.vhacd import VHACDMeshDecomposer
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def cache_dir():
    return create_cache_dir("tmp")


@pytest.fixture
def clean_cache(cache_dir):
    clear_cache(cache_dir)
    yield
    clear_cache(cache_dir)


@pytest.fixture
def non_convex_mesh():
    stl_path = os.path.join(
        Path(files("semantic_digital_twin")).parent.parent,
        "resources",
        "stl",
        "jeroen_cup.stl",
    )
    world_with_stl = STLParser(stl_path).parse()
    body: Body = world_with_stl.root
    return body.collision[0]


@pytest.fixture
def convex_mesh():
    # A box is convex
    return Mesh.box(extents=(1, 1, 1))


@pytest.fixture(
    scope="module",
    params=[COACDMeshDecomposer, VHACDMeshDecomposer, BulletVHACDMeshDecomposer],
    ids=["coacd", "vhacd", "bullet_vhacd"],
)
def mesh_decomposer(request):
    return request.param()


def test_convert_non_convex_mesh_decomposes(
    clean_cache, cache_dir, non_convex_mesh, mesh_decomposer
):
    """
    Test that for a non-convex mesh, the function produces a valid .obj file in the cache directory.
    """
    output_path = convert_to_decomposed_obj_and_save_in_tmp(
        non_convex_mesh, mesh_decomposer=mesh_decomposer, cache_dir=cache_dir
    )

    assert os.path.exists(output_path)
    assert output_path.endswith(".obj")
    assert str(cache_dir) in output_path
    # The decomposer should have produced multiple convex parts, written as multiple
    # 'o' groups in the OBJ. A single-group OBJ would mean decomposition didn't run.
    n_objects = sum(1 for line in open(output_path) if line.startswith("o "))
    assert n_objects > 1
    assert os.path.getsize(output_path) > 0


def test_convert_convex_mesh_saves_directly(clean_cache, cache_dir, convex_mesh):
    """
    Test that for a convex mesh, the function saves it as an .obj file.
    """
    output_path = convert_to_decomposed_obj_and_save_in_tmp(
        convex_mesh, cache_dir=cache_dir
    )

    assert os.path.exists(output_path)
    assert output_path.endswith(".obj")
    assert str(cache_dir) in output_path


def test_convert_caching_behavior(
    clean_cache, cache_dir, non_convex_mesh, mesh_decomposer
):
    """
    Test that calling the function twice with the same mesh returns the same file path
    and does not re-run the decomposition process (same modification time).
    """
    path1 = convert_to_decomposed_obj_and_save_in_tmp(
        non_convex_mesh, mesh_decomposer=mesh_decomposer, cache_dir=cache_dir
    )

    # Capture modification time
    mtime1 = os.path.getmtime(path1)

    path2 = convert_to_decomposed_obj_and_save_in_tmp(
        non_convex_mesh, mesh_decomposer=mesh_decomposer, cache_dir=cache_dir
    )

    assert path1 == path2
    assert os.path.getmtime(path2) == mtime1


def test_generated_obj_is_loadable(
    clean_cache, cache_dir, non_convex_mesh, mesh_decomposer
):
    """
    Verify that the generated file can be loaded via pb.load_convex_shape.
    """
    output_path = convert_to_decomposed_obj_and_save_in_tmp(
        non_convex_mesh, mesh_decomposer=mesh_decomposer, cache_dir=cache_dir
    )
    shape = pb.load_convex_shape(
        output_path, single_shape=False, scaling=pb.Vector3(1, 1, 1)
    )
    assert shape is not None

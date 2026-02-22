import os

import giskardpy_bullet_bindings as pb
import pytest
import trimesh
from pkg_resources import resource_filename

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.collision_checking.pybullet_collision_detector import (
    clear_cache,
    convert_to_decomposed_obj_and_save_in_tmp,
    create_cache_dir,
)
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
        resource_filename("semantic_digital_twin", "../../"),
        "resources",
        "stl",
        "jeroen_cup.stl",
    )
    world_with_stl = STLParser(stl_path).parse()
    body: Body = world_with_stl.root
    mesh = body.collision[0]
    return mesh.mesh


@pytest.fixture
def convex_mesh():
    # A box is convex
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    return mesh


def test_convert_non_convex_mesh_decomposes(clean_cache, cache_dir, non_convex_mesh):
    """
    Test that for a non-convex mesh, the function produces a valid .obj file in the cache directory.
    """
    output_path = convert_to_decomposed_obj_and_save_in_tmp(
        non_convex_mesh, cache_dir=cache_dir
    )

    assert os.path.exists(output_path)
    assert output_path.endswith(".obj")
    assert str(cache_dir) in output_path
    # For non-convex, it should have triggered VHACD (we can't easily check if VHACD ran
    # but we can check if it exists and is not empty)
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


def test_convert_caching_behavior(clean_cache, non_convex_mesh):
    """
    Test that calling the function twice with the same mesh returns the same file path
    and does not re-run the decomposition process (same modification time).
    """
    path1 = convert_to_decomposed_obj_and_save_in_tmp(non_convex_mesh)

    # Capture modification time
    mtime1 = os.path.getmtime(path1)

    path2 = convert_to_decomposed_obj_and_save_in_tmp(non_convex_mesh)

    assert path1 == path2
    assert os.path.getmtime(path2) == mtime1


def test_generated_obj_is_loadable(clean_cache, non_convex_mesh):
    """
    Verify that the generated file can be loaded via pb.load_convex_shape.
    """
    output_path = convert_to_decomposed_obj_and_save_in_tmp(non_convex_mesh)
    shape = pb.load_convex_shape(
        output_path, single_shape=False, scaling=pb.Vector3(1, 1, 1)
    )
    assert shape is not None

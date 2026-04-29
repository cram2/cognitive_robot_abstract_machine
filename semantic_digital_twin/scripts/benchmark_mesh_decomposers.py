import os
import tempfile
import time
from importlib.resources import files
from pathlib import Path

import trimesh

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
from semantic_digital_twin.world_description.world_entity import Body


def load_non_convex_mesh():
    stl_path = os.path.join(
        Path(files("semantic_digital_twin")).parent.parent,
        "resources",
        "stl",
        "jeroen_cup.stl",
    )
    body: Body = STLParser(stl_path).parse().root
    return body.collision[0]


def run(decomposer, mesh, cache_dir):
    clear_cache(cache_dir)
    start = time.perf_counter()
    output_path = convert_to_decomposed_obj_and_save_in_tmp(
        mesh, mesh_decomposer=decomposer, cache_dir=cache_dir
    )
    elapsed = time.perf_counter() - start
    size_kb = os.path.getsize(output_path) / 1024
    n_hulls = sum(1 for line in open(output_path) if line.startswith("o "))
    return elapsed, size_kb, n_hulls


def main():
    cache_dir = create_cache_dir("benchmark_mesh_decomposers")
    non_convex_mesh = load_non_convex_mesh()

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_obj_path = os.path.join(tmp_dir, "input.obj")
        with open(input_obj_path, "w") as f:
            f.write(trimesh.exchange.obj.export_obj(non_convex_mesh.mesh))
        input_size_kb = os.path.getsize(input_obj_path) / 1024
    input_n_vertices = len(non_convex_mesh.mesh.vertices)
    input_n_faces = len(non_convex_mesh.mesh.faces)

    decomposers = [
        ("coacd", COACDMeshDecomposer()),
        ("vhacd", VHACDMeshDecomposer()),
        ("bullet_vhacd", BulletVHACDMeshDecomposer()),
    ]

    results = []
    for name, decomposer in decomposers:
        elapsed, size_kb, n_hulls = run(decomposer, non_convex_mesh, cache_dir)
        results.append((name, elapsed, size_kb, n_hulls))

    clear_cache(cache_dir)

    print("\nDecomposer characterization on jeroen_cup.stl (library defaults):")
    print(
        f"  input mesh: {input_size_kb:.2f} KB as .obj, "
        f"{input_n_vertices} vertices, {input_n_faces} faces"
    )
    print(
        f"{'name':<14} {'time (s)':>10} {'size (KB)':>10}"
        f" {'hulls':>6} {'ms/hull':>9} {'KB/hull':>9}"
    )
    for name, elapsed, size_kb, n_hulls in results:
        ms_per_hull = (elapsed * 1000) / n_hulls
        kb_per_hull = size_kb / n_hulls
        print(
            f"{name:<14} {elapsed:>10.3f} {size_kb:>10.2f}"
            f" {n_hulls:>6} {ms_per_hull:>9.2f} {kb_per_hull:>9.3f}"
        )


if __name__ == "__main__":
    main()

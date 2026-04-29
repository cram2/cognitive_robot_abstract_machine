from dataclasses import dataclass
from enum import StrEnum
from typing import List

import trimesh

from semantic_digital_twin.pipeline.mesh_decomposition.base import MeshDecomposer
from semantic_digital_twin.world_description.geometry import Mesh


class FillMode(StrEnum):
    """
    The Fill Mode determines how V-HACD "solidifies" your mesh before it starts breaking it into convex pieces.
    """

    FLOOD = "flood"
    """
    Fills the volume from the outside in.
    Best for Solid, closed (watertight) objects.
    """

    SURFACE = "surface"
    """
    Only processes the skin of the mesh.
    Best for hollow objects or "open" meshes (like a plane).
    """

    RAYCAST = "raycast"
    """
    Checks interior by shooting rays.
    Best for messy geometry with holes or overlapping parts.
    """


@dataclass
class VHACDMeshDecomposer(MeshDecomposer):
    """
    Decompose meshes using Voxelized Hierarchical Approximate Convex Decomposition.
    Read more about it here: https://pypi.org/project/vhacdx/
    """

    max_convex_hulls: int = 64
    """
    The maximum number of separate convex pieces the algorithm is allowed to generate.
    
    Originally referenced as maxConvexHulls.
    """

    resolution: int = 400000
    """
    The number of "voxels" (3D pixels) used to represent the shape during processing; 
    higher values capture finer details but take longer to compute.
    """

    minimum_volume_percent_error_allowed: float = 1.0
    """
    The threshold for how much the final decomposition can vary in volume from the original shape before the algorithm stops refining.    
    
    Originally referenced as minimumVolumePercentErrorAllowed.
    """

    max_recursion_depth: int = 10
    """
    How many times the algorithm can split a shape into smaller sub-parts to find a better fit.
    
    Originally referenced as maxRecursionDepth.
    """

    shrink_wrap: bool = True
    """
    When enabled, the algorithm "shrinks" the convex hulls to better wrap around the surface of the original mesh.
    
    Originally referenced as shrinkWrap.
    """

    fill_mode: FillMode = FillMode.FLOOD
    """
    Determines how the inside of the mesh is identified; "flood" fills the interior starting from the outside 
    (useful for solid objects).
    
    Originally referenced as fillMode.
    """

    max_vertices_per_convex_hull: int = 64
    """
    The maximum number of points allowed for each individual convex hull generated.
    
    Originally referenced as maxNumVerticesPerCH.
    """

    asynchronous: bool = True
    """
    If true, the decomposition runs in a separate thread, preventing the main program from freezing during calculation.
    
    Originally referenced as asyncACD.
    """

    min_edge_length: int = 2
    """
    The minimum size an edge must be to be considered during the simplification process.
    
    Originally referenced as minEdgeLength.
    """

    find_best_plane: bool = False
    """
    Tells the algorithm to search for the mathematically optimal plane when splitting a shape into two parts.
    
    Originally referenced as findBestPlane.
    """

    def apply_to_mesh(self, mesh: Mesh) -> List[Mesh]:
        decomposed = mesh.mesh.convex_decomposition(
            maxConvexHulls=self.max_convex_hulls,
            resolution=self.resolution,
            minimumVolumePercentErrorAllowed=self.minimum_volume_percent_error_allowed,
            maxRecursionDepth=self.max_recursion_depth,
            shrinkWrap=self.shrink_wrap,
            fillMode=self.fill_mode.value,
            maxNumVerticesPerCH=self.max_vertices_per_convex_hull,
            asyncACD=self.asynchronous,
            minEdgeLength=self.min_edge_length,
            findBestPlane=self.find_best_plane,
        )
        new_geometry = [
            Mesh.from_trimesh(mesh=decomposed_part, origin=mesh.origin)
            for decomposed_part in decomposed
        ]
        return new_geometry

    def apply_to_mesh_and_save(self, mesh: Mesh, output_path: str) -> str:
        parts = self.apply_to_mesh(mesh)
        trimesh.Scene([p.mesh for p in parts]).export(output_path, file_type="obj")
        return output_path

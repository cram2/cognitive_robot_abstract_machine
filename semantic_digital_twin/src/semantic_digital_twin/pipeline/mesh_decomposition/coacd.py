import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, List

import coacd
import numpy as np
import trimesh

from semantic_digital_twin.pipeline.mesh_decomposition.base import MeshDecomposer
from semantic_digital_twin.world_description.geometry import Mesh


class ApproximationMode(StrEnum):
    """
    Approximation shape type
    """

    BOX = "box"
    CONVEX_HULL = "ch"


class PreprocessingMode(StrEnum):
    """
    Manifold preprocessing mode
    """

    AUTO = "auto"
    """
    Automatically chose based on the geometry.
    """

    ON = "on"
    """
    Force turn on the pre-processing
    """

    OFF = "off"
    """
    Force turn off the pre-processing
    """


@dataclass
class COACDMeshDecomposer(MeshDecomposer):
    """
    COACDMeshDecomposer is a class for decomposing complex 3D meshes into simpler convex components
    using the COACD (Convex Optimization for Approximate Convex Decomposition) algorithm. It is
    designed to preprocess, analyze, and process 3D meshes with a focus on efficiency and scalability
    in fields such as robotics, gaming, and simulation.

    Check https://github.com/SarahWeiii/CoACD for further details.
    """

    threshold: float = 0.05
    """
    Determines how much the decomposed parts can deviate from the original shape. 
    A lower value leads to more pieces and higher precision, while a higher value allows for a 
    "looser," simpler approximation.
    """

    max_convex_hull: Optional[int] = None
    """
    Limits the total number of convex pieces generated. 
    If set to None, the algorithm will keep creating pieces until the threshold is met.
    Works only when merge is enabled (may introduce convex hull with a concavity larger than the threshold)
    """

    preprocess_mode: PreprocessingMode = PreprocessingMode.AUTO
    """
    Determines if the mesh is cleaned before processing.
    """

    preprocess_resolution: int = 50
    """
    The grid size used during the initial cleanup phase. 
    Low values are fast but might "melt" away small features of 
    your model before the main decomposition even starts.
    Range: 20 - 100
    """

    resolution: int = 2000
    """
    Defines the sampling density of the input mesh. 
    Think of it like the "DPI" of a printer; a higher resolution captures 
    finer details of the surface but increases the time it takes to calculate the decomposition.
    Range: 1 000 - 10 000
    """

    search_nodes: int = 20
    """
    Max number of child nodes in the monte carlo tree search (10 - 40).
    """

    search_iterations: int = 150
    """
    Number of search iterations in the monte carlo tree search (60 - 2000).
    """

    search_depth: int = 3
    """
    Maximum search depth in the monte carlo tree search (2 - 7).
    """

    pca: bool = False
    """
    Enable PCA pre-processing.
    Stands for Principal Component Analysis. 
    If enabled, it uses the orientation of the object’s volume to align the decomposition cuts. 
    It’s great for long, thin objects but can sometimes ignore local symmetry.
    """

    merge: bool = True
    """
    Enable merge postprocessing.
    After splitting the mesh, 
    the algorithm checks if any adjacent pieces are "convex enough" to be joined back together. 
    This helps keep the final count of pieces low without sacrificing much accuracy.
    """

    max_convex_hull_vertices: Optional[int] = None
    """
    Maximum vertex value for each convex hull, only when decimate is enabled.
    Limits how many "corners" (vertices) each individual convex piece can have. 
    This is vital for physics engines (like Bullet or PhysX) which often have a hard limit (e.g., 32 or 64 vertices)
    for real-time performance.
    """

    extrude_margin: Optional[float] = None
    """
    Extrude margin, only when extrude is enabled.
    Adds a small "padding" or thickness to the generated pieces. 
    Useful if you're experiencing "tunneling" 
    in physics simulations where objects pass through each other.
    """

    approximation_mode: ApproximationMode = ApproximationMode.BOX
    """
    Defines the shape of the primitives used during the search phase.
    """

    seed: int = field(default_factory=lambda: np.random.randint(2**32))
    """
    Random seed used for sampling.
    """

    def apply_to_mesh(self, mesh: Mesh) -> List[Mesh]:
        """
        Apply the COACD mesh decomposition to a given mesh.
        Returns a list of TriangleMesh objects representing the decomposed convex parts.
        """
        new_geometry = []

        trimesh_mesh = mesh.mesh
        if mesh.scale.x == mesh.scale.y == mesh.scale.z:
            trimesh_mesh.apply_scale(mesh.scale.x)
        else:
            logging.warning("Ambiguous scale for mesh, using uniform scale only.")

        coacd_mesh = coacd.Mesh(trimesh_mesh.vertices, trimesh_mesh.faces)
        if self.max_convex_hull is not None:
            max_convex_hull = self.max_convex_hull
        else:
            max_convex_hull = -1
        parts = coacd.run_coacd(
            mesh=coacd_mesh,
            apx_mode=str(self.approximation_mode),
            threshold=self.threshold,
            max_convex_hull=max_convex_hull,
            preprocess_mode=str(self.preprocess_mode),
            resolution=self.resolution,
            mcts_nodes=self.search_nodes,
            mcts_iterations=self.search_iterations,
            mcts_max_depth=self.search_depth,
            pca=self.pca,
            merge=self.merge,
            decimate=self.max_convex_hull_vertices is not None,
            max_ch_vertex=self.max_convex_hull_vertices or 256,
            extrude=self.extrude_margin is not None,
            extrude_margin=self.extrude_margin or 0.01,
            seed=self.seed,
        )

        for vs, fs in parts:
            new_geometry.append(
                Mesh.from_trimesh(mesh=trimesh.Trimesh(vs, fs), origin=mesh.origin)
            )

        return new_geometry

    def apply_to_mesh_and_save(self, mesh: Mesh, output_path: str) -> str:
        parts = self.apply_to_mesh(mesh)
        trimesh.Scene([p.mesh for p in parts]).export(output_path, file_type="obj")
        return output_path

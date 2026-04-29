import logging
import os
import tempfile
from dataclasses import dataclass
from typing import List

import giskardpy_bullet_bindings as bullet
import trimesh

from semantic_digital_twin.pipeline.mesh_decomposition.base import MeshDecomposer
from semantic_digital_twin.world_description.geometry import Mesh

logger = logging.getLogger(__name__)


@dataclass
class BulletVHACDMeshDecomposer(MeshDecomposer):
    """
    Decompose meshes using Voxelized Hierarchical Approximate Convex Decomposition
    via the Bullet binding (``giskardpy_bullet_bindings.vhacd``).

    Any parameter left at ``-1`` is forwarded as-is and tells V-HACD to use its
    internal default for that parameter.
    """

    concavity: float = -1
    """
    Maximum concavity of each generated convex hull.
    """

    alpha: float = -1
    """
    Bias toward symmetry-plane clipping during decomposition.
    """

    beta: float = -1
    """
    Bias toward revolution-axis clipping during decomposition.
    """

    gamma: float = -1
    """
    Maximum allowed concavity during the merge step.
    """

    min_volume_per_convex_hull: float = -1
    """
    Minimum volume to add vertices to a convex hull.
    """

    resolution: int = -1
    """
    Voxel grid resolution used to represent the mesh during processing.
    """

    max_vertices_per_convex_hull: int = -1
    """
    Maximum number of vertices allowed per generated convex hull.
    """

    depth: int = -1
    """
    Maximum recursion depth of the decomposition.
    """

    plane_downsampling: int = -1
    """
    Granularity of the search for the best clipping plane.
    """

    convex_hull_downsampling: int = -1
    """
    Precision of the convex-hull generation during clipping plane search.
    """

    pca: int = -1
    """
    Whether to enable Principal Component Analysis preprocessing (1) or not (0).
    """

    mode: int = -1
    """
    Decomposition mode: 0 = voxel-based, 1 = tetrahedron-based.
    """

    convex_hull_approximation: int = -1
    """
    Whether to approximate the generated convex hulls (1) or not (0).
    """

    log_path: str = "/tmp/vhacd.log"
    """
    Path to a file where V-HACD writes its log output.
    """

    def apply_to_mesh(self, mesh: Mesh) -> List[Mesh]:
        """
        Decompose the mesh and return the resulting convex parts as ``Mesh`` objects.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "output.obj")
            self.apply_to_mesh_and_save(mesh, output_path)
            return self._load_convex_parts(output_path, origin=mesh.origin)

    def apply_to_mesh_and_save(self, mesh: Mesh, output_path: str) -> str:
        """
        Decompose the mesh and write the resulting multi-hull .obj directly to
        ``output_path``.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.obj")
            with open(input_path, "w") as f:
                f.write(trimesh.exchange.obj.export_obj(mesh.mesh))
            bullet.vhacd(
                input_path,
                output_path,
                self.log_path,
                self.concavity,
                self.alpha,
                self.beta,
                self.gamma,
                self.min_volume_per_convex_hull,
                self.resolution,
                self.max_vertices_per_convex_hull,
                self.depth,
                self.plane_downsampling,
                self.convex_hull_downsampling,
                self.pca,
                self.mode,
                self.convex_hull_approximation,
            )
        return output_path

    @staticmethod
    def _load_convex_parts(path: str, origin) -> List[Mesh]:
        loaded = trimesh.load(path, process=False)
        return [
            Mesh.from_trimesh(mesh=part, origin=origin)
            for part in loaded.split(only_watertight=False)
        ]

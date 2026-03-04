import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, List

import coacd
import numpy as np
import trimesh

from semantic_digital_twin.pipeline.pipeline import Step
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import TriangleMesh, Shape, Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


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
class MeshDecomposer(Step, ABC):
    """
    MeshDecomposer is an abstract base class for decomposing complex 3D meshes into simpler convex components.
    It provides methods to apply the decomposition to meshes, shapes, bodies, and entire worlds.
    Subclasses should implement the `apply_to_mesh` method to define the specific decomposition algorithm.
    3D meshes are represented using the `trimesh` library, and the decomposed parts are returned as a list of `TriangleMesh` objects.
    """

    @abstractmethod
    def apply_to_mesh(self, mesh: Mesh) -> List[TriangleMesh]:
        """
        Apply the mesh decomposition to a given mesh.
        Returns a list of TriangleMesh objects representing the decomposed convex parts.
        """
        ...

    def apply_to_shape(self, shape: Shape) -> List[TriangleMesh]:
        """
        Apply the mesh decomposition to a given shape.
        If the shape is a Mesh, it will be decomposed into multiple TriangleMesh objects.
        Otherwise, the shape will be returned as is in a list.
        """
        new_geometry = []
        if isinstance(shape, Mesh):
            new_geometry = self.apply_to_mesh(shape)
        else:
            new_geometry.append(shape)

        return new_geometry

    def apply_to_body(self, body: Body) -> Body:
        """
        Apply the mesh decomposition to all shapes in a given body.
        The body's collision shapes will be replaced with the decomposed shapes.
        Returns the modified body.
        """
        new_geometry = []
        for shape in body.visual:
            decomposed_shapes = self.apply_to_shape(shape)
            new_geometry.extend(decomposed_shapes)

        body.collision = ShapeCollection(new_geometry)
        return body

    def _apply(self, world: World) -> World:
        """
        Apply the mesh decomposition to all bodies in a given world.
        Each body's collision shapes will be replaced with the decomposed shapes.
        Returns the modified world.
        """
        for body in world.bodies:
            self.apply_to_body(body)

        return world


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
    Concavity threshold for terminating the decomposition (0.01 - 1)
    """

    max_convex_hull: Optional[int] = None
    """
    Maximum number of convex hulls in the result. 
    Works only when merge is enabled (may introduce convex hull with a concavity larger than the threshold)
    """

    preprocess_mode: PreprocessingMode = PreprocessingMode.AUTO
    """
    Manifold preprocessing mode.
    """

    preprocess_resolution: int = 50
    """
    Resolution for manifold preprocess (20~100)
    """

    resolution: int = 2000
    """
    Sampling resolution for Hausdorff distance calculation (1 000 - 10 000)
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
    Enable PCA pre-processing
    """

    merge: bool = True
    """
    Enable merge postprocessing.
    """

    max_convex_hull_vertices: Optional[int] = None
    """
    Maximum vertex value for each convex hull, only when decimate is enabled.
    """

    extrude_margin: Optional[float] = None
    """
    Extrude margin, only when extrude is enabled
    """

    approximation_mode: ApproximationMode = ApproximationMode.BOX
    """
    Approximation mode to use.
    """

    seed: int = field(default_factory=lambda: np.random.randint(2**32))
    """
    Random seed used for sampling.
    """

    def apply_to_mesh(self, mesh: Mesh) -> List[TriangleMesh]:
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
                TriangleMesh(mesh=trimesh.Trimesh(vs, fs), origin=mesh.origin)
            )

        return new_geometry

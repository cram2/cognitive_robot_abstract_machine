"""
Exceptions raised while decomposing meshes into convex parts.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

from semantic_digital_twin.world_description.geometry import Mesh

# %% meshes without convex parts


@dataclass(kw_only=True)
class MeshWithoutConvexPartsError(DataclassException):
    """
    Raised when a mesh can be decomposed into no convex part: the decomposition finds
    none, and its vertices span no plane a convex hull could be built on.
    """

    mesh: Mesh
    """
    The mesh that has no convex parts.
    """

    def error_message(self) -> str:
        return (
            f"The mesh with {len(self.mesh.mesh.vertices)} vertices has no convex "
            f"parts: its vertices are empty, a single point or a line."
        )

    def suggest_correction(self) -> str:
        return (
            "Remove the mesh from the body's collision shapes, or give it vertices "
            "that span at least a plane."
        )

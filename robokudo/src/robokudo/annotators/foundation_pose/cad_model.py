"""
CAD models of the object classes whose poses are estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import trimesh


# %% Configuration


@dataclass
class CadModelSource:
    """
    A mesh file showing the objects of one class.
    """

    class_name: str
    """
    Class name of the :class:`robokudo.types.annotation.Classification` the objects are
    detected with.
    """

    mesh_file: Path
    """
    The mesh file, in any format :func:`trimesh.load` reads.
    """

    scale_to_meters: float = 1.0
    """
    Factor that converts the units of :attr:`mesh_file` to meters.
    """


# %% Loaded CAD model


@dataclass
class CadModel:
    """
    The mesh of an object class, in meters.
    """

    class_name: str
    """
    Class name of the :class:`robokudo.types.annotation.Classification` the objects are
    detected with.
    """

    mesh: trimesh.Trimesh
    """
    The mesh, in meters.
    """

    estimator_mesh_file: Path
    """
    File holding exactly :attr:`mesh`, as one mesh in meters.
    """

    @classmethod
    def from_source(
        cls, source: CadModelSource, rewritten_mesh_directory: Path
    ) -> CadModel:
        """
        Load a mesh file, merging all its meshes into one and converting it to meters.

        :param source: The mesh file to load.
        :param rewritten_mesh_directory: Directory to write the mesh to if it differs from
            the content of :attr:`CadModelSource.mesh_file`.
        :return: The CAD model of the source.
        """
        loaded = trimesh.load(source.mesh_file)
        if isinstance(loaded, trimesh.Trimesh):
            mesh, is_single_mesh = loaded, True
        else:
            mesh, is_single_mesh = loaded.to_mesh(), len(loaded.geometry) == 1
        mesh.apply_scale(source.scale_to_meters)
        if is_single_mesh and source.scale_to_meters == 1.0:
            return cls(
                class_name=source.class_name,
                mesh=mesh,
                estimator_mesh_file=source.mesh_file,
            )

        rewritten_mesh_file = (
            rewritten_mesh_directory / source.class_name / source.mesh_file.name
        )
        rewritten_mesh_file.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(rewritten_mesh_file)
        return cls(
            class_name=source.class_name,
            mesh=mesh,
            estimator_mesh_file=rewritten_mesh_file,
        )

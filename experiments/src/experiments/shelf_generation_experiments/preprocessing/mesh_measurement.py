from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import trimesh

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point2
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox


# %% mesh measurement
@dataclass(frozen=True)
class CorrectedPosition:
    """
    An object's horizontal position after the mesh-centring correction, with the
    provenance of that position.
    """

    position: Point2
    """
    The object's position in world coordinates.
    """

    is_mesh_corrected: bool
    """
    Whether :attr:`position` is the mesh's measured bounding-box centre rather than the
    dataset's unmodified recorded position.
    """


@dataclass
class MeshMeasurements:
    """
    Measures cached meshes, so an object's recorded position can be corrected onto its
    mesh's true horizontal centre and a shelf's real base and top can be located.

    A sage10k object's recorded position is its mesh's local origin, which the dataset
    does not guarantee to be that mesh's centre -- much as a room's recorded position is
    its lower-left corner.
    """

    source_id_to_path: dict[str, Path]
    """
    Maps a mesh's source id to the cached scene directory holding it, as returned by
    :func:`~experiments.shelf_generation_experiments.utils.build_source_id_to_path`.
    """

    _bounds_by_source_id: dict[str, Optional[VolumetricBoundingBox]] = field(
        default_factory=dict
    )
    """
    Memoizes each measured mesh, since many objects share one asset.

    ``None`` records that a mesh was not available to measure.
    """

    @property
    def measured_mesh_count(self) -> int:
        """
        How many distinct meshes were actually loaded and measured.
        """
        return sum(bounds is not None for bounds in self._bounds_by_source_id.values())

    def corrected_position(
        self, source_id: str, position: Point2, yaw_degrees: float
    ) -> CorrectedPosition:
        """
        Correct *position* onto the true centre of *source_id*'s mesh.

        Falls back to *position* unchanged, flagged as uncorrected, when the mesh is not
        cached locally, so preprocessing still runs on a partial mesh cache without
        silently passing off uncorrected data as corrected.

        :param source_id: Identifies the object's mesh asset.
        :param position: The object's recorded world position.
        :param yaw_degrees: The object's own yaw, which the mesh-local offset is rotated
            by to reach world axes.
        :return: The corrected position and whether the mesh supplied it.
        """
        bounds = self.bounds(source_id)
        if bounds is None:
            return CorrectedPosition(position=position, is_mesh_corrected=False)
        local_offset = Point2(
            x=(bounds.min_x + bounds.max_x) / 2,
            y=(bounds.min_y + bounds.max_y) / 2,
        )
        world_T_local = HomogeneousTransformationMatrix.from_xyz_rpy(
            yaw=math.radians(yaw_degrees)
        )
        world_offset = local_offset.transform(world_T_local)
        return CorrectedPosition(
            position=Point2(
                x=position.x + world_offset.x, y=position.y + world_offset.y
            ),
            is_mesh_corrected=True,
        )

    def bounds(self, source_id: str) -> Optional[VolumetricBoundingBox]:
        """
        The measurements of *source_id*'s mesh, loading it on first request.

        :param source_id: Identifies the mesh asset to measure.
        :return: The mesh's measurements, or ``None`` when it is not cached.
        """
        if source_id not in self._bounds_by_source_id:
            self._bounds_by_source_id[source_id] = self._measure(source_id)
        return self._bounds_by_source_id[source_id]

    def _measure(self, source_id: str) -> Optional[VolumetricBoundingBox]:
        """
        Load *source_id*'s mesh and measure its bounding box.

        :param source_id: Identifies the mesh asset to measure.
        :return: The mesh's measurements, or ``None`` when it is not cached.
        """
        return MeshMeasurements._load_mesh_bounds(
            source_id, self.source_id_to_path.get(source_id)
        )

    @staticmethod
    def _load_mesh_bounds(
        source_id: str, scene_directory: Optional[Path]
    ) -> Optional[VolumetricBoundingBox]:
        """
        Load and measure *source_id*'s mesh from *scene_directory*.

        A staticmethod purely for symmetry with the rest of this class's read-only
        helpers; it needs no ``self``.

        :param source_id: Identifies the mesh asset to measure.
        :param scene_directory: The cached scene directory holding the mesh, or ``None``
            when it is not cached locally.
        :return: The mesh's measurements, or ``None`` when it is not cached.
        """
        if scene_directory is None:
            return None
        mesh = trimesh.load(
            str(scene_directory / "objects" / f"{source_id}.ply"), process=False
        )
        minimum_bound, maximum_bound = mesh.bounds
        # VolumetricBoundingBox.from_mesh() passes mesh.bounds straight through as
        # numpy scalars, which pass for floats until PostgreSQL is handed their repr
        # instead of a number, so they are converted here rather than at every place a
        # measurement ends up in a stored field.
        return VolumetricBoundingBox(
            min_x=float(minimum_bound[0]),
            min_y=float(minimum_bound[1]),
            min_z=float(minimum_bound[2]),
            max_x=float(maximum_bound[0]),
            max_y=float(maximum_bound[1]),
            max_z=float(maximum_bound[2]),
            # The mesh's own frame has no reference frame of its own to carry; this
            # placeholder is never read back, since nothing here calls the
            # AxisAlignedBox operations that dereference origin.reference_frame.
            origin=HomogeneousTransformationMatrix(),
        )

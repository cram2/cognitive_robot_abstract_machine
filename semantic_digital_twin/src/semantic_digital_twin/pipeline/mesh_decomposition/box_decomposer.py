from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import ndimage

from semantic_digital_twin.pipeline.mesh_decomposition.base import MeshDecomposer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Mesh, Box, Scale


@dataclass(frozen=True)
class FrozenBox:
    """
    A frozen non-oriented box optimized for the BoxDecomposer.

    .. important::

        Use this only in the context of the BoxDecomposer.
        If you want Boxes for other purposes than this decomposition,
        use the :py:class:`semantic_digital_twin.world_description.geometry.Box` class.
    """

    x: float
    """
    X position.
    """

    y: float
    """
    Y position.
    """

    z: float
    """
    Z position.
    """

    scale: Scale
    """
    The size of the box.
    """


@dataclass(frozen=True)
class FrozenIndexBox:
    """
    A box defined by voxel indices.

    .. important::

        Use this only in the context of the BoxDecomposer.
    """

    x0: int
    """
    The start index along the x-axis.
    """

    x1: int
    """
    The end index along the x-axis.
    """

    y0: int
    """
    The start index along the y-axis.
    """

    y1: int
    """
    The end index along the y-axis.
    """

    z0: int
    """
    The start index along the z-axis.
    """

    z1: int
    """
    The end index along the z-axis.
    """

    def dims(self) -> tuple[int, int, int]:
        """
        Calculate the dimensions of the box in voxels.

        :return: The dimensions along each axis.
        """
        return self.x1 - self.x0, self.y1 - self.y0, self.z1 - self.z0

    def volume_vox(self) -> int:
        """
        Calculate the volume of the box in voxels.

        :return: The volume.
        """
        dx, dy, dz = self.dims()
        return max(0, dx) * max(0, dy) * max(0, dz)

    def thin_axis(self) -> int:
        """
        Identify the axis with the smallest dimension.

        :return: The index of the thinnest axis.
        """
        dx, dy, dz = self.dims()
        return int(np.argmin([dx, dy, dz]))

    def thickness_vox(self) -> int:
        """
        Calculate the thickness of the box in voxels.

        :return: The smallest dimension.
        """
        dx, dy, dz = self.dims()
        return int(min(dx, dy, dz))

    def planar_area_vox(self) -> int:
        """
        Calculate the area of the largest face in voxels.

        :return: The planar area.
        """
        dx, dy, dz = sorted(self.dims())
        return int(dy * dz)

    def to_frozen_box(self, pitch: float, origin: np.ndarray) -> FrozenBox:
        """
        Convert the index-based box to a metric frozen box.

        :param pitch: The size of one voxel.
        :param origin: The origin of the voxel grid.
        :return: The converted box.
        """
        mins = origin + pitch * np.array([self.x0, self.y0, self.z0], dtype=float)
        maxs = origin + pitch * np.array([self.x1, self.y1, self.z1], dtype=float)
        position = (mins + maxs) / 2.0
        size = maxs - mins
        return FrozenBox(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            scale=Scale(x=float(size[0]), y=float(size[1]), z=float(size[2])),
        )

    def intersection_volume_vox(self, other: FrozenIndexBox) -> int:
        """
        Calculate the intersection volume of this index-based box with another index based box in voxels.

        :param other: The second box.
        :return: The intersection volume.
        """
        dx = max(0, min(self.x1, other.x1) - max(self.x0, other.x0))
        dy = max(0, min(self.y1, other.y1) - max(self.y0, other.y0))
        dz = max(0, min(self.z1, other.z1) - max(self.z0, other.z0))
        return dx * dy * dz

    @classmethod
    def create_from_axis_info(
        cls,
        axis: int,
        start: int,
        stop: int,
        dim1_min: int,
        dim1_max: int,
        dim2_min: int,
        dim2_max: int,
    ) -> FrozenIndexBox:
        """
        Create an IndexBox based on axis and coordinates.

        :param axis: The thin axis of the box.
        :param start: The start index of the thin axis.
        :param stop: The end index of the thin axis.
        :param dim1_min: The start index of the first planar dimension.
        :param dim1_max: The end index of the first planar dimension.
        :param dim2_min: The start index of the second planar dimension.
        :param dim2_max: The end index of the second planar dimension.
        :return: The created box.
        """
        if axis == 0:
            return FrozenIndexBox(start, stop, dim1_min, dim1_max, dim2_min, dim2_max)
        elif axis == 1:
            return FrozenIndexBox(dim1_min, dim1_max, start, stop, dim2_min, dim2_max)
        else:
            return FrozenIndexBox(dim1_min, dim1_max, dim2_min, dim2_max, start, stop)


def greedy_merge_boxes(
    occupancy: np.ndarray, pitch: float, origin: np.ndarray
) -> list[FrozenBox]:
    """
    Greedily merge occupied voxels into boxes.

    :param occupancy: The 3D occupancy grid.
    :param pitch: The size of one voxel.
    :param origin: The 3D origin of the occupancy grid.
    :return: A list of FrozenBoxes.
    """
    remaining_occupancy = occupancy.copy()
    nx, ny, nz = remaining_occupancy.shape
    boxes: list[FrozenBox] = []

    for z in range(nz):
        for y in range(ny):
            x = 0
            while x < nx:
                if not remaining_occupancy[x, y, z]:
                    x += 1
                    continue

                x1 = _find_extent(remaining_occupancy, x, y, z, axis=0)
                y1 = _find_extent(remaining_occupancy, x, y, z, axis=1, x_limit=x1)
                z1 = _find_extent(
                    remaining_occupancy, x, y, z, axis=2, x_limit=x1, y_limit=y1
                )

                remaining_occupancy[x:x1, y:y1, z:z1] = False

                index_box = FrozenIndexBox(x, x1, y, y1, z, z1)
                boxes.append(index_box.to_frozen_box(pitch, origin))

                x = x1

    return boxes


def _find_extent(
    occupancy: np.ndarray,
    x: int,
    y: int,
    z: int,
    axis: int,
    x_limit: int = -1,
    y_limit: int = -1,
) -> int:
    """
    Find the extent of a box along a given axis in the occupancy grid.

    :param occupancy: The 3D occupancy grid.
    :param x: The start x-coordinate.
    :param y: The start y-coordinate.
    :param z: The start z-coordinate.
    :param axis: The axis along which to find the extent.
    :param x_limit: The maximum x-coordinate (exclusive) to consider.
    :param y_limit: The maximum y-coordinate (exclusive) to consider.
    :return: The end index (exclusive) along the axis.
    """
    nx, ny, nz = occupancy.shape
    if axis == 0:
        x1 = x + 1
        while x1 < nx and occupancy[x1, y, z]:
            x1 += 1
        return x1
    elif axis == 1:
        y1 = y + 1
        while y1 < ny and occupancy[x:x_limit, y1, z].all():
            y1 += 1
        return y1
    else:
        z1 = z + 1
        while z1 < nz and occupancy[x:x_limit, y:y_limit, z1].all():
            z1 += 1
        return z1


def clean_occupancy(occupancy: np.ndarray, fill_thin_holes: bool = True) -> np.ndarray:
    """
    Clean the occupancy grid by filling holes.

    :param occupancy: The 3D occupancy grid.
    :param fill_thin_holes: Whether to fill 1-voxel thin holes.
    :return: The cleaned occupancy grid.
    """
    cleaned_occupancy = occupancy.copy().astype(bool)
    cleaned_occupancy = ndimage.binary_fill_holes(cleaned_occupancy)

    if not fill_thin_holes:
        return cleaned_occupancy.astype(bool)

    empty_mask = ~cleaned_occupancy
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, num_labels = ndimage.label(empty_mask, structure=structure)
    if num_labels == 0:
        return cleaned_occupancy.astype(bool)

    slices = ndimage.find_objects(labels)
    for label_idx, slice_obj in enumerate(slices, start=1):
        if slice_obj is None:
            continue
        sx, sy, sz = slice_obj
        dx = sx.stop - sx.start
        dy = sy.stop - sy.start
        dz = sz.stop - sz.start

        # Optional crack fill; disable with --no-thin-hole-fill if it fattens boards too much.
        if dx <= 1 or dy <= 1 or dz <= 1:
            cleaned_occupancy[labels == label_idx] = True

    return cleaned_occupancy.astype(bool)


def detect_planar_boards(
    occupancy: np.ndarray,
    max_thickness_voxels: int = 2,
    min_span_voxels: int = 3,
    min_fill_ratio: float = 0.75,
) -> list[FrozenIndexBox]:
    """
    Detect planar board-like structures in the occupancy grid.

    :param occupancy: The 3D occupancy grid.
    :param max_thickness_voxels: Maximum thickness of the board in voxels.
    :param min_span_voxels: Minimum span of the board in other dimensions.
    :param min_fill_ratio: Minimum ratio of occupied voxels to bounding box area.
    :return: A list of candidate IndexBoxes.
    """
    candidates: list[FrozenIndexBox] = []

    for axis in range(3):
        candidates.extend(
            _detect_boards_along_axis(
                occupancy,
                axis,
                max_thickness_voxels,
                min_span_voxels,
                min_fill_ratio,
            )
        )

    return candidates


def _detect_boards_along_axis(
    occupancy: np.ndarray,
    axis: int,
    max_thickness_voxels: int,
    min_span_voxels: int,
    min_fill_ratio: float,
) -> list[FrozenIndexBox]:
    """
    Detect board-like structures along a specific axis.

    :param occupancy: The 3D occupancy grid.
    :param axis: The axis (0, 1, or 2) along which boards are thin.
    :param max_thickness_voxels: Maximum thickness of the board in voxels.
    :param min_span_voxels: Minimum span of the board in other dimensions.
    :param min_fill_ratio: Minimum ratio of occupied voxels to bounding box area.
    :return: A list of candidate IndexBoxes.
    """
    num_voxels = occupancy.shape[axis]
    candidates: list[FrozenIndexBox] = []

    for start_voxel in range(num_voxels):
        for thickness in range(1, max_thickness_voxels + 1):
            end_voxel = start_voxel + thickness
            if end_voxel > num_voxels:
                continue

            slab = _get_slab(occupancy, axis, start_voxel, end_voxel)
            if not slab.any():
                continue

            projection = slab.any(axis=axis)
            labels, num_labels = ndimage.label(projection)
            for label in range(1, num_labels + 1):
                mask = labels == label
                other_indices = np.where(mask)
                if other_indices[0].size == 0:
                    continue

                dim1_min, dim1_max = (
                    int(other_indices[0].min()),
                    int(other_indices[0].max()) + 1,
                )
                dim2_min, dim2_max = (
                    int(other_indices[1].min()),
                    int(other_indices[1].max()) + 1,
                )

                if (dim1_max - dim1_min) < min_span_voxels or (
                    dim2_max - dim2_min
                ) < min_span_voxels:
                    continue

                if (
                    _calculate_fill_ratio(mask[dim1_min:dim1_max, dim2_min:dim2_max])
                    < min_fill_ratio
                ):
                    continue

                candidates.append(
                    FrozenIndexBox.create_from_axis_info(
                        axis,
                        start_voxel,
                        end_voxel,
                        dim1_min,
                        dim1_max,
                        dim2_min,
                        dim2_max,
                    )
                )

    return candidates


def _get_slab(occupancy: np.ndarray, axis: int, start: int, stop: int) -> np.ndarray:
    """
    Extract a slab from the occupancy grid along the given axis.
    A slab is a 2D slice along the given axis.

    :param occupancy: The 3D occupancy grid.
    :param axis: The axis along which to extract the slab.
    :param start: The start index.
    :param stop: The end index.
    :return: The extracted slab.
    """
    if axis == 0:
        return occupancy[start:stop, :, :]
    elif axis == 1:
        return occupancy[:, start:stop, :]
    else:
        return occupancy[:, :, start:stop]


def _calculate_fill_ratio(mask2d: np.ndarray) -> float:
    """
    Calculate the fill ratio of a 2D mask.

    :param mask2d: The 2D mask to check.
    :return: The ratio of filled voxels.
    """
    if mask2d.size == 0:
        return 0.0
    return float(mask2d.mean())


def deduplicate_index_boxes(
    candidates: list[FrozenIndexBox],
    overlap_ratio_threshold: float = 0.8,
) -> list[FrozenIndexBox]:
    """
    Deduplicate IndexBox candidates.

    Prefer board candidates with:
    1) larger planar area
    2) thinner thickness
    3) then larger volume

    This avoids keeping a too-thick candidate when a thinner candidate covers
    the same board.

    :param candidates: A list of candidate IndexBoxes.
    :param overlap_ratio_threshold: The threshold for overlapping boxes to be considered duplicates.
    :return: A list of deduplicated IndexBoxes.
    """
    ordered_candidates = sorted(
        candidates,
        key=lambda box: (
            -box.planar_area_vox(),  # prefer large boards
            box.thickness_vox(),  # prefer thinner boards
            -box.volume_vox(),  # tie-breaker
            box.x0,
            box.y0,
            box.z0,
        ),
    )
    kept_boxes: list[FrozenIndexBox] = []

    for candidate in ordered_candidates:
        candidate_volume = candidate.volume_vox()
        if candidate_volume == 0:
            continue

        discard = False
        for previous in kept_boxes:
            intersection_volume = candidate.intersection_volume_vox(previous)

            # If most of this candidate is already covered, drop it.
            if intersection_volume / candidate_volume >= overlap_ratio_threshold:
                discard = True
                break

            # Extra rule: if same planar support but candidate is thicker and overlaps strongly,
            # drop the thicker one.
            if _is_redundant_thicker_box(candidate, previous, intersection_volume):
                discard = True
                break

        if not discard:
            kept_boxes.append(candidate)

    return kept_boxes


def _is_redundant_thicker_box(
    candidate: FrozenIndexBox,
    previous: FrozenIndexBox,
    intersection_volume: int,
) -> bool:
    """
    Check if the candidate is a redundant thicker box compared to a previous one.

    :param candidate: The candidate box to check.
    :param previous: A previously kept box.
    :param intersection_volume: The volume of intersection between the two boxes.
    :return: True if the candidate is redundant and thicker.
    """
    same_thin_axis = candidate.thin_axis() == previous.thin_axis()
    candidate_area = candidate.planar_area_vox()
    previous_area = previous.planar_area_vox()
    similar_planar_area = (
        min(candidate_area, previous_area) / max(candidate_area, previous_area) >= 0.9
    )

    if (
        same_thin_axis
        and similar_planar_area
        and candidate.thickness_vox() >= previous.thickness_vox()
    ):
        min_volume = min(candidate.volume_vox(), previous.volume_vox())
        if intersection_volume / min_volume >= 0.6:
            return True
    return False


def subtract_index_boxes_from_occupancy(
    occupancy: np.ndarray, boxes: list[FrozenIndexBox]
) -> np.ndarray:
    """
    Subtract boxes from the occupancy grid.

    :param occupancy: The 3D occupancy grid.
    :param boxes: A list of IndexBoxes to subtract.
    :return: The occupancy grid with the boxes removed.
    """
    remaining_occupancy = occupancy.copy()
    for box in boxes:
        remaining_occupancy[box.x0 : box.x1, box.y0 : box.y1, box.z0 : box.z1] = False
    return remaining_occupancy


@dataclass
class BoxDecomposer(MeshDecomposer):
    """
    Decompose a mesh into boxes using voxelization.

    This is very efficient and works well for blocky furniture.
    Blocky furniture is furniture that has many box-shaped parts, e.g. IKEA shelves.

    This works poorly for non-blocky furniture.

    A board in this context is something like a board (the supporting surfaces that hold books) in a bookshelf.

    The algorithm works by first voxelizing the mesh and removing thin voxel cracks.
    Next, it detects planar boards and then for each axis (X, Y, Z):

        Extract thin slabs (1–N voxels thick)
        Deduplicate boards
        Remove overlapping duplicates, preferring thinner boards.
        Merge leftovers

    The results are that large planar structures (shelves, walls) become clean single boxes,
    while smaller details are handled separately.
    """

    voxel_size: float = 0.02
    """
    Voxel size in mesh units.
    """
    fill_thin_holes: bool = True
    """
    Whether to fill 1-voxel cracks/voids or not
    """
    max_thickness_vox: int = 2
    """
    Maximum board thickness in voxels.
    """
    min_span_vox: int = 3
    """
    Threshold for keeping boards.
    """
    min_fill_ratio: float = 0.75
    """
    Minimum ratio of occupied voxels to bounding box area.
    """
    overlap_threshold: float = 0.8
    """
    Overlap threshold at which two boards are merged into one.
    """

    def apply_to_mesh(self, mesh: Mesh) -> List[Box]:
        """
        Decompose a mesh into boxes.

        :param mesh: The mesh to decompose.
        :return: A list of Box objects.
        """
        trimesh_mesh = mesh.mesh
        voxelized = trimesh_mesh.voxelized(pitch=self.voxel_size).fill()

        occupancy = voxelized.matrix.astype(bool)
        origin = np.asarray(voxelized.translation, dtype=float)

        occupancy = clean_occupancy(
            occupancy,
            fill_thin_holes=self.fill_thin_holes,
        )

        board_candidates = detect_planar_boards(
            occupancy=occupancy,
            max_thickness_voxels=self.max_thickness_vox,
            min_span_voxels=self.min_span_vox,
            min_fill_ratio=self.min_fill_ratio,
        )
        board_index_boxes = deduplicate_index_boxes(
            board_candidates,
            overlap_ratio_threshold=self.overlap_threshold,
        )
        board_boxes = [
            box.to_frozen_box(self.voxel_size, origin) for box in board_index_boxes
        ]

        remainder_occupancy = subtract_index_boxes_from_occupancy(
            occupancy, board_index_boxes
        )
        leftover_boxes = greedy_merge_boxes(
            remainder_occupancy, self.voxel_size, origin
        )

        all_boxes = board_boxes + leftover_boxes

        return [
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    box.x,
                    box.y,
                    box.z,
                    reference_frame=mesh.origin.reference_frame,
                ),
                scale=box.scale,
            )
            for box in all_boxes
        ]

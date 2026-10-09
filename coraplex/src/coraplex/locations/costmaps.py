from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from matplotlib import colors
from skimage.measure import label
from typing_extensions import (
    List,
    Optional,
    Iterator,
    Set,
)

from coraplex.locations.base import Location
from krrood.entity_query_language.exceptions import NonPositiveLimitValue
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Quaternion,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    Point3,
    Pose,
    Pose2D,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.datastructures.dataclasses import Context

logger = logging.getLogger("coraplex")


@dataclass
class Costmap(Location):
    """
    The base class of all Costmaps.
    Costmaps describe regions in the world that are suitable for a certaint task.
    """

    resolution: float
    """
    The distance in metre in the real-world which is represented by a single entry in the locations. 
    """
    height: Optional[int] = field(kw_only=True, default=None)
    """
    Height of the locations.
    """
    width: Optional[int] = field(kw_only=True, default=None)
    """
    Width of the locations.
    """
    origin: Pose = field(kw_only=True, default_factory=Pose)
    """
    Origin pose of the locations.
    """
    map: np.ndarray = field(default_factory=lambda: np.zeros((10, 10)), kw_only=True)
    """
    Numpy array to save the locations distribution
    
    Costmaps represent the 2D distribution in a numpy array where axis 0 is the X-Axis of the coordinate system and axis 1 
    is the Y-Axis of the coordinate system. An increase in the index of the axis of the numpy array corresponds to an increase in the 
    value of the spatial axis. The factor by how the value of the index of the numpy corresponds to the spatial coordinate 
    system is given by the resolution. 

    Furthermore, there is a difference in the origin of the two representations while the numpy arrays start from the top left 
    corner, the origin given as Pose is placed in the middle of the array. The locations is build around the origin and 
    since the array start from 0, 0 in the corner this conversion is necessary. 

                y-axis      0, 10
        0,0 ------------------
            ------------------
            ------------------
    x-axis  ------------------
            ------------------
            ------------------
      10, 0 ------------------
    """

    world: World
    """
    The world from which this locations was created.
    """
    visualization_ids: List[int] = field(default_factory=list, init=False)

    def close_visualization(self) -> None:
        """
        Removes the visualization from the World.
        """
        for visualization_id in self.visualization_ids:
            self.world.remove_visual_object(visualization_id)
        self.visualization_ids = []

    def merge(self, other: Costmap) -> Costmap:
        """
        Merges the values of two locations and returns a new locations that has for
        every cell the merged values of both inputs. To merge two locations they
        need to fulfill 3 constrains:

        1. They need to have the same size
        2. They need to have the same x and y coordinates in the origin
        3. They need to have the same resolution

        If any of these constrains is not fulfilled a ValueError will be raised.

        :param other: The other locations with which this locations should be merged.
        :return: A new locations that contains the merged values, carrying this map's
            :attr:`number_of_samples` and :attr:`seed`.
        """
        if self.width != other.width or self.height != other.height:
            raise ValueError("You can only merge locations of the same size.")
        elif (
            not np.allclose(self.origin.x, other.origin.x)
            or not np.allclose(self.origin.y, other.origin.y)
            or not np.allclose(
                self.origin.rotation_matrix, other.origin.rotation_matrix
            )
        ):
            raise ValueError(
                "To merge locations, the x and y coordinate as well as the orientation must be equal."
            )
        elif self.resolution != other.resolution:
            raise ValueError("To merge two locations their resolution must be equal.")
        elif self.world != other.world:
            raise ValueError(
                "To merge two locations they must belong to the same world."
            )
        new_map = np.zeros((self.height, self.width))
        # A numpy array of the positions where both locations are greater than 0
        merge = np.logical_and(self.map > 0, other.map > 0)
        new_map[merge] = self.map[merge] * other.map[merge]
        maximum_value = np.max(new_map)
        if maximum_value != 0:
            new_map = (new_map / np.max(new_map)).reshape((self.height, self.width))
        else:
            new_map = new_map.reshape((self.height, self.width))
            logger.warning("Merged locations is empty.")
        return Costmap(
            resolution=self.resolution,
            height=self.height,
            width=self.width,
            origin=self.origin,
            map=new_map,
            world=self.world,
            number_of_samples=self.number_of_samples,
            seed=self.seed,
        )

    def __add__(self, other: Costmap) -> Costmap:
        """
        Overloading of the "+" operator for merging of Costmaps. Furthermore, checks if 'other' is actual a Costmap and
        raises a ValueError if this is not the case. Please check :func:`~Costmap.merge` for further information of merging.

        :param other: Another Costmap
        :return: A new Costmap that contains the merged values from this Costmap and the other Costmap
        """
        if isinstance(other, Costmap):
            return self.merge(other)
        else:
            raise ValueError(
                f"Can only combine two locations other type was {type(other)}"
            )

    def __and__(self, other):
        return self.merge(other)

    def candidates(self) -> Iterator[Pose2D]:
        return self.sample(self.number_of_samples, self.seed)

    def sample(
        self, number_of_samples: int, seed: Optional[int] = None
    ) -> Iterator[Pose2D]:
        """
        Sample pose candidates from this map.

        The sample count is capped at the number of entries this map holds, and every
        candidate faces this map's origin.

        :param number_of_samples: How many candidates to sample.
        :param seed: Fixes the sampling, or ``None`` to sample afresh.
        :return: The candidate poses, in the order they should be tried.
        :raises NonPositiveLimitValue: If asked for fewer than one candidate.
        """
        if number_of_samples < 1:
            raise NonPositiveLimitValue(number_of_samples)

        # An entry is only ever offered once, so the whole map is all there is to
        # sample.
        return self._sample(
            min(number_of_samples, self.map.size),
            np.random.default_rng(seed),
        )

    def _orientation_facing_origin(self, position: Point3) -> Quaternion:
        """
        The orientation a candidate sampled at the given position is offered with.

        A candidate faces this map's origin, so that whatever the map was built around
        is in front of the robot standing there.

        :param position: Where the candidate lies, in world frame.
        :return: The orientation for that candidate.
        """
        angle = (
            np.arctan2(
                position.y - self.origin.y,
                position.x - self.origin.x,
            )
            + np.pi
        )[0]
        return RotationMatrix.from_rpy(0, 0, angle).quaternion

    def _budget_per_segment(
        self, segments: List[np.ndarray], number_of_samples: int
    ) -> List[int]:
        """
        Split a sample budget over this map's segments in proportion to their summed
        ratings; what a segment cannot take goes to the best rated segments that can.

        :param segments: This map's segments, the best rated first.
        :param number_of_samples: How many candidates the whole map was asked for.
        :return: How many to sample from each segment, in the order they were given.
        """
        # Only entries rated above zero can be sampled.
        capacities = [int(np.count_nonzero(segment)) for segment in segments]
        ratings = np.array([segment.sum() for segment in segments], dtype=float)
        if not ratings.any():
            return [0] * len(segments)
        shares = number_of_samples * ratings / ratings.sum()
        budgets = np.minimum(np.floor(shares).astype(int), capacities)
        # Spend the rest on the best rated segments first (segments are sorted that way).
        unspent = number_of_samples - int(budgets.sum())
        for index, capacity in enumerate(capacities):
            if unspent <= 0:
                break
            taken = min(unspent, capacity - int(budgets[index]))
            budgets[index] += taken
            unspent -= taken
        return budgets.tolist()

    def _pick_entries(
        self,
        ratings: NDArray[np.float64],
        count: int,
        random_generator: np.random.Generator,
    ) -> NDArray[np.intp]:
        """
        Draw entries without repetition, each with a probability proportional to its
        rating; entries rated zero are never drawn.

        :param ratings: The flattened map, one rating per entry.
        :param count: How many entries to pick at most.
        :param random_generator: The source of randomness to sample from.
        :return: The indices to offer, in the order they should be offered.
        """
        offerable = min(count, int(np.count_nonzero(ratings)))
        if offerable <= 0:
            return np.empty(0, dtype=np.intp)
        return random_generator.choice(
            ratings.size, offerable, replace=False, p=ratings / ratings.sum()
        )

    def _sample(
        self,
        number_of_samples: int,
        random_generator: np.random.Generator,
    ) -> Iterator[Pose2D]:
        """
        Sample candidates, the given number of them spread over this map's segments.

        :param number_of_samples: How many candidates to sample, no more than this map
            holds.
        :param random_generator: The source of randomness to sample with.
        :Yield: A candidate pose.
        """
        segmented_maps = self.segment_map()
        budgets = self._budget_per_segment(segmented_maps, number_of_samples)
        for segmented_map, budget in zip(segmented_maps, budgets):
            indices = self._pick_entries(
                segmented_map.flatten(), budget, random_generator
            )
            indices = np.column_stack(np.unravel_index(indices, segmented_map.shape))

            height = segmented_map.shape[0]
            width = segmented_map.shape[1]
            center = np.array([height // 2, width // 2])
            for index in indices:
                # Compute world position independent of origin orientation:
                # map indices increase with world axes; origin is at the center.
                offset = (index - center) * self.resolution
                position = self.origin.position + Vector3(offset[0], offset[1], 0)

                yield Pose2D.from_pose(
                    Pose(
                        position,
                        self._orientation_facing_origin(position),
                        self.world.root,
                    )
                )

    def segment_map(self) -> List[np.ndarray]:
        """
        Finds partitions in the locations and isolates them, a partition is a number of entries in the locations which are
        neighbours. Returns a list of numpy arrays with one partition per array.

        :return: A list of numpy arrays with one partition per array
        """
        # In case the map is empty we just return the map
        if np.sum(self.map) == 0:
            return [self.map]

        discrete_map = np.copy(self.map)
        # Label only works on integer arrays
        discrete_map[discrete_map != 0] = 1

        labeled_map, number_of_labels = label(
            discrete_map, return_num=True, connectivity=2
        )
        result_maps = []
        # We don't want the maps for value 0
        for label_value in range(1, number_of_labels + 1):
            copy_map = deepcopy(self.map)
            copy_map[labeled_map != label_value] = 0
            result_maps.append(copy_map)
        # Maps with the highest values go first
        result_maps.sort(key=lambda segment: np.max(segment), reverse=True)
        return result_maps


@dataclass
class OccupancyCostmap(Costmap):
    """
    The occupancy Costmap represents a map of the environment where obstacles or
    positions which are inaccessible for a robot have a value of -1.
    """

    distance_to_obstacle: float
    """
    The distance by which obstacles in the occupancy map are inflated and are therefore not valid positions, in meter
    """

    robot_view: AbstractRobot
    """
    Robot semantic annotation which is used to create the map
    """

    _distance_to_obstacle_index: int = field(init=False, default=None)
    """
    Conversion of the distance_to_obstacle to index range for the internal representation.
    """

    def __post_init__(self):
        self._distance_to_obstacle_index = max(
            int(self.distance_to_obstacle / self.resolution), 1
        )
        self.map = self._create_from_world()

    def create_ray_mask_around_origin(self):
        """
        Determines the occupied space around the origin position using ray testing. A
        ray is cast from the robot's base height straight down to the ground and if it
        hits something the position is considered occupied.

        Neither the robot itself, nor whatever it carries, nor the floor it drives on
        makes a position occupied.

        :return: A 2d numpy array of the occupied space
        """
        origin_position = self.origin.position.to_list()
        # Generate 2d grid with indices
        indices = np.concatenate(
            np.dstack(
                np.mgrid[
                    int(-self.width / 2) : int(self.width / 2),
                    int(-self.width / 2) : int(self.width / 2),
                ]
            ),
            axis=0,
        ) * self.resolution + np.array(origin_position[:2])

        # base height of the robot plus a safty offset
        base_height = self.robot_view.mobile_base.bounding_box.height + 0.1
        # Every ray runs straight down from the robot's base height to the ground
        ray_origins = np.pad(
            indices, (0, 1), mode="constant", constant_values=base_height
        )[:-1]
        ray_targets = np.pad(indices, (0, 1), mode="constant", constant_values=0)[:-1]
        # Zips both arrays such that there are tuples for every coordinate that
        # only differ in the z-coordinate
        rays = np.dstack(np.dstack((ray_origins, ray_targets))).T

        free_space_mask = np.ones(len(rays))

        ray_tracer = RayTracer(self.world)
        _, hit_ray_indices, hit_bodies = ray_tracer.ray_test(rays[:, 0], rays[:, 1])

        unoccupied_entities = self._floor_bodies
        if self.robot_view:
            unoccupied_entities.update(
                self.world.get_kinematic_structure_entities_of_branch(
                    self.robot_view.root
                )
            )
        free_space_mask[hit_ray_indices] = [
            1 if body in unoccupied_entities else 0 for body in hit_bodies
        ]

        return np.flip(np.reshape(free_space_mask, (self.width, self.width)))

    @property
    def _floor_bodies(self) -> Set[Body]:
        """
        The floor slabs of the world, which a robot drives on rather than around.

        Only each floor's own body counts: a floor annotation also reaches the rooms
        standing on it, and their walls are obstacles like any other.

        :return: The bodies of the world's floors.
        """
        return {
            floor.root for floor in self.world.get_semantic_annotations_by_type(Floor)
        }

    def inflate_obstacles(self, map: np.ndarray):
        """
        Inflates found obstacles in the environment by the distance_to_obstacle factor.

        :param map: Map of obstacles to inflate.
        :return: The map with inflated obstacles.
        """
        window_shape = (
            self._distance_to_obstacle_index * 2,
            self._distance_to_obstacle_index * 2,
        )
        view_shape = tuple(np.subtract(map.shape, window_shape) + 1) + window_shape
        strides = map.strides + map.strides

        windows = np.lib.stride_tricks.as_strided(map, view_shape, strides)
        windows = windows.reshape(windows.shape[:-2] + (-1,))

        window_sums = np.sum(windows, axis=2)
        map = (window_sums == (self._distance_to_obstacle_index * 2) ** 2).astype(
            "int16"
        )
        return map

    def _create_from_world(self) -> np.ndarray:
        """
        Creates an Occupancy Costmap for the specified World.
        This map marks every position as valid that has no object above it. After
        creating the locations the distance to obstacle parameter is applied.
        """

        ray_mask = self.create_ray_mask_around_origin()

        map = np.pad(
            ray_mask,
            (
                int(self._distance_to_obstacle_index / 2),
                int(self._distance_to_obstacle_index / 2),
            ),
        )

        map = self.inflate_obstacles(map)
        # The map loses some size due to the strides and because I dont want to
        # deal with indices outside of the index range
        offset = self.width - map.shape[0]
        odd = 0 if offset % 2 == 0 else 1
        map = np.pad(map, (offset // 2, offset // 2 + odd))

        return np.flip(map)

    @classmethod
    def default_map(
        cls,
        context: Context,
        target: Pose,
        *,
        resolution: float = 0.02,
        cells: int = 200,
    ) -> OccupancyCostmap:
        """
        Creates an occupancy costmap around a target, keeping the robot base's radius
        clear of obstacles.

        :param context: The context to create the occupancy cost map.
        :param target: The target pose for the occupancy cost map.
        :param resolution: Edge length of a cell, in meters.
        :param cells: Number of cells along each side of the map.
        :returns: The occupancy cost map.
        """
        ground_pose = deepcopy(target)
        ground_pose.z = 0

        return OccupancyCostmap(
            resolution=resolution,
            width=cells,
            height=cells,
            world=context.world,
            distance_to_obstacle=context.robot.mobile_base.base_radius,
            robot_view=context.robot,
            origin=ground_pose,
        )


@dataclass
class VisibilityCostmap(Costmap):
    """
    A locations that represents the visibility of a specific point for every position around
    this point. For a detailed explanation on how the creation of the locations works
    please look here: `PhD Thesis (page 173) <https://mediatum.ub.tum.de/doc/1239461/1239461.pdf>`_
    """

    minimum_height: float

    maximum_height: float

    target_object: Optional[Union[Body, Pose]] = None

    def __post_init__(self):
        self.origin: Pose = (
            Pose(reference_frame=self.world.root) if not self.origin else self.origin
        )
        self._generate_map()

    def _create_images(self) -> List[np.ndarray]:
        """
        Creates four depth images in every direction around the point
        for which the locations should be created. The depth images are converted
        to metre, meaning that every entry in the depth images represents the
        distance to the next object in metre.

        :return: A list of four depth images, the images are represented as 2D arrays.
        """
        images = []

        ray_tracer = RayTracer(self.world)

        origin_copy = deepcopy(self.origin).homogeneous_matrix

        for _ in range(4):
            origin_copy = origin_copy @ HomogeneousTransformationMatrix.from_xyz_rpy(
                yaw=np.pi / 2
            )
            images.append(
                ray_tracer.create_depth_map(
                    origin_copy,
                    resolution=CameraResolution(
                        width=self.width,
                        height=self.width,
                    ),
                    min_distance=0.1,
                )
            )

        return images

    def _generate_map(self):
        """
        This method generates the resulting density map by using the algorithm explained
        in Lorenz Mösenlechners `PhD Thesis (page 178) <https://mediatum.ub.tum.de/doc/1239461/1239461.pdf>`_
        The resulting map is then saved to :py:attr:`self.map`
        """
        depth_images = self._create_images()
        # A 2D array where every cell contains the arctan2 value with respect to
        # the middle of the array. Additionally, the interval is shifted such that
        # it is between 0 and 2pi
        angles = (
            np.arctan2(
                np.mgrid[
                    -int(self.width / 2) : int(self.width / 2),
                    -int(self.width / 2) : int(self.width / 2),
                ][0],
                np.mgrid[
                    -int(self.width / 2) : int(self.width / 2),
                    -int(self.width / 2) : int(self.width / 2),
                ][1],
            )
            + np.pi
        )
        image_indices = np.zeros(angles.shape)

        # Just for completion, since the image_indices array has zeros in every position this
        # operation is not necessary.
        # image_indices[np.logical_and(angles <= np.pi * 0.25, angles >= np.pi * 1.75)] = 0

        # Creates a 2D array which contains the index of the depth image for every
        # coordinate
        image_indices[
            np.logical_and(angles >= np.pi * 1.25, angles <= np.pi * 1.75)
        ] = 3
        image_indices[np.logical_and(angles >= np.pi * 0.75, angles < np.pi * 1.25)] = 2
        image_indices[np.logical_and(angles >= np.pi * 0.25, angles < np.pi * 0.75)] = 1

        indices = np.dstack(np.mgrid[0 : self.width, 0 : self.width])
        depth_indices = np.zeros(indices.shape)
        # x-value of index: image_indices == n, :1
        # y-value of index: image_indices == n, 1:2

        # (y, size-x-1) for index between 1.25 pi and 1.75 pi
        depth_indices[image_indices == 3, :1] = indices[image_indices == 3, 1:2]
        depth_indices[image_indices == 3, 1:2] = (
            self.width - indices[image_indices == 3, :1] - 1
        )

        # (size-x-1, y) for index between 0.75 pi and 1.25 pi
        depth_indices[image_indices == 2, :1] = (
            self.width - indices[image_indices == 2, :1] - 1
        )
        depth_indices[image_indices == 2, 1:2] = indices[image_indices == 2, 1:2]

        # (size-y-1, x) for index between 0.25 pi and 0.75 pi
        depth_indices[image_indices == 1, :1] = (
            self.width - indices[image_indices == 1, 1:2] - 1
        )
        depth_indices[image_indices == 1, 1:2] = indices[image_indices == 1, :1]

        # (x, y) for index between 0.25 pi and 1.75 pi
        depth_indices[image_indices == 0, :1] = indices[image_indices == 0, :1]
        depth_indices[image_indices == 0, 1:2] = indices[image_indices == 0, 1:2]

        # Convert back to origin in the middle of the locations
        depth_indices[:, :, :1] -= self.width / 2
        depth_indices[:, :, 1:2] = np.absolute(
            self.width / 2 - depth_indices[:, :, 1:2]
        )

        # Sets the y index for the coordinates of the middle of the locations to 1,
        # the computed value is 0 which would cause an error in the next step where
        # the calculation divides the x coordinates by the y coordinates
        depth_indices[int(self.width / 2), int(self.width / 2), 1] = 1

        # Calculate columns for the respective position in the locations
        columns = (
            np.around(
                (
                    (depth_indices[:, :, :1] / depth_indices[:, :, 1:2])
                    * (self.width / 2)
                )
                + self.width / 2
            )
            .reshape((self.width, self.width))
            .astype("int16")
        )

        # An array with size * size that contains the euclidean distance to the
        # origin (in the middle of the locations) from every cell
        distances = np.maximum(
            np.linalg.norm(
                np.dstack(
                    np.mgrid[
                        -int(self.width / 2) : int(self.width / 2),
                        -int(self.width / 2) : int(self.width / 2),
                    ]
                ),
                axis=2,
            ),
            0.001,
        )

        # Row ranges
        # Calculation of the ranges of coordinates in the row which have to be
        # taken into account. The range is from row_minimum to row_maximum.
        # These are two arrays with shape: size*size, the row_minimum constrains the beginning
        # of the range for every coordinate and row_maximum contains the end for each
        # coordinate
        row_minimum = (
            np.arctan((self.minimum_height - self.origin.z) / distances) * self.width
        ) + self.width / 2
        row_maximum = (
            np.arctan((self.maximum_height - self.origin.z) / distances) * self.width
        ) + self.width / 2

        row_minimum = np.minimum(np.around(row_minimum), self.width - 1).astype("int16")
        row_maximum = np.minimum(np.around(row_maximum), self.width - 1).astype("int16")

        row_ranges = np.dstack((row_minimum, row_maximum + 1)).reshape(
            (self.width**2, 2)
        )
        row_indices = np.arange(self.width)
        # Calculates a mask from the row_minimum and row_maximum values. This mask is for every
        # coordinate respectively and determines which values from the computed column
        # of the depth image should be taken into account for the locations.
        # A Mask of a single coordinate has the length of the column of the depth image
        # and together with the computed column at this coordinate determines which
        # values of the depth image make up the value of the visibility locations at this
        # point.
        mask = (
            (row_ranges[:, 0, None] <= row_indices)
            & (row_ranges[:, 1, None] > row_indices)
        ).reshape((self.width, self.width, self.width))

        values = np.zeros((self.width, self.width))
        map = np.zeros((self.width, self.width))
        # This is done to iterate over the depth images one at a time
        for image_index in range(4):
            row_masks = mask[image_indices == image_index].T
            # This statement does several things, first it takes the values from
            # the depth image for this quarter of the locations. The values taken are
            # the complete columns of the depth image (which where computed beforehand)
            # and checks if the values in them are greater than the distance to the
            # respective coordinates. This does not take the row ranges into account.
            values = (
                depth_images[image_index][
                    :, columns[image_indices == image_index].flatten()
                ]
                < np.tile(
                    distances[image_indices == image_index][:, None], (1, self.width)
                ).T
                * self.resolution
            )
            # This applies the created mask of the row ranges to the values of
            # the columns which are compared in the previous statement
            masked = np.ma.masked_array(values, mask=~row_masks)
            # The calculated values are added to the locations
            map[image_indices == image_index] = np.sum(masked, axis=0)
        map /= np.max(map)
        # Weird flipping shit so that the map fits the orientation of the visualization.
        # the locations in itself is consistent and just needs to be flipped to fit the world coordinate system
        map = np.flip(map, axis=0)
        map = np.flip(map)

        # Invert the map
        inverted_map = np.zeros(map.shape)
        inverted_map[map == 0] = 1
        inverted_map[map != 0] = 0

        self.map = inverted_map


@dataclass
class GaussianCostmap(Costmap):
    """
    Gaussian Costmaps are 2D gaussian distributions around the origin with the given mean and sigma
    """

    mean: int
    """
    The mean input for the gaussian distribution, this also specifies 
    the length of the side of the resulting locations. The locations is Created
    as a square.
    """

    sigma: float
    """
    The sigma input for the gaussian distribution.
    """

    world: World
    """
    The world to use.
    """

    def __post_init__(self):
        self.gaussian_window: np.ndarray = self._create_gaussian_window(
            self.mean, self.sigma
        )
        self.map: np.ndarray = np.outer(self.gaussian_window, self.gaussian_window)
        cut_distance = int(0.05 * self.mean)
        center = int(self.mean / 2)
        # Cuts out the middle 5% of the gaussian to avoid the robot being too close to the target since this is usually
        # bad for reaching the target with a end_effector. 15% is a magic number that might need some tuning in the future
        self.map[
            center - cut_distance : center + cut_distance,
            center - cut_distance : center + cut_distance,
        ] = 0
        self.size: float = self.mean
        self.width = int(self.size)
        self.height = int(self.size)

    def _create_gaussian_window(
        self, mean: int, standard_deviation: float
    ) -> np.ndarray:
        """
        Creates a window of values with a gaussian distribution of the given size
        and standard deviation.

        Code from `Scipy <https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/windows.py#L976>`_
        """
        offsets = np.arange(0, mean) - (mean - 1.0) / 2.0
        twice_variance = 2 * standard_deviation * standard_deviation
        return np.exp(-(offsets**2) / twice_variance)


@dataclass
class RingCostmap(Costmap):
    """
    Creates a ring locations, similar to the gaussian locations but this looks more like a donut. Can be used to create poses
    for reaching a point for the robot.
    """

    standard_deviation: int
    """
    Standard deviation of the gaussian distribution that makes up the ring.
    """

    distance: float
    """
    Distance between the center of the locations and the center of the ring. A distance of 0 results in a gaussian locations
    """

    def __post_init__(self):
        self.map = self.ring()

    @classmethod
    def from_arm_reach_distance(
        cls,
        context: Context,
        arm: Arm,
        origin: Pose,
        reach_fraction: float,
        *,
        resolution: float = 0.02,
        cells: int = 200,
        standard_deviation: int = 15,
    ) -> RingCostmap:
        """
        Creates a ring costmap around a target the robot is to reach with one arm.

        :param context: The context holding the robot and world.
        :param arm: The arm that will do the reaching.
        :param origin: The target the ring is drawn around.
        :param reach_fraction: The fraction of the arm's length the ring stands off
            the target by.
        :param resolution: Edge length of a cell, in meters.
        :param cells: Number of cells along each side of the map.
        :param standard_deviation: How far, in cells, the ring spreads around its
            stand-off distance.
        :returns: The ring costmap.
        """
        return cls(
            resolution=resolution,
            width=cells,
            height=cells,
            standard_deviation=standard_deviation,
            distance=arm.approximate_length() * reach_fraction,
            world=context.world,
            origin=origin,
        )

    def ring(self) -> np.ndarray:
        radius_in_pixels = self.distance / self.resolution

        y, x = np.ogrid[: self.width, : self.height]
        center_x = (self.height - int(self.height % 2 == 0)) / 2.0
        center_y = (self.width - int(self.width % 2 == 0)) / 2.0

        distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        ring_costmap = np.exp(
            -((distance_from_center - radius_in_pixels) ** 2)
            / (2 * self.standard_deviation**2)
        )
        return ring_costmap


grid_color_map = colors.ListedColormap(["white", "black", "green", "red", "blue"])


# Mainly used for debugging
# Data is 2d array
def plot_grid(data: np.ndarray) -> None:
    """
    An auxiliary method only used for debugging, it will plot a 2D numpy array using MatplotLib.
    """
    rows = data.shape[0]
    columns = data.shape[1]
    figure, axes = plt.subplots()
    axes.imshow(data, cmap=grid_color_map)
    # draw gridlines
    # axes.grid(which='major', axis='both', linestyle='-', rgba_color='k', linewidth=1)
    axes.set_xticks(np.arange(0.5, rows, 1))
    axes.set_yticks(np.arange(0.5, columns, 1))
    plt.tick_params(axis="both", labelsize=0, length=0)
    # figure.set_size_inches((8.5, 11), forward=False)
    # plt.savefig(saveImageName + ".png", dpi=500)
    plt.show()

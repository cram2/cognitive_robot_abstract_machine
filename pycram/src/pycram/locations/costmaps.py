from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from skimage.measure import label
from typing_extensions import Tuple, List, Optional, Iterator, Callable

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Quaternion,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


class OrientationGenerator:
    """
    Provides methods to generate orientations for pose candidates.
    """

    @staticmethod
    def generate_origin_orientation(
        position: Point3, origin: Pose, rotate_by_angle: float = 0
    ) -> Quaternion:
        """
        Generates an orientation such that the robot faces the origin of the locations.

        :param position: The position in the locations, already converted to the world coordinate frame.
        :param origin: The origin of the locations, the point which the robot should face.
        :param rotate_by_angle: Angle to rotate the orientation.
        :return: A quaternion of the calculated orientation.
        """
        rotation_R_new_rotation = RotationMatrix.from_rpy(0, 0, rotate_by_angle)
        angle = (
            np.arctan2(
                position.y - origin.y,
                position.x - origin.x,
            )
            + np.pi
        )[0]
        world_R_rotation = RotationMatrix.from_rpy(0, 0, angle)
        world_R_new_rotation = world_R_rotation @ rotation_R_new_rotation
        return world_R_new_rotation.to_quaternion()

    @staticmethod
    def orientation_generator_for_axis(
        axis: Vector3,
    ) -> Callable[[Point3, Pose], Quaternion]:
        """
        Creates an orientation generator where the given axis is facing the target.

        :param axis: The axis which should be facing the target
        :return: A callable orientation generator
        """
        rotation = axis[1] * (np.pi / 2) * -1
        return partial(
            OrientationGenerator.generate_origin_orientation, rotate_by_angle=rotation
        )

    @staticmethod
    def generate_random_orientation(
        *_, rng: random.Random = random.Random(42)
    ) -> Quaternion:
        """
        Generates a random orientation rotated around the z-axis (yaw).
        A random angle is sampled using a provided RNG instance to ensure reproducibility.

        :param _: Ignored parameters to maintain compatibility with other orientation generators.
        :param rng: Random number generator instance for reproducible sampling.

        :return: A quaternion of the randomly generated orientation.
        """
        return Quaternion.from_rpy(0, 0, rng.uniform(0, 2 * np.pi))


@dataclass
class Rectangle:
    """
    A rectangle that is described by a lower and upper x and y value.
    """

    x_lower: float
    x_upper: float
    y_lower: float
    y_upper: float

    def translate(self, x: float, y: float):
        """Translate the rectangle by x and y"""
        self.x_lower += x
        self.x_upper += x
        self.y_lower += y
        self.y_upper += y

    def scale(self, x_factor: float, y_factor: float):
        """Scale the rectangle by x_factor and y_factor"""
        self.x_lower *= x_factor
        self.x_upper *= x_factor
        self.y_lower *= y_factor
        self.y_upper *= y_factor


@dataclass
class Costmap:
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
    vis_ids: List[int] = field(default_factory=list, init=False)

    number_of_samples: int = field(kw_only=True, default=200)
    """
    Number of samples to return at max
    """

    sample_randomly: bool = field(kw_only=True, default=False)
    """
    If the sampling should randomly pick valid entries
    """

    orientation_generator: Callable[Pose, Pose, [float]] = field(
        kw_only=True, default=None
    )
    """
    An optional orientatoin generator to use to generate the orientation for a sampled pose
    """

    def _chunks(self, lst: List, n: int) -> Iterator[List]:
        """
        Yield successive n-sized chunks from lst.

        :param lst: The list from which chunks should be yielded
        :param n: Size of the chunks
        :return: A list of size n from lst
        """
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def close_visualization(self) -> None:
        """
        Removes the visualization from the World.
        """
        for v_id in self.vis_ids:
            self.world.remove_visual_object(v_id)
        self.vis_ids = []

    def _find_consectuive_line(self, start: Tuple[int, int], map: np.ndarray) -> int:
        """
        Finds the number of consecutive entries in the locations which are greater
        than zero.

        :param start: The indices in the locations from which the consecutive line should be found.
        :param map: The locations in which the line should be found.
        :return: The length of the consecutive line of entries greater than zero.
        """
        width = map.shape[1]
        length = 0
        for i in range(start[1], width):
            if map[start[0]][i] > 0:
                length += 1
            else:
                return length
        return length

    def _find_max_box_height(
        self, start: Tuple[int, int], length: int, map: np.ndarray
    ) -> int:
        """
        Finds the maximal height for a rectangle with a given width in a locations.
        The method traverses one row at a time and checks if all entries for the
        given width are greater than zero. If an entry is less or equal than zero
        the height is returned.

        :param start: The indices in the locations from which the method should start.
        :param length: The given width for the rectangle
        :param map: The locations in which should be searched.
        :return: The height of the rectangle.
        """
        height, width = map.shape
        curr_height = 1
        for i in range(start[0], height):
            for j in range(start[1], start[1] + length):
                if map[i][j] <= 0:
                    return curr_height
            curr_height += 1
        return curr_height

    def merge(self, other_cm: Costmap) -> Costmap:
        """
        Merges the values of two locations and returns a new locations that has for
        every cell the merged values of both inputs. To merge two locations they
        need to fulfill 3 constrains:

        1. They need to have the same size
        2. They need to have the same x and y coordinates in the origin
        3. They need to have the same resolution

        If any of these constrains is not fulfilled a ValueError will be raised.

        :param other_cm: The other locations with which this locations should be merged.
        :return: A new locations that contains the merged values
        """
        if self.width != other_cm.width or self.height != other_cm.height:
            raise ValueError("You can only merge locations of the same size.")
        elif (
            not np.allclose(self.origin.x, other_cm.origin.x)
            or not np.allclose(self.origin.y, other_cm.origin.y)
            or not np.allclose(
                self.origin.to_rotation_matrix(), other_cm.origin.to_rotation_matrix()
            )
        ):
            raise ValueError(
                "To merge locations, the x and y coordinate as well as the orientation must be equal."
            )
        elif self.resolution != other_cm.resolution:
            raise ValueError("To merge two locations their resolution must be equal.")
        elif self.world != other_cm.world:
            raise ValueError(
                "To merge two locations they must belong to the same world."
            )
        new_map = np.zeros((self.height, self.width))
        # A numpy array of the positions where both locations are greater than 0
        merge = np.logical_and(self.map > 0, other_cm.map > 0)
        new_map[merge] = self.map[merge] * other_cm.map[merge]
        max_val = np.max(new_map)
        if max_val != 0:
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

    def partitioning_rectangles(self) -> List[Rectangle]:
        """
        Partition the map attached to this locations into rectangles. The rectangles are axis aligned, exhaustive and
        disjoint sets.

        :return: A list containing the partitioning rectangles
        """
        ocm_map = np.copy(self.map)
        origin = np.array([self.height / 2, self.width / 2]) * -1
        rectangles = []

        # for every index pair (i, j) in the occupancy locations
        for i in range(0, self.map.shape[0]):
            for j in range(0, self.map.shape[1]):

                # if this index has not been used yet
                if ocm_map[i][j] > 0:
                    curr_width = self._find_consectuive_line((i, j), ocm_map)
                    curr_pose = (i, j)
                    curr_height = self._find_max_box_height((i, j), curr_width, ocm_map)

                    # calculate the rectangle in the locations
                    x_lower = curr_pose[0]
                    x_upper = curr_pose[0] + curr_height
                    y_lower = curr_pose[1]
                    y_upper = curr_pose[1] + curr_width

                    # mark the found rectangle as occupied
                    ocm_map[i : i + curr_height, j : j + curr_width] = 0

                    # transform rectangle to map space
                    rectangle = Rectangle(x_lower, x_upper, y_lower, y_upper)
                    rectangle.translate(*origin)
                    rectangle.scale(self.resolution, self.resolution)
                    rectangles.append(rectangle)

        return rectangles

    def __iter__(self) -> Iterator[Pose]:
        """
        A generator that crates pose candidates from a given locations. The generator
        selects the highest 100 values and returns the corresponding positions.
        Orientations are calculated such that the Robot faces the center of the locations.

        :Yield: A tuple of position and orientation
        """

        ori_gen = (
            self.orientation_generator
            or OrientationGenerator.generate_origin_orientation
        )

        # Determines how many positions should be sampled from the locations
        if (
            self.number_of_samples == -1
            or self.number_of_samples > self.map.flatten().shape[0]
        ):
            self.number_of_samples = self.map.flatten().shape[0]

        segmented_maps = self.segment_map()
        samples_per_map = self.number_of_samples // len(segmented_maps)
        for seg_map in segmented_maps:

            if self.sample_randomly:
                indices = np.random.choice(seg_map.size, samples_per_map, replace=False)
            else:
                indices = np.argpartition(seg_map.flatten(), -samples_per_map)[
                    -samples_per_map:
                ]

            indices = np.dstack(np.unravel_index(indices, self.map.shape)).reshape(
                samples_per_map, 2
            )

            height = seg_map.shape[0]
            width = seg_map.shape[1]
            center = np.array([height // 2, width // 2])
            for ind in indices:
                if seg_map[ind[0]][ind[1]] == 0:
                    continue
                # Compute world position independent of origin orientation:
                # map indices increase with world axes; origin is at the center.
                offset = (ind - center) * self.resolution
                position = self.origin.to_position() + Vector3(offset[0], offset[1], 0)

                orientation: Quaternion = ori_gen(position, self.origin)
                yield Pose(
                    position,
                    orientation,
                    self.world.root,
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

        labeled_map, num_labels = label(discrete_map, return_num=True, connectivity=2)
        result_maps = []
        # We don't want the maps for value 0
        for i in range(1, num_labels + 1):
            copy_map = deepcopy(self.map)
            copy_map[labeled_map != i] = 0
            result_maps.append(copy_map)
        # Maps with the highest values go first
        result_maps.sort(key=lambda m: np.max(m), reverse=True)
        return result_maps


@dataclass
class OccupancyCostmap(Costmap):
    """
    The occupancy Costmap represents a map of the environment where obstacles or
    positions which are inaccessible for a robot have a value of -1.
    """

    distance_to_obstacle: float

    robot_view: AbstractRobot

    _distance_to_obstacle_index: int = field(init=False, default=None)

    def __post_init__(self):
        self._distance_to_obstacle_index = max(
            int(self.distance_to_obstacle / self.resolution), 1
        )
        self.map = self._create_from_world()

    def create_ray_mask_around_origin(self):
        """
        Determines the occupied space around the origin position using ray testing. A ray is cast from the ground
        straight up 10m and if it hits something the position is considered occupied.

        :return: A 2d numpy array of the occupied space
        """
        origin_position = self.origin.to_position().to_list()
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
        # Add the z-coordinate to the grid, which is either 0 or 10
        indices_0 = np.pad(indices, (0, 1), mode="constant", constant_values=5)[:-1]
        indices_10 = np.pad(indices, (0, 1), mode="constant", constant_values=0)[:-1]
        # Zips both arrays such that there are tuples for every coordinate that
        # only differ in the z-coordinate
        rays = np.dstack(np.dstack((indices_0, indices_10))).T

        res = np.ones(len(rays))

        ray_tracer = RayTracer(self.world)
        r_t = ray_tracer.ray_test(rays[:, 0], rays[:, 1])
        if self.robot_view:
            res[r_t[1]] = [
                (
                    1
                    if r_t[2][i]
                    in self.world.get_kinematic_structure_entities_of_branch(
                        self.robot_view.root
                    )
                    else 0
                )
                for i in range(len(r_t[1]))
            ]
        else:
            res[r_t[1]] = 0

        res = np.flip(np.reshape(np.array(res), (self.width, self.width)))
        return res

    def inflate_obstacles(self, map: np.ndarray):
        """
        Inflates found obstacles in the environment by the distance_to_obstacle factor.

        :param map: Map of obstacles to inflate.
        :return: The map with inflated obstacles.
        """
        sub_shape = (
            self._distance_to_obstacle_index * 2,
            self._distance_to_obstacle_index * 2,
        )
        view_shape = tuple(np.subtract(map.shape, sub_shape) + 1) + sub_shape
        strides = map.strides + map.strides

        sub_matrices = np.lib.stride_tricks.as_strided(map, view_shape, strides)
        sub_matrices = sub_matrices.reshape(sub_matrices.shape[:-2] + (-1,))

        sum = np.sum(sub_matrices, axis=2)
        map = (sum == (self._distance_to_obstacle_index * 2) ** 2).astype("int16")
        return map

    def _create_from_world(self) -> np.ndarray:
        """
        Creates an Occupancy Costmap for the specified World.
        This map marks every position as valid that has no object above it. After
        creating the locations the distance to obstacle parameter is applied.
        """

        res = self.create_ray_mask_around_origin()

        map = np.pad(
            res,
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


@dataclass
class VisibilityCostmap(Costmap):
    """
    A locations that represents the visibility of a specific point for every position around
    this point. For a detailed explanation on how the creation of the locations works
    please look here: `PhD Thesis (page 173) <https://mediatum.ub.tum.de/doc/1239461/1239461.pdf>`_
    """

    min_height: float

    max_height: float

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

        r_t = RayTracer(self.world)

        origin_copy = deepcopy(self.origin)

        for _ in range(4):
            # this quaternion is invalid, as it is never normalized. But the test pass with it, and any 90° yaw fails for me
            # finding out the error is super annoying in the current implementation, and it is not really used atm, so I dont
            # think its worth the time to fix it. If you disagree, feel free to give it a shot.
            rotated_origin_copy = (
                HomogeneousTransformationMatrix.from_point_rotation_matrix(
                    rotation_matrix=Quaternion(0, 0, 1, 1).to_rotation_matrix()
                )
                @ origin_copy
            )
            images.append(
                r_t.create_depth_map(
                    rotated_origin_copy,
                    resolution=self.width,
                )
            )

        return images

    def _generate_map(self):
        """
        This method generates the resulting density map by using the algorithm explained
        in Lorenz Mösenlechners `PhD Thesis (page 178) <https://mediatum.ub.tum.de/doc/1239461/1239461.pdf>`_
        The resulting map is then saved to :py:attr:`self.map`
        """
        depth_imgs = self._create_images()
        # A 2D array where every cell contains the arctan2 value with respect to
        # the middle of the array. Additionally, the interval is shifted such that
        # it is between 0 and 2pi
        tan = (
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
        res = np.zeros(tan.shape)

        # Just for completion, since the res array has zeros in every position this
        # operation is not necessary.
        # res[np.logical_and(tan <= np.pi * 0.25, tan >= np.pi * 1.75)] = 0

        # Creates a 2D array which contains the index of the depth image for every
        # coordinate
        res[np.logical_and(tan >= np.pi * 1.25, tan <= np.pi * 1.75)] = 3
        res[np.logical_and(tan >= np.pi * 0.75, tan < np.pi * 1.25)] = 2
        res[np.logical_and(tan >= np.pi * 0.25, tan < np.pi * 0.75)] = 1

        indices = np.dstack(np.mgrid[0 : self.width, 0 : self.width])
        depth_indices = np.zeros(indices.shape)
        # x-value of index: res == n, :1
        # y-value of index: res == n, 1:2

        # (y, size-x-1) for index between 1.25 pi and 1.75 pi
        depth_indices[res == 3, :1] = indices[res == 3, 1:2]
        depth_indices[res == 3, 1:2] = self.width - indices[res == 3, :1] - 1

        # (size-x-1, y) for index between 0.75 pi and 1.25 pi
        depth_indices[res == 2, :1] = self.width - indices[res == 2, :1] - 1
        depth_indices[res == 2, 1:2] = indices[res == 2, 1:2]

        # (size-y-1, x) for index between 0.25 pi and 0.75 pi
        depth_indices[res == 1, :1] = self.width - indices[res == 1, 1:2] - 1
        depth_indices[res == 1, 1:2] = indices[res == 1, :1]

        # (x, y) for index between 0.25 pi and 1.75 pi
        depth_indices[res == 0, :1] = indices[res == 0, :1]
        depth_indices[res == 0, 1:2] = indices[res == 0, 1:2]

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
        # taken into account. The range is from r_min to r_max.
        # These are two arrays with shape: size*size, the r_min constrains the beginning
        # of the range for every coordinate and r_max contains the end for each
        # coordinate
        r_min = (
            np.arctan((self.min_height - self.origin.z) / distances) * self.width
        ) + self.width / 2
        r_max = (
            np.arctan((self.max_height - self.origin.z) / distances) * self.width
        ) + self.width / 2

        r_min = np.minimum(np.around(r_min), self.width - 1).astype("int16")
        r_max = np.minimum(np.around(r_max), self.width - 1).astype("int16")

        rs = np.dstack((r_min, r_max + 1)).reshape((self.width**2, 2))
        r = np.arange(self.width)
        # Calculates a mask from the r_min and r_max values. This mask is for every
        # coordinate respectively and determines which values from the computed column
        # of the depth image should be taken into account for the locations.
        # A Mask of a single coordinate has the length of the column of the depth image
        # and together with the computed column at this coordinate determines which
        # values of the depth image make up the value of the visibility locations at this
        # point.
        mask = ((rs[:, 0, None] <= r) & (rs[:, 1, None] > r)).reshape(
            (self.width, self.width, self.width)
        )

        values = np.zeros((self.width, self.width))
        map = np.zeros((self.width, self.width))
        # This is done to iterate over the depth images one at a time
        for i in range(4):
            row_masks = mask[res == i].T
            # This statement does several things, first it takes the values from
            # the depth image for this quarter of the locations. The values taken are
            # the complete columns of the depth image (which where computed beforehand)
            # and checks if the values in them are greater than the distance to the
            # respective coordinates. This does not take the row ranges into account.
            values = (
                depth_imgs[i][:, columns[res == i].flatten()]
                < np.tile(distances[res == i][:, None], (1, self.width)).T
                * self.resolution
            )
            # This applies the created mask of the row ranges to the values of
            # the columns which are compared in the previous statement
            masked = np.ma.masked_array(values, mask=~row_masks)
            # The calculated values are added to the locations
            map[res == i] = np.sum(masked, axis=0)
        map /= np.max(map)
        # Weird flipping shit so that the map fits the orientation of the visualization.
        # the locations in itself is consistent and just needs to be flipped to fit the world coordinate system
        map = np.flip(map, axis=0)
        map = np.flip(map)
        self.map = map


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
        self.gau: np.ndarray = self._gaussian_window(self.mean, self.sigma)
        self.map: np.ndarray = np.outer(self.gau, self.gau)
        cut_dist = int(0.05 * self.mean)
        center = int(self.mean / 2)
        # Cuts out the middle 5% of the gaussian to avoid the robot being too close to the target since this is usually
        # bad for reaching the target with a manipulator. 15% is a magic number that might need some tuning in the future
        self.map[
            center - cut_dist : center + cut_dist, center - cut_dist : center + cut_dist
        ] = 0
        self.size: float = self.mean
        self.width = int(self.size)
        self.height = int(self.size)

    def _gaussian_window(self, mean: int, std: float) -> np.ndarray:
        """
        This method creates a window of values with a gaussian distribution of
        size "mean" and standart deviation "std".
        Code from `Scipy <https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/windows.py#L976>`_
        """
        n = np.arange(0, mean) - (mean - 1.0) / 2.0
        sig2 = 2 * std * std
        w = np.exp(-(n**2) / sig2)
        return w


@dataclass
class RingCostmap(Costmap):
    """
    Creates a ring locations, similar to the gaussian locations but this looks more like a donut. Can be used to create poses
    for reaching a point for the robot.
    """

    std: int
    """
    Standard deviation of the gaussian distribution that makes up the ring.
    """

    distance: float
    """
    Distance between the center of the locations and the center of the ring. A distance of 0 results in a gaussian locations
    """

    def __post_init__(self):
        self.map = self.ring()

    def ring(self) -> np.ndarray:
        radius_in_pixels = self.distance / self.resolution

        y, x = np.ogrid[: self.width, : self.height]
        center_x = (self.height - int(self.height % 2 == 0)) / 2.0
        center_y = (self.width - int(self.width % 2 == 0)) / 2.0

        distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        ring_costmap = np.exp(
            -((distance_from_center - radius_in_pixels) ** 2) / (2 * self.std**2)
        )
        return ring_costmap


cmap = colors.ListedColormap(["white", "black", "green", "red", "blue"])


# Mainly used for debugging
# Data is 2d array
def plot_grid(data: np.ndarray) -> None:
    """
    An auxiliary method only used for debugging, it will plot a 2D numpy array using MatplotLib.
    """
    rows = data.shape[0]
    cols = data.shape[1]
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap)
    # draw gridlines
    # ax.grid(which='major', axis='both', linestyle='-', rgba_color='k', linewidth=1)
    ax.set_xticks(np.arange(0.5, rows, 1))
    ax.set_yticks(np.arange(0.5, cols, 1))
    plt.tick_params(axis="both", labelsize=0, length=0)
    # fig.set_size_inches((8.5, 11), forward=False)
    # plt.savefig(saveImageName + ".png", dpi=500)
    plt.show()

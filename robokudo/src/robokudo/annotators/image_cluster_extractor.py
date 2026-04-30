"""Image-based object cluster extraction.

This module provides functionality for extracting object clusters from color images using HSV color segmentation.
The main class :class:`ImageClusterExtractor` implements color-based segmentation and contour detection to identify
object clusters in RGB images.

Key features:

* HSV color space thresholding
* Contour detection and filtering
* 3D point cloud generation from depth data
* ROI and mask generation
* Query-based color parameter adjustment
* Visualization of detected clusters
"""

import copy
from timeit import default_timer

import cv2
import numpy as np
import open3d as o3d
from py_trees.common import Status
from typing_extensions import Optional, TYPE_CHECKING, Tuple, Dict

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import scale_cam_intrinsics
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

if TYPE_CHECKING:
    import numpy.typing as npt


class ImageClusterExtractor(BaseAnnotator):
    """Extract object clusters from images using color segmentation.

    This annotator performs the following steps:

    * Converts RGB image to HSV color space
    * Applies HSV thresholding based on configured parameters
    * Detects and filters contours based on size
    * Generates point clouds from depth data for each contour
    * Creates ObjectHypothesis annotations with ROIs and masks
    * Provides visualization of detected clusters

    The HSV thresholds can be adjusted dynamically based on color queries.
    """

    class ViewMode:
        """Visualization modes for the annotator output."""

        masked_object: int = 1
        """Show masked RGB image of detected objects"""

        depth_mask: int = 2
        """Show depth mask of detected objects"""

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for ImageClusterExtractor.

        Parameters:

        * HSV thresholding ranges
        * Contour filtering parameters
        * Point cloud generation settings
        * Color name to HSV range mappings
        * Outlier removal parameters
        """

        class Parameters:
            """Parameter class containing all configurable settings."""

            def __init__(self) -> None:
                """Initialize default parameter values."""
                self.hsv_min: Tuple[int, int, int] = (150, 130, 85)
                self.hsv_max: Tuple[int, int, int] = (200, 255, 255)
                self.erosion_iterations: int = 2

                # This parameter controls the filtering of the initial list of contours.
                # It is used to avoid very small contours when calculating 3d points etc.
                self.contour_min_size: int = 1000

                self.color_name_to_hsv_range: Dict[
                    str, Dict[str, Tuple[int, int, int]]
                ] = dict()
                self.color_name_to_hsv_range["blue"] = {
                    "hsv_min": (150, 130, 85),
                    "hsv_max": (200, 255, 255),
                }
                self.color_name_to_hsv_range["red"] = {
                    "hsv_min": (215, 150, 95),
                    "hsv_max": (280, 255, 255),
                }

                self.outlier_removal: bool = True
                self.outlier_removal_nb_neighbors: int = 20
                self.outlier_removal_std_ratio: float = 2.0
                self.num_of_objects: int = 2

                # The minimal amount of 3D points of the object's pointcloud
                # This check is applied AFTER self.contour_min_size
                self.min_points_threshold: int = 62

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    # def dyn_rec_callback(self, config, level):
    #    self.rk_logger.info("Received reconf call: " + str(config))
    #    self.descriptor.parameters.hsv_min = (config['h_min'], config['s_min'], config['v_min'])
    #    self.descriptor.parameters.hsv_max = (config['h_max'], config['s_max'], config['v_max'])
    #    self.descriptor.parameters.erosion_iterations = config['erosion_iterations']
    #    self.descriptor.parameters.contour_min_size = config['contour_min_size']
    #    self.descriptor.parameters.num_of_objects = config['num_of_objects']
    #    self.descriptor.parameters.min_points_threshold = config['min_points_threshold']
    #    return config

    def __init__(
        self,
        name: str = "ImageClusterExtractor",
        descriptor: "ImageClusterExtractor.Descriptor" = Descriptor(),
    ) -> None:
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.color: Optional[npt.NDArray] = None
        self.depth: Optional[npt.NDArray] = None
        self.query = None
        self.cam_intrinsics = None

        # TODO Refactor this to new RPC method without using ROS
        # Add variables (name, description, default value, min, max, edit_method)
        # self.declare_parameter("h_min", self.descriptor.parameters.hsv_min[0],
        #                       ParameterDescriptor(min_value=str(0), max_value=str(359)))
        # self.declare_parameter("h_max", self.descriptor.parameters.hsv_max[0],
        #                       ParameterDescriptor(min_value=str(0), max_value=str(359)))
        # self.declare_parameter("s_min", self.descriptor.parameters.hsv_min[1],
        #                       ParameterDescriptor(min_value=str(0), max_value=str(255)))
        # self.declare_parameter("s_max", self.descriptor.parameters.hsv_max[1],
        #                       ParameterDescriptor(min_value=str(0), max_value=str(255)))

        # self.declare_parameter("v_min", self.descriptor.parameters.hsv_min[2],
        #                       ParameterDescriptor(min_value=str(0), max_value=str(255)))
        # self.declare_parameter("v_max", self.descriptor.parameters.hsv_max[2],
        #                       ParameterDescriptor(min_value=str(0), max_value=str(255)))

        # self.declare_parameter("erosion_iterations", self.descriptor.parameters.erosion_iterations,
        #                       ParameterDescriptor(min_value=str(0), max_value=str(20)))

        # self.declare_parameter("contour_min_size", self.descriptor.parameters.contour_min_size,
        #                       ParameterDescriptor(min_value=str(0), max_value=str(20000)))

        # self.declare_parameter("num_of_objects", self.descriptor.parameters.num_of_objects,
        #                       ParameterDescriptor(min_value=str(1), max_value=str(6)))

        # self.declare_parameter("min_points_threshold", self.descriptor.parameters.min_points_threshold,
        # ParameterDescriptor(min_value=str(0), max_value=str(100)))

        self.display_mode = self.ViewMode.masked_object

    def adjust_hsv_threshold_to_query(self) -> None:
        """Adjust HSV thresholds based on color query.

        Checks for a color query in the CAS and updates the HSV thresholding parameters
        if a matching color is found in the color_name_to_hsv_range mapping.
        """
        try:
            self.query = self.get_cas().get(CASViews.QUERY)
        except KeyError:
            return

        if not self.query:
            return

        if len(self.query.obj.color) == 0:
            return

        color = self.query.obj.color[0]
        if color not in self.descriptor.parameters.color_name_to_hsv_range:
            return

        self.descriptor.parameters.hsv_min = (
            self.descriptor.parameters.color_name_to_hsv_range[color]["hsv_min"]
        )
        self.descriptor.parameters.hsv_max = (
            self.descriptor.parameters.color_name_to_hsv_range[color]["hsv_max"]
        )

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """Process input images to detect and annotate object clusters.

        The method:

        * Scales color image to match depth image
        * Converts to HSV and applies thresholding
        * Detects and filters contours
        * Generates point clouds for each contour
        * Creates ObjectHypothesis annotations
        * Generates visualization output

        :return: SUCCESS if clusters found, FAILURE if no clusters
        :raises Exception: If no contours found or processing fails
        """
        start_timer = default_timer()

        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)
        self.depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
        self.cam_intrinsics = copy.deepcopy(self.get_cas().get(CASViews.CAM_INTRINSIC))

        # Scale the image down so that it matches the depth image size
        resized_color = None
        try:
            resized_color = get_scaled_color_image_for_depth_image(
                self.get_cas(), self.color
            )
            scale_cam_intrinsics(self)
        except RuntimeError as e:
            self.rk_logger.error(
                "No color to depth ratio set by your camera driver! Can't scale image for Point Cloud creation."
            )
            raise Exception(
                "No color to depth ratio set by your camera driver! Can't scale image for Point Cloud creation."
            )

        self.hsv = cv2.cvtColor(resized_color, cv2.COLOR_BGR2HSV_FULL)

        self.adjust_hsv_threshold_to_query()

        # Apply the HSV threshold on the image and find contours on the resultant binary image
        hsv_mask = cv2.inRange(
            self.hsv,
            self.descriptor.parameters.hsv_min,
            self.descriptor.parameters.hsv_max,
        )
        contours, hierarchy = cv2.findContours(
            image=hsv_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            # Fail if no contours have been found
            raise Exception(f"Couldn't find contour")

        # Visualization purposes
        result = copy.deepcopy(resized_color)
        result = cv2.bitwise_and(result, result, mask=hsv_mask)

        # Calculate the areas spanned by the Contours in order to find the largest ones
        # Filter the contours based on hierarchy
        contours = [
            contour for i, contour in enumerate(contours) if hierarchy[0][i][3] == -1
        ]
        contour_areas = np.asarray([cv2.contourArea(c) for c in contours])
        filtered_contour_areas = [
            area
            for area in contour_areas
            if area > self.descriptor.parameters.contour_min_size
        ]
        sorted_areas = sorted(filtered_contour_areas, reverse=True)
        largest_elements = sorted_areas[: self.descriptor.parameters.num_of_objects]
        filtered_contours = [
            contours[i]
            for i, area in enumerate(contour_areas)
            if area in largest_elements
        ]

        # Draw a 'mask' based on the contours
        # This will completely mask out the area inside a contour instead of just drawing contour points
        # List to store ObjectHypothesis instances
        object_hypotheses = []
        visualized_geometries = []
        for i, contour in enumerate(filtered_contours):
            biggest_contour_mask = np.zeros_like(self.depth, dtype=np.uint8)
            cv2.drawContours(
                image=biggest_contour_mask,
                contours=[contour],
                contourIdx=-1,
                color=255,
                thickness=cv2.FILLED,
            )

            # Apply Erosion to remove background points which might be included due to imperfect calibration between
            # RGB and Depth images
            kernel = np.ones((5, 5), np.uint8)
            biggest_contour_mask = cv2.erode(
                biggest_contour_mask,
                kernel,
                iterations=self.descriptor.parameters.erosion_iterations,
            )
            biggest_contour_mask_rgb = cv2.cvtColor(
                biggest_contour_mask, cv2.COLOR_GRAY2BGR
            )

            # Cluster creation
            # Here we'll exploit, that the depth creation in open3d will ignore depth values of 0
            # https://github.com/isl-org/Open3D/issues/1662
            # color_rgb = cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB)
            color_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
            depth_masked = copy.deepcopy(self.depth)

            depth_masked = np.where(
                biggest_contour_mask == 255, depth_masked, 0
            )  # mask all depth values

            o3d_color = o3d.geometry.Image(color_rgb)
            o3d_depth = o3d.geometry.Image(
                depth_masked
            )  # Please note that depth values should be in mm
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth, convert_rgb_to_intensity=False, depth_trunc=9.0
            )

            cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self.cam_intrinsics
            )

            if self.descriptor.parameters.outlier_removal:
                cloud, indices_after_outlier = cloud.remove_statistical_outlier(
                    nb_neighbors=self.descriptor.parameters.outlier_removal_nb_neighbors,
                    std_ratio=self.descriptor.parameters.outlier_removal_std_ratio,
                )
            visualized_geometries.append(cloud)

            if len(cloud.points) >= self.descriptor.parameters.min_points_threshold:
                # Create an ObjectHypothesis instance

                object_hypothesis = ObjectHypothesis()
                # Set ObjectHypothesis attributes
                object_hypothesis.id = str(i)
                object_hypothesis.source = self.name
                object_hypothesis.points = cloud

                # Calculate bounding rectangle for the current contour
                x, y, w, h = cv2.boundingRect(contour)
                # Set ROI attributes for the ObjectHypothesis
                object_hypothesis.roi.roi.pos.x = x
                object_hypothesis.roi.roi.pos.y = y
                object_hypothesis.roi.roi.width = w
                object_hypothesis.roi.roi.height = h

                # TODO Generate proper Masks

                # TODO Allow multiple objects to be detected

                object_hypotheses.append(object_hypothesis)
                self.get_cas().annotations.append(object_hypothesis)

        #
        # Create visualization Output
        #
        if self.display_mode == self.ViewMode.masked_object:
            visualization_img = copy.deepcopy(result)
        elif self.display_mode == self.ViewMode.depth_mask:
            visualization_img = copy.deepcopy(biggest_contour_mask_rgb)
        else:
            visualization_img = copy.deepcopy(result)

        for oh in object_hypotheses:
            assert isinstance(oh, ObjectHypothesis)
            oh_roi = oh.roi.roi
            upper_left = (oh_roi.pos.x, oh_roi.pos.y)
            upper_left_text = (oh_roi.pos.x, oh_roi.pos.y - 5)

            font = cv2.FONT_HERSHEY_COMPLEX
            visualization_img = cv2.putText(
                visualization_img,
                f"ROI-{oh.id}({len(oh.points.points)})",
                upper_left_text,
                font,
                0.5,
                (0, 0, 255),
                1,
                2,
            )
            visualization_img = cv2.rectangle(
                visualization_img,
                upper_left,
                (oh_roi.pos.x + oh_roi.width, oh_roi.pos.y + oh_roi.height),
                (0, 0, 255),
                2,
            )

        self.get_annotator_output_struct().set_image(visualization_img)
        self.get_annotator_output_struct().set_geometries(visualized_geometries)
        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def key_callback(self, key: int) -> None:
        """Handle keyboard input to change visualization mode.

        :param key: ASCII value of pressed key
        """
        if key == ord("1"):
            self.display_mode = self.ViewMode.masked_object
        if key == ord("2"):
            self.display_mode = self.ViewMode.depth_mask

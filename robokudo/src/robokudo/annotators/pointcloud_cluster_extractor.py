"""Point cloud clustering and object detection.

This module provides annotators for:

* Extracting object clusters from point clouds above detected planes
* Generating ROIs and masks for detected clusters
* Visualizing detected clusters with unique colors

The module uses DBSCAN clustering and oriented bounding boxes (OBB) for object detection.

.. note::
   The clustering process requires a plane annotation in the CAS to function.
"""

from __future__ import annotations
import copy
import logging
from timeit import default_timer
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from typing_extensions import Tuple, List, TYPE_CHECKING
from py_trees.common import Status
from robokudo.annotators.core import ThreadedAnnotator, BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Plane
from robokudo.types.cv import ImageROI
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import draw_bounding_boxes_from_object_hypotheses
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo.utils.o3d_helper import put_obb_on_target_obb, concatenate_clouds
from robokudo.utils.transform import get_transform_matrix

if TYPE_CHECKING:
    import numpy.typing as npt

DILATION_KERNEL = np.ones((3, 3), np.uint8)


def generate_roi_with_mask_from_points(
    image_height: int,
    image_width: int,
    pc_cam_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    cloud: o3d.geometry.PointCloud,
    color2depth_ratio: Tuple[int, int] = (1, 1),
) -> Tuple[ImageROI, npt.NDArray]:
    """Generate ROI and mask from projected point cloud points.

    Projects 3D points to image plane and creates:

    * ROI bounding the projected points
    * Binary mask of projected points
    * Dilated mask to account for point sparsity

    :param image_height: Height of the target image
    :param image_width: Width of the target image
    :param pc_cam_intrinsics: Camera intrinsic parameters
    :param cloud: Point cloud to project
    :param color2depth_ratio: Scale ratio between color and depth images, defaults to (1, 1)
    :return: Tuple of (ROI with mask, full image mask)
    """
    # TODO: Use robokudo.utils.o3d_helper.get_mask_from_pointcloud

    # Project 3D points to image plane for ROI generation
    # This will project the matched points into the full-size input image
    k = pc_cam_intrinsics.intrinsic_matrix
    cropped_3d_points = np.asarray(cloud.points)
    uvd = cropped_3d_points @ k.T
    x = (uvd[:, 0] / uvd[:, 2]).astype(int)
    y = (uvd[:, 1] / uvd[:, 2]).astype(int)

    # We have to respect the ratio of the depth and rgb image to properly project back.
    # Invert from RGB-> Depth to Depth->RGB
    depth2color_ratio = (1 / color2depth_ratio[0], 1 / color2depth_ratio[1])
    x_scaled = (x * depth2color_ratio[0]).astype(int)
    y_scaled = (y * depth2color_ratio[1]).astype(int)

    full_mask = np.zeros((image_height, image_width), np.uint8)
    full_mask[y_scaled, x_scaled] = 255
    roi = ImageROI()
    roi.roi.pos.x = x_scaled.min()
    roi.roi.pos.y = y_scaled.min()
    roi.roi.width = x_scaled.max() - x_scaled.min()
    roi.roi.height = y_scaled.max() - y_scaled.min()

    roi.mask = full_mask[
        roi.roi.pos.y : roi.roi.pos.y + roi.roi.height,
        roi.roi.pos.x : roi.roi.pos.x + roi.roi.width,
    ]
    roi.mask = cv2.dilate(roi.mask, DILATION_KERNEL, iterations=1)

    return roi, full_mask


def cluster_points(
    input_cloud: o3d.geometry.PointCloud,
    eps: float,
    min_points: int,
    rk_logger: logging.Logger,
) -> List[int]:
    """Cluster points using DBSCAN algorithm.

    :param input_cloud: Point cloud to cluster
    :param eps: DBSCAN epsilon parameter (density threshold)
    :param min_points: Minimum points for a cluster
    :param rk_logger: Logger instance for progress reporting
    :return: List of point indices for each cluster

    .. note::
       Points labeled as noise (label=-1) are ignored in the output.
    """
    # Only print progress of DBSCAN when in logging level DEBUG
    if rk_logger.getEffectiveLevel() < logging.getLevelNamesMapping()["DEBUG"]:
        print_progress = True
    else:
        print_progress = False
    labels = np.array(
        input_cloud.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=print_progress
        )
    )

    max_label = labels.max()
    rk_logger.debug(f"point cloud has {max_label + 1} clusters")
    all_cluster_indices = []
    # DBScan returns label==-1 for all points that appear to be noise
    # This will be ignored
    for val in range(0, labels.max() + 1):
        cluster_indices = np.where(labels == val)[0]
        all_cluster_indices.append(cluster_indices)

    return all_cluster_indices


class PointCloudClusterExtractor(ThreadedAnnotator):
    """Annotator for object detection.

    The main idea is to look for the Plane annotation in the CAS extract the points that belong to the plane,
    constructs an oriented bounding box around it, pull it up to create volume above the plane and
    then extract all the inlier points. After that, a clustering process is executed
    and each cluster is annotated in the CAS.

    .. warning::
       Requires a Plane annotation in the CAS to function.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for point cloud clustering."""

        class Parameters:
            """Parameters for configuring cluster extraction."""

            def __init__(self):
                self.dbscan_min_cluster_count: int = 90
                """Minimum points for DBSCAN cluster"""

                self.min_cluster_count: int = 1000
                """Minimum total points for valid cluster"""

                self.min_on_plane_point_count: int = 90
                """Minimum points above plane, defaults to 90"""

                self.eps: float = 0.04
                """DBSCAN epsilon parameter (density threshold)"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "PointCloudClusterExtractor",
        descriptor: "PointCloudClusterExtractor.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the cluster extractor.

        :param name: The name of this annotator instance, defaults to "PointCloudClusterExtractor"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)

    def points_and_indices_from_plane(
        self,
        plane_model: Plane,
        cloud: o3d.geometry.PointCloud,
    ) -> Tuple[
        o3d.geometry.OrientedBoundingBox,
        o3d.geometry.OrientedBoundingBox,
        o3d.geometry.PointCloud,
        List[int],
        o3d.geometry.PointCloud,
    ]:
        """Extract points above a detected plane.

        Based on an input cloud and a plane model, extract all the points that are above that plane in the input cloud.
        Returns various data structures for further analysis and visualization.

        :param plane_model:
        :param cloud:
        :return: A tuple consisting of:
            plane_obb = The OBB around the points belonging to the plane_model in cloud
            on_plane_obb = The OBB put above plane_obb, with the same width and depth, but extended upwards
            on_plane_cloud = Points found in on_plane_obb
            on_plane_cloud_indices = Indices of Points found in on_plane_obb
            outlier_cloud = All points that do not belong to the plane/table itself.
        :raises Exception: If insufficient points found above plane

        .. note::
           The function ensures the plane normal points upward by rotating
           the OBB if necessary.
        """

        plane_model = plane_model
        plane_cloud = cloud.select_by_index(plane_model.inliers)

        # This cloud will contain everything that doesn't belong to the plane
        # cloud and was in the analysis of the PlaneAnnotation.
        outlier_cloud = cloud.select_by_index(plane_model.inliers, invert=True)
        plane_obb = plane_cloud.get_oriented_bounding_box()
        assert isinstance(plane_obb, o3d.geometry.OrientedBoundingBox)

        # Do we need to fix the Plane OBB orientation?
        # The normal should point 'up' from the detected plane
        vec_plane_obb_normal = np.matmul(
            np.array([0, 0, 1]), plane_obb.R
        )  # transform normal vector with plane rotation
        # Dot product between the vector pointing to the plane center from 0,0,0 and the plane obb normal
        val = np.dot(plane_obb.center, vec_plane_obb_normal)
        # print(f"Dot product between VP and Plane OBB Normal: {val}")
        if val > 0:
            # fix the OBB orientation. It is pointing downwards now
            # Rotate by 180 deg around X
            rotation = R.from_euler("x", np.radians(180)).as_matrix()
            new_obb_rotation = np.matmul(plane_obb.R, rotation)
            plane_obb.R = new_obb_rotation

        # ------------------------------------------------
        # Plane OBB should be good now (normal points up)
        # Begin the actual segmentation

        # Construct OBB which is used to segment objects on the plane
        # We need to transform it so that it is on top of the found plane
        above_plane_obb = o3d.geometry.OrientedBoundingBox()
        above_plane_obb.extent = np.array(
            [plane_obb.extent[0], plane_obb.extent[1], 0.8]
        )

        on_plane_obb = put_obb_on_target_obb(above_plane_obb, plane_obb)
        on_plane_cloud = outlier_cloud.crop(on_plane_obb)
        on_plane_cloud_indices = on_plane_obb.get_point_indices_within_bounding_box(
            outlier_cloud.points
        )
        assert isinstance(on_plane_cloud, o3d.geometry.PointCloud)
        on_plane_point_count = len(on_plane_cloud.points)
        if on_plane_point_count < self.descriptor.parameters.min_on_plane_point_count:
            self.feedback_message = (
                f"above plane point cloud has not enough "
                f"points ({self.descriptor.parameters.on_plane_point_count} "
                f"< {self.descriptor.parameters.min_on_plane_point_count}). Skipping"
            )
            raise Exception(self.feedback_message)

        return (
            plane_obb,
            on_plane_obb,
            on_plane_cloud,
            on_plane_cloud_indices,
            outlier_cloud,
        )

    @catch_and_raise_to_blackboard
    def compute(self) -> Status:
        """Process point cloud to detect and annotate object clusters.

        The method:

        * Extracts points above detected plane
        * Clusters points using DBSCAN
        * Creates ObjectHypothesis annotations for each cluster
        * Generates ROIs and masks for visualization
        * Visualizes clusters with unique colors

        :return: SUCCESS if clusters are found, FAILURE if no clusters or errors
        :raises Exception: If no plane model in CAS or insufficient points
        """
        start_timer = default_timer()
        self.rk_logger.info("PCE Start")
        cloud = self.get_cas().get(CASViews.CLOUD)
        color2depth_ratio = self.get_cas().get(CASViews.COLOR2DEPTH_RATIO)
        pc_cam_intrinsics = self.get_cas().get(CASViews.PC_CAM_INTRINSIC)
        assert isinstance(pc_cam_intrinsics, o3d.camera.PinholeCameraIntrinsic)

        color = self.get_cas().get(CASViews.COLOR_IMAGE)
        height, width, d = color.shape
        vis_mask = np.zeros((height, width), np.uint8)
        plane_models = self.get_cas().filter_annotations_by_type(Plane)
        if plane_models is None or plane_models == []:
            self.feedback_message = "No plane model in CAS. Aborting."
            raise Exception(self.feedback_message)

        plane_model: Plane = plane_models[0]

        # Segment the pointcloud first. Get the points inside the volume above the given plane and related variables
        (
            plane_obb,
            on_plane_obb,
            on_plane_cloud,
            on_plane_cloud_indices,
            outlier_cloud,
        ) = self.points_and_indices_from_plane(plane_model=plane_model, cloud=cloud)

        # Cluster the points above the plane to get a list of point indices for each cluster
        all_cluster_indices = cluster_points(
            input_cloud=on_plane_cloud,
            eps=self.descriptor.parameters.eps,
            min_points=self.descriptor.parameters.dbscan_min_cluster_count,
            rk_logger=self.rk_logger,
        )

        cmap = plt.get_cmap("tab20")
        clusters = {}
        all_vis_clouds = []
        object_hypotheses = []
        cluster_idx = 0
        for cluster_indices in all_cluster_indices:
            if len(cluster_indices) < self.descriptor.parameters.min_cluster_count:
                self.rk_logger.debug(
                    f"Skipping cluster {cluster_idx} with {len(cluster_indices)} points"
                )
                continue

            # Work the indices back from our crop towards our original pointcloud
            # This will allow us to reference the full Cloud in the CAS
            # directly, and we don't need to store partial point clouds
            cluster_indices_in_outlier_cloud = np.take(
                on_plane_cloud_indices, cluster_indices
            )
            cluster_cloud = outlier_cloud.select_by_index(
                list(cluster_indices_in_outlier_cloud)
            )

            clusters[cluster_idx] = {
                "cloud": cluster_cloud,
                "vis_cloud": copy.deepcopy(cluster_cloud),
            }

            clusters[cluster_idx]["vis_cloud"].paint_uniform_color(
                cmap.colors[(cluster_idx % 20)]
            )  # limit to 0-19 range due to map size
            all_vis_clouds.append(clusters[cluster_idx]["vis_cloud"])

            self.rk_logger.debug(
                f"Extracted cluster {cluster_idx} with {len(cluster_cloud.points)} points"
            )

            # Add each cluster to CAS
            object_hypothesis = ObjectHypothesis()
            object_hypothesis.source = self.name
            object_hypothesis.points = cluster_cloud
            object_hypothesis.point_indices = [cluster_indices_in_outlier_cloud]

            object_hypothesis.id = str(cluster_idx)
            object_hypothesis.roi, full_mask_cluster = (
                generate_roi_with_mask_from_points(
                    image_height=height,
                    image_width=width,
                    pc_cam_intrinsics=pc_cam_intrinsics,
                    cloud=cluster_cloud,
                    color2depth_ratio=color2depth_ratio,
                )
            )
            # Add the image pixel belonging to the cluster to the mask containing all found objects
            vis_mask[full_mask_cluster == 255] = 255

            self.get_cas().annotations.append(object_hypothesis)
            object_hypotheses.append(object_hypothesis)
            cluster_idx += 1

        if len(object_hypotheses) == 0:
            self.rk_logger.warning(f"No Clusters have been found.")
            end_timer = default_timer()
            self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
            return Status.FAILURE

        ##### Visualization
        # 3D
        visualization_cloud = concatenate_clouds(all_vis_clouds)

        plane_normal = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        t = get_transform_matrix(plane_obb.R, plane_obb.center)
        plane_normal.transform(t)

        x = [
            {"name": "Visualization cloud PCE", "geometry": visualization_cloud},
            {"name": "Plane OBB", "geometry": plane_obb},
            {"name": "Plane Normal", "geometry": plane_normal},
            {"name": "On Plane OBB", "geometry": on_plane_obb},
            {"name": "Cloud", "geometry": cloud},
        ]

        self.get_annotator_output_struct().set_geometries(x)

        vis_mask = cv2.dilate(vis_mask, DILATION_KERNEL, iterations=1)
        # make mask a 3 channel mask in order to make it work with addWeighted
        mask3d = np.stack((vis_mask, vis_mask, vis_mask), axis=2)

        # Overlay the mask with the input RGB image
        visualization_img = cv2.addWeighted(color, 0.5, mask3d, 0.5, 0)
        draw_bounding_boxes_from_object_hypotheses(
            visualization_img,
            object_hypotheses,
            lambda oh: f"ROI-{oh.id}({len(oh.points.points)})",
        )

        self.get_annotator_output_struct().set_image(visualization_img)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        self.rk_logger.info("PCE Done")
        return Status.SUCCESS


class NaivePointCloudClusterExtractor(ThreadedAnnotator):
    """Annotator for object detection. Simply takes an input cloud and clusters it.

    .. note::
       Unlike PointCloudClusterExtractor, this does not require a plane model.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for naive point cloud clustering."""

        class Parameters:
            """Parameters for configuring cluster extraction."""

            def __init__(self) -> None:
                self.dbscan_min_cluster_count: int = 90
                """Minimum points for DBSCAN cluster"""

                self.min_cluster_count: int = 1000
                """Minimum total points for valid cluster"""

                self.min_on_plane_point_count: int = 90
                """Minimum points above plane"""

                self.eps: float = 0.04
                """DBSCAN epsilon parameter (density threshold)"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "NaivePointCloudClusterExtractor",
        descriptor: "NaivePointCloudClusterExtractor.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the naive cluster extractor.

        :param name: The name of this annotator instance, defaults to "NaivePointCloudClusterExtractor"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)

    @catch_and_raise_to_blackboard
    def compute(self) -> Status:
        """Process point cloud to detect and annotate object clusters.

        The method:

        * Clusters points directly using DBSCAN
        * Creates ObjectHypothesis annotations for each cluster
        * Generates ROIs and masks for visualization
        * Visualizes clusters with unique colors

        :return: SUCCESS if clusters found, FAILURE if no clusters or errors
        :raises Exception: If insufficient points in clusters
        """
        start_timer = default_timer()
        cloud = self.get_cas().get(CASViews.CLOUD)
        color2depth_ratio = self.get_cas().get(CASViews.COLOR2DEPTH_RATIO)
        pc_cam_intrinsics = self.get_cas().get(CASViews.PC_CAM_INTRINSIC)
        assert isinstance(pc_cam_intrinsics, o3d.camera.PinholeCameraIntrinsic)

        color = self.get_cas().get(CASViews.COLOR_IMAGE)
        height, width, d = color.shape
        vis_mask = np.zeros((height, width), np.uint8)

        if len(cloud.points) == 0:
            self.feedback_message = "Input cloud is empty - Can't compute clustering"
            raise Exception("Input cloud is empty - Can't compute clustering")

        # Cluster the points above the plane to get a list of point indices for each cluster
        all_cluster_indices = cluster_points(
            input_cloud=cloud,
            eps=self.descriptor.parameters.eps,
            min_points=self.descriptor.parameters.dbscan_min_cluster_count,
            rk_logger=self.rk_logger,
        )

        cmap = plt.get_cmap("tab20")
        clusters = {}
        all_vis_clouds = []
        object_hypotheses = []
        cluster_idx = 0
        for cluster_indices in all_cluster_indices:
            if len(cluster_indices) < self.descriptor.parameters.min_cluster_count:
                self.rk_logger.debug(
                    f"Skipping cluster {cluster_idx} with {len(cluster_indices)} points"
                )
                continue

            cluster_cloud = cloud.select_by_index(list(cluster_indices))

            clusters[cluster_idx] = {
                "cloud": cluster_cloud,
                "vis_cloud": copy.deepcopy(cluster_cloud),
            }

            clusters[cluster_idx]["vis_cloud"].paint_uniform_color(
                cmap.colors[(cluster_idx % 20)]
            )  # limit to 0-19 range due to map size
            all_vis_clouds.append(clusters[cluster_idx]["vis_cloud"])

            self.rk_logger.debug(
                f"Extracted cluster {cluster_idx} with {len(cluster_cloud.points)} points"
            )

            # Add each cluster to CAS
            object_hypothesis = ObjectHypothesis()
            object_hypothesis.source = self.name
            object_hypothesis.points = cluster_cloud
            object_hypothesis.point_indices = [cluster_indices]

            object_hypothesis.id = str(cluster_idx)
            object_hypothesis.roi, full_mask_cluster = (
                generate_roi_with_mask_from_points(
                    image_height=height,
                    image_width=width,
                    pc_cam_intrinsics=pc_cam_intrinsics,
                    cloud=cluster_cloud,
                    color2depth_ratio=color2depth_ratio,
                )
            )
            # Add the image pixel belonging to the cluster to the mask containing all found objects
            vis_mask[full_mask_cluster == 255] = 255

            self.get_cas().annotations.append(object_hypothesis)
            object_hypotheses.append(object_hypothesis)
            cluster_idx += 1

        if len(object_hypotheses) == 0:
            self.rk_logger.warning(f"No Clusters have been found.")
            end_timer = default_timer()
            self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
            return Status.FAILURE

        ##### Visualization
        # 3D
        visualization_cloud = concatenate_clouds(all_vis_clouds)

        x = [
            {"name": "Visualization cloud PCE", "geometry": visualization_cloud},
        ]

        self.get_annotator_output_struct().set_geometries(x)

        vis_mask = cv2.dilate(vis_mask, DILATION_KERNEL, iterations=1)
        # make mask a 3-channel mask in order to make it work with addWeighted
        mask3d = np.stack((vis_mask, vis_mask, vis_mask), axis=2)

        # Overlay the mask with the input RGB image
        visualization_img = cv2.addWeighted(color, 0.5, mask3d, 0.5, 0)
        draw_bounding_boxes_from_object_hypotheses(
            visualization_img,
            object_hypotheses,
            lambda oh: f"ROI-{oh.id}({len(oh.points.points)})",
        )

        self.get_annotator_output_struct().set_image(visualization_img)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        self.rk_logger.info("PCE Stop")
        return Status.SUCCESS

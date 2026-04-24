from __future__ import annotations
from robokudo.annotators.core import ThreadedAnnotator

import copy
from enum import Enum
import cv2
import numpy as np
import open3d as o3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, List, Any, Dict, Optional, Tuple

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import PoseAnnotation
from robokudo.types.cv import TSDFAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.comparators import TranslationComparator
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image
from robokudo.utils.o3d_helper import scale_o3d_cam_intrinsics
from robokudo.utils.transform import get_transform_matrix_from_q

if TYPE_CHECKING:
    import numpy.typing as npt


class TSDFAnnotatorVisualizationModes(Enum):
    """Enumeration of supported visualization modes for TSDF volumes."""

    NONE = "none"
    MESH = "mesh"
    POINTCLOUD = "pointcloud"
    VOXEL_POINTCLOUD = "voxel_pointcloud"


class TSDFAnnotator(ThreadedAnnotator):
    """Annotator for per-object TSDF volume integration.

    .. note::
        The annotator requires a camera to world transform and object poses to integrate the TSDF volumes.
        To segment objects out of the image a mask is required too. Higher quality masks yield better results.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for the TSDF annotator."""

        class Parameters(BaseAnnotator.Descriptor.Parameters):
            """Parameters for configuring the TSDF annotator."""

            def __init__(self) -> None:
                self.voxel_length: float = 0.005
                """Voxel length for the TSDF volume of each object."""

                self.sdf_trunc: float = 0.04
                """SDF truncation distance for the TSDF volume of each object."""

                self.depth_scale: float = 1000.0
                """Depth scale for the TSDF volume integration."""

                self.depth_trunc: float = 3.0
                """Depth truncation distance for the TSDF volume integration."""

                self.geometry_visualization_mode: "TSDFAnnotatorVisualizationModes" = (
                    TSDFAnnotatorVisualizationModes.MESH
                )
                """Whether to visualize the TSDF as a mesh or a point cloud. """

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "TSDFAnnotator",
        descriptor: "TSDFAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        super().__init__(name, descriptor)

        self.translation_comparator = TranslationComparator(
            weight=1.0, max_distance=0.4
        )
        """Translation comparator for comparing world positions between pipeline iterations."""

        self.tracked_objects: List[Dict[str, Any]] = []
        """A list of tracked object data, including the bounding box and the TSDF volume."""

    def find_tracked_object(
        self, pose: PoseAnnotation, threshold: float = 0.75
    ) -> Optional[Dict[str, Any]]:
        """Check whether the given bounding box is similar to one that has been seen before.

        :param pose: The pose to check
        :param threshold: The similarity threshold between 0.0 and 1.0 for considering the bounding box similar.
        :return: The most similar tracked object or none if no similar object was found.
        """
        best_sim: float = 0.0
        best_object: Optional[Dict[str, Any]] = None
        for tracked_object in self.tracked_objects:
            sim = self.translation_comparator.compute_similarity(
                pose.translation, tracked_object["pose"].translation
            )
            if sim > best_sim:
                best_object = tracked_object
                best_sim = sim
        return best_object if best_sim > threshold else None

    def compute(self) -> Status:
        """Generate a TSDF volume for each object hypothesis or integrate into existing volumes.

        :return: Whether the TSDF integration was successful or not.
        """
        cas = self.get_cas()
        if cas is None:
            return Status.FAILURE

        ohs: List[ObjectHypothesis] = cas.filter_annotations_by_type(ObjectHypothesis)
        if len(ohs) == 0:
            return Status.FAILURE

        color2depth_ratio: Tuple[float, float] = cas.get(CASViews.COLOR2DEPTH_RATIO)

        # Only process ohs with pose
        oh_data: List[
            Tuple[ObjectHypothesis, PoseAnnotation, Tuple[int, int, int, int]]
        ] = []
        for oh in ohs:
            poses = cas.filter_by_type(PoseAnnotation, oh.annotations)
            if len(poses) > 0:
                roi: Tuple[int, int, int, int] = oh.roi.roi.get_corner_points()
                roi = (
                    int(roi[0] * color2depth_ratio[0]),
                    int(roi[1] * color2depth_ratio[1]),
                    int(roi[2] * color2depth_ratio[0]),
                    int(roi[3] * color2depth_ratio[1]),
                )
                oh_data.append((oh, poses[0], roi))
            else:
                self.rk_logger.debug(f"No pose found for object {oh.id}, skipping...")

        if len(oh_data) == 0:
            return Status.FAILURE

        cam_intrinsic: o3d.camera.PinholeCameraIntrinsic = scale_o3d_cam_intrinsics(
            cas.get(CASViews.CAM_INTRINSIC), color2depth_ratio[0], color2depth_ratio[1]
        )
        color_image: npt.NDArray[np.uint8] = get_scaled_color_image_for_depth_image(
            cas, copy.deepcopy(self.get_cas().get(CASViews.COLOR_IMAGE))
        )
        depth_image: npt.NDArray[np.float32] = copy.deepcopy(
            self.get_cas().get(CASViews.DEPTH_IMAGE)
        )

        geometries: List[o3d.geometry.Geometry] = []
        masks: List[npt.NDArray[np.uint8]] = []
        for i, (oh, pose, roi) in enumerate(oh_data):
            mask: npt.NDArray[np.uint8] = oh.roi.mask
            if mask is None:
                self.rk_logger.debug(f"No mask found for object {oh.id}, skipping...")
                continue

            # Mask should be or (0, 1)
            if mask.max() == 255:
                mask = np.clip(mask, 0, 1)

            mask = get_scaled_color_image_for_depth_image(cas, mask)

            if mask.shape[:2] == depth_image.shape[:2]:
                masked_depth = depth_image * mask
                masked_color = color_image * mask[..., np.newaxis]
            else:
                masked_depth = np.zeros_like(depth_image, dtype=np.float32)
                masked_depth[roi[1] : roi[3], roi[0] : roi[2]] = (
                    depth_image[roi[1] : roi[3], roi[0] : roi[2]] * mask
                )

                masked_color = np.zeros_like(color_image, dtype=np.uint8)
                masked_color[roi[1] : roi[3], roi[0] : roi[2]] = (
                    color_image[roi[1] : roi[3], roi[0] : roi[2]]
                    * mask[..., np.newaxis]
                )

            masks.append(mask)

            # Convert BGR (cv2 default) to RGB (open3d default) in-place
            cv2.cvtColor(masked_color, cv2.COLOR_BGR2RGB, dst=masked_color)

            # Create RGBD image from masked data
            rgbd_masked: o3d.geometry.RGBDImage = (
                o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color=o3d.geometry.Image(masked_color),
                    depth=o3d.geometry.Image(masked_depth),
                    depth_scale=self.descriptor.parameters.depth_scale,
                    depth_trunc=self.descriptor.parameters.depth_trunc,
                    convert_rgb_to_intensity=False,
                )
            )

            pose_in_cam = get_transform_matrix_from_q(pose.rotation, pose.translation)

            tracked_object = self.find_tracked_object(pose)
            if tracked_object is not None:
                volume = tracked_object["volume"]
                tracked_object["pose"] = pose
            else:
                volume = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=self.descriptor.parameters.voxel_length,
                    sdf_trunc=self.descriptor.parameters.sdf_trunc,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                )
                self.tracked_objects.append(
                    {
                        "pose": pose,
                        "volume": volume,
                    }
                )

            # Integrate with pose in cam
            volume.integrate(
                rgbd_masked,
                intrinsic=cam_intrinsic,
                extrinsic=pose_in_cam,
            )

            volume_an = TSDFAnnotation(source=self.name)
            volume_an.volume = volume
            volume_an.transform = pose_in_cam
            oh.annotations.append(volume_an)

            vis_mode = self.descriptor.parameters.geometry_visualization_mode
            if vis_mode != TSDFAnnotatorVisualizationModes.NONE:
                vis_geom = self.get_visualization_geometries(volume_an)
                geometries.extend(vis_geom)

        vis_image = self.get_visualization_image(color_image, oh_data, masks)
        self.get_annotator_output_struct().set_image(vis_image)

        vis_mode = self.descriptor.parameters.geometry_visualization_mode
        if vis_mode != TSDFAnnotatorVisualizationModes.NONE:
            self.rk_logger.debug(
                f"Visualizing geometry for {len(geometries) // 3} objects"
            )
            self.get_annotator_output_struct().set_geometries(geometries)

        return Status.SUCCESS

    def get_visualization_image(
        self,
        image: npt.NDArray[np.uint8],
        oh_data: List[
            Tuple[ObjectHypothesis, PoseAnnotation, Tuple[int, int, int, int]]
        ],
        masks: List[npt.NDArray[np.uint8]],
    ) -> npt.NDArray[np.uint8]:
        """Get an image containing the bounding boxes and masks of objects.

        :param image: The image to add the visualizations to.
        :param oh_data: The object hypotheses to visualize.
        :param masks: The masks to visualize.
        :return: The image with the visualizations.
        """
        self.rk_logger.debug(f"Visualizing image for {len(oh_data)} objects")

        overlay = np.zeros_like(image, dtype=np.uint8)
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 0-1
        for i, (oh, _, roi) in enumerate(oh_data):
            cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

            mask = masks[i]  # (H, W)
            if mask.shape[:2] != image.shape[:2]:
                combined_mask[roi[1] : roi[3], roi[0] : roi[2]] = np.maximum(
                    combined_mask[roi[1] : roi[3], roi[0] : roi[2]], mask
                )
            else:
                combined_mask = np.maximum(combined_mask, mask)

        # Set the green channel (index 1 in RGB)
        overlay[:, :, 1] = 255 * combined_mask
        image = cv2.addWeighted(image, 1 - 0.5, overlay.astype(np.uint8), 0.5, 0)

        return image

    def get_visualization_geometries(
        self, volume: TSDFAnnotation
    ) -> List[o3d.geometry.Geometry]:
        """Get visualization geometries for the TSDF volume depending on the visualization mode.

        :param volume: The TSDF volume annotation to visualize.
        :return: List of visualization geometries for the given volume annotation.
        """
        vis_mode = self.descriptor.parameters.geometry_visualization_mode
        if vis_mode == TSDFAnnotatorVisualizationModes.POINTCLOUD:
            return [volume.get_point_cloud(), volume.get_coordinate_frame()]
        elif vis_mode == TSDFAnnotatorVisualizationModes.VOXEL_POINTCLOUD:
            return [volume.get_voxel_point_cloud(), volume.get_coordinate_frame()]
        elif vis_mode == TSDFAnnotatorVisualizationModes.MESH:
            return [volume.get_mesh(), volume.get_coordinate_frame()]
        else:
            self.rk_logger.warning(f"Visualization mode '{vis_mode}' not supported!")
            return []

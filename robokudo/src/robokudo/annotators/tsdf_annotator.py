from __future__ import annotations

# %% Imports
import copy
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import open3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, Any

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.cas import CASViews
from robokudo.io.camera_model_adapters import Open3DCameraModelAdapter
from robokudo.types.annotation import PoseAnnotation, TSDFAnnotation
from robokudo.types.cv import Rect
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.comparators import TranslationComparator
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image
from robokudo.utils.transform import get_transform_matrix_from_q

if TYPE_CHECKING:
    import numpy.typing as npt

    from robokudo.types.cv import ColorToDepthRatio


# %% Volume integration inputs
@dataclass
class VolumeIntegrationInput:
    """An object's pose and image region for volume integration."""

    object_hypothesis: ObjectHypothesis
    """Object whose volume is integrated."""

    pose: PoseAnnotation
    """Object pose relative to the camera."""

    depth_image_region: Rect
    """Object bounds in depth-image coordinates."""


# %% Volume integration


class TSDFAnnotatorVisualizationModes(Enum):
    """
    Enumeration of supported visualization modes for TSDF volumes.
    """

    NONE = "none"
    MESH = "mesh"
    POINTCLOUD = "pointcloud"
    VOXEL_POINTCLOUD = "voxel_pointcloud"


class TSDFAnnotator(ThreadedAnnotator):
    """
    Annotator for per-object TSDF volume integration.

    .. note::
        The annotator requires a camera to world transform and object poses to integrate the TSDF volumes.
        To segment objects out of the image a mask is required too. Higher quality masks yield better results.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration descriptor for the TSDF annotator.
        """

        class Parameters(BaseAnnotator.Descriptor.Parameters):
            """
            Parameters for configuring the TSDF annotator.
            """

            def __init__(self) -> None:
                self.voxel_length: float = 0.005
                """
                Voxel length for the TSDF volume of each object.
                """

                self.signed_distance_truncation: float = 0.04
                """
                Signed distance truncation for the TSDF volume of each object.
                """

                self.depth_scale: float = 1000.0
                """
                Depth scale for the TSDF volume integration.
                """

                self.depth_truncation: float = 3.0
                """
                Depth truncation distance for the TSDF volume integration.
                """

                self.geometry_visualization_mode: TSDFAnnotatorVisualizationModes = (
                    TSDFAnnotatorVisualizationModes.MESH
                )
                """
                Whether to visualize the TSDF as a mesh or a point cloud.
                """

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "TSDFAnnotator",
        descriptor: TSDFAnnotator.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)

        self.translation_comparator = TranslationComparator(
            weight=1.0, max_distance=0.4
        )
        """
        Translation comparator for comparing world positions between pipeline iterations.
        """

        self.tracked_objects: list[dict[str, Any]] = []
        """
        A list of tracked object data, including the bounding box and the TSDF volume.
        """

    def find_tracked_object(
        self, pose: PoseAnnotation, threshold: float = 0.75
    ) -> dict[str, Any] | None:
        """
        Check whether the given bounding box is similar to one that has been seen before.

        :param pose: The pose to check
        :param threshold: The similarity threshold between 0.0 and 1.0 for considering the bounding box similar.
        :return: The most similar tracked object or none if no similar object was found.
        """
        best_sim: float = 0.0
        best_object: dict[str, Any] | None = None
        for tracked_object in self.tracked_objects:
            sim = self.translation_comparator.compute_similarity(
                pose.translation, tracked_object["pose"].translation
            )
            if sim > best_sim:
                best_object = tracked_object
                best_sim = sim
        return best_object if best_sim > threshold else None

    def compute(self) -> Status:
        """
        Generate a TSDF volume for each object hypothesis or integrate into existing volumes.

        :return: Whether the TSDF integration was successful or not.
        """
        cas = self.get_cas()
        if cas is None:
            return Status.FAILURE

        object_hypotheses: list[ObjectHypothesis] = cas.filter_annotations_by_type(
            ObjectHypothesis
        )
        if len(object_hypotheses) == 0:
            return Status.FAILURE

        color_to_depth_ratio: ColorToDepthRatio = cas.color_to_depth_ratio

        volume_integration_inputs: list[VolumeIntegrationInput] = []
        for object_hypothesis in object_hypotheses:
            poses = cas.filter_by_type(PoseAnnotation, object_hypothesis.annotations)
            if len(poses) == 0:
                self.rk_logger.debug(
                    f"No pose found for object {object_hypothesis.id}, skipping..."
                )
                continue

            depth_image_region = color_to_depth_ratio.depth_region(
                object_hypothesis.roi.roi
            )
            volume_integration_inputs.append(
                VolumeIntegrationInput(
                    object_hypothesis=object_hypothesis,
                    pose=poses[0],
                    depth_image_region=depth_image_region,
                )
            )

        if len(volume_integration_inputs) == 0:
            return Status.FAILURE

        camera_intrinsic = Open3DCameraModelAdapter.to_intrinsic(
            cas.require_camera_observation().effective_camera_model.scaled(
                scale_x=color_to_depth_ratio.horizontal,
                scale_y=color_to_depth_ratio.vertical,
            )
        )
        color_image: npt.NDArray[np.uint8] = get_scaled_color_image_for_depth_image(
            cas, copy.deepcopy(self.get_cas().get(CASViews.COLOR_IMAGE))
        )
        depth_image: npt.NDArray[np.float32] = copy.deepcopy(
            self.get_cas().get(CASViews.DEPTH_IMAGE)
        )

        geometries: list[open3d.geometry.Geometry] = []
        masks: list[npt.NDArray[np.uint8]] = []
        for volume_integration_input in volume_integration_inputs:
            object_hypothesis = volume_integration_input.object_hypothesis
            pose = volume_integration_input.pose
            left, top, right, bottom = (
                volume_integration_input.depth_image_region.get_corner_points()
            )
            mask: npt.NDArray[np.uint8] = object_hypothesis.roi.mask
            if mask is None:
                self.rk_logger.debug(
                    f"No mask found for object {object_hypothesis.id}, skipping..."
                )
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
                masked_depth[top:bottom, left:right] = (
                    depth_image[top:bottom, left:right] * mask
                )

                masked_color = np.zeros_like(color_image, dtype=np.uint8)
                masked_color[top:bottom, left:right] = (
                    color_image[top:bottom, left:right] * mask[..., np.newaxis]
                )

            masks.append(mask)

            # Convert BGR (cv2 default) to RGB (open3d default) in-place
            cv2.cvtColor(masked_color, cv2.COLOR_BGR2RGB, dst=masked_color)

            # Create RGBD image from masked data
            rgbd_masked: open3d.geometry.RGBDImage = (
                open3d.geometry.RGBDImage.create_from_color_and_depth(
                    color=open3d.geometry.Image(masked_color),
                    depth=open3d.geometry.Image(masked_depth),
                    depth_scale=self.descriptor.parameters.depth_scale,
                    depth_trunc=self.descriptor.parameters.depth_truncation,
                    convert_rgb_to_intensity=False,
                )
            )

            pose_in_camera = get_transform_matrix_from_q(
                pose.rotation, pose.translation
            )

            tracked_object = self.find_tracked_object(pose)
            if tracked_object is not None:
                volume = tracked_object["volume"]
                tracked_object["pose"] = pose
            else:
                volume = open3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=self.descriptor.parameters.voxel_length,
                    sdf_trunc=self.descriptor.parameters.signed_distance_truncation,
                    color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                )
                self.tracked_objects.append(
                    {
                        "pose": pose,
                        "volume": volume,
                    }
                )

            # Integrate with pose in camera coordinates
            volume.integrate(
                rgbd_masked,
                intrinsic=camera_intrinsic,
                extrinsic=pose_in_camera,
            )

            volume_annotation = TSDFAnnotation(source=self.name)
            volume_annotation.volume = volume
            volume_annotation.transform = pose_in_camera
            object_hypothesis.annotations.append(volume_annotation)

            vis_mode = self.descriptor.parameters.geometry_visualization_mode
            if vis_mode != TSDFAnnotatorVisualizationModes.NONE:
                vis_geom = self.get_visualization_geometries(volume_annotation)
                geometries.extend(vis_geom)

        vis_image = self.get_visualization_image(
            color_image, volume_integration_inputs, masks
        )
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
        volume_integration_inputs: list[VolumeIntegrationInput],
        masks: list[npt.NDArray[np.uint8]],
    ) -> npt.NDArray[np.uint8]:
        """
        Get an image containing the bounding boxes and masks of objects.

        :param image: The image to add the visualizations to.
        :param volume_integration_inputs: Objects with poses and depth-image bounds.
        :param masks: The masks to visualize.
        :return: The image with the visualizations.
        """
        self.rk_logger.debug(
            f"Visualizing image for {len(volume_integration_inputs)} objects"
        )

        overlay = np.zeros_like(image, dtype=np.uint8)
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 0-1
        for hypothesis_index, volume_integration_input in enumerate(
            volume_integration_inputs
        ):
            left, top, right, bottom = (
                volume_integration_input.depth_image_region.get_corner_points()
            )
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            mask = masks[hypothesis_index]  # (H, W)
            if mask.shape[:2] != image.shape[:2]:
                combined_mask[top:bottom, left:right] = np.maximum(
                    combined_mask[top:bottom, left:right], mask
                )
            else:
                combined_mask = np.maximum(combined_mask, mask)

        # Set the green channel (index 1 in RGB)
        overlay[:, :, 1] = 255 * combined_mask
        image = cv2.addWeighted(image, 1 - 0.5, overlay.astype(np.uint8), 0.5, 0)

        return image

    def get_visualization_geometries(
        self, volume: TSDFAnnotation
    ) -> list[open3d.geometry.Geometry]:
        """
        Get visualization geometries for the TSDF volume depending on the visualization mode.

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

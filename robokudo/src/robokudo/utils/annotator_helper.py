"""Annotator utility functions for Robokudo.

This module provides helper functions for working with annotators and their data.
It supports:

* Coordinate frame transformations
* Point cloud manipulation
* Image processing
* Bounding box visualization
* Camera parameter handling
"""

from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np
import open3d as o3d
from typing_extensions import Callable, List, TYPE_CHECKING

from robokudo.cas import CASViews
from robokudo.types.annotation import PoseAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image
from robokudo.utils.o3d_helper import scale_o3d_cam_intrinsics
from robokudo.utils.transform import (
    get_transform_matrix_from_q,
    get_translation_from_transform_matrix,
    get_quaternion_from_transform_matrix,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from robokudo.annotators.core import BaseAnnotator
    from robokudo.cas import CAS


def transform_pose_from_cam_to_world(cas: CAS, pose: PoseAnnotation) -> PoseAnnotation:
    """Transform pose from camera to world coordinates.

    Uses transform from CAS to convert pose from camera frame to world frame.

    :param cas: CAS containing camera-to-world transform
    :param pose: Pose in camera coordinates
    :return: Transformed pose in world coordinates
    :raises AssertionError: If camera-to-world transform not in CAS
    """
    pose_mat = get_transform_matrix_from_q(pose.rotation, pose.translation)

    cam_to_world_transform = get_cam_to_world_transform_matrix(cas)

    pose_in_world_mat = np.matmul(cam_to_world_transform, pose_mat)

    new_pose = PoseAnnotation()
    new_pose.rotation = list(get_quaternion_from_transform_matrix(pose_in_world_mat))
    new_pose.translation = list(
        get_translation_from_transform_matrix(pose_in_world_mat)
    )
    new_pose.source = pose.source

    return new_pose


def transform_pose_from_world_to_cam(cas: CAS, pose: PoseAnnotation) -> PoseAnnotation:
    """Transform pose from world to camera coordinates.

    Uses transform from CAS to convert pose from world frame to camera frame.

    :param cas: CAS containing camera-to-world transform
    :param pose: Pose in world coordinates
    :return: Transformed pose in camera coordinates
    :raises AssertionError: If camera-to-world transform not in CAS
    """
    pose_mat = get_transform_matrix_from_q(pose.rotation, pose.translation)
    world_to_cam_transform = get_world_to_cam_transform_matrix(cas)

    pose_in_cam_mat = np.matmul(world_to_cam_transform, pose_mat)

    new_pose = PoseAnnotation()
    new_pose.rotation = list(get_quaternion_from_transform_matrix(pose_in_cam_mat))
    new_pose.translation = list(get_translation_from_transform_matrix(pose_in_cam_mat))

    return new_pose


def transform_cloud_from_cam_to_world(
    cas: CAS, cloud: o3d.geometry.PointCloud, transform_inplace: bool = False
) -> o3d.geometry.PointCloud:
    """Transform point cloud from camera to world coordinates.

    Uses transform from CAS to convert point cloud from camera frame to world frame.

    :param cas: CAS containing camera-to-world transform
    :param cloud: Point cloud in camera coordinates
    :param transform_inplace: Whether to modify input cloud directly
    :return: Transformed point cloud in world coordinates
    :raises KeyError: If camera-to-world transform not in CAS
    """
    t = get_cam_to_world_transform_matrix(cas)

    if transform_inplace:
        return cloud.transform(t)
    return deepcopy(cloud).transform(t)


def transform_cloud_from_world_to_cam(
    cas: CAS, cloud: o3d.geometry.PointCloud, transform_inplace: bool = False
) -> o3d.geometry.PointCloud:
    """Transform point cloud from world to camera coordinates.

    Uses transform from CAS to convert point cloud from world frame to camera frame.

    :param cas: CAS containing camera-to-world transform
    :param cloud: Point cloud in world coordinates
    :param transform_inplace:  This bool will control if the given 'cloud' is deepcopied before the transformation is done.
    Set this value to True, if it is OK that the given cloud variable is changed.
    Set this to value to False (default), if a deepcopy should performed beforehand.
    :return: Transformed point cloud in camera coordinates
    :raises KeyError: If camera-to-world transform not in CAS
    """
    t = get_world_to_cam_transform_matrix(cas)

    if transform_inplace:
        return cloud.transform(t)
    return deepcopy(cloud).transform(t)


def get_cam_to_world_transform_matrix(cas: CAS) -> npt.NDArray:
    """Get camera-to-world transform matrix from CAS.

    :param cas: CAS containing camera-to-world transform
    :return: 4x4 camera-to-world transform matrix
    :raises KeyError: If camera-to-world transform not in CAS
    """
    if cas.cam_to_world_transform is None:
        raise KeyError(
            "Camera-to-world transform not found in CAS. "
            "Is lookup_viewpoint in camera config set to True? "
            "Or if reading from a database: Has the data been recorded with lookup_viewpoint set to true?"
        )
    return cas.cam_to_world_transform.to_np()


def get_world_to_cam_transform_matrix(cas: CAS) -> npt.NDArray:
    """Get world-to-camera transform matrix from CAS.

    :param cas: CAS containing camera-to-world transform
    :return: 4x4 world-to-camera transform matrix
    :raises KeyError: If camera-to-world transform not in CAS
    """
    world_to_cam_transform = np.linalg.inv(get_cam_to_world_transform_matrix(cas))

    return world_to_cam_transform


def draw_bounding_boxes_from_object_hypotheses(
    img: npt.NDArray,
    object_hypotheses: List[ObjectHypothesis],
    text_function: Callable[[ObjectHypothesis], str],
) -> None:
    """Draw bounding boxes for object hypotheses on the image.

    Draws rectangles and labels for each object hypothesis ROI.

    :param img: Input image to draw on
    :param object_hypotheses: List of object hypotheses
    :param text_function: Function to generate label text from hypothesis
    """

    for oh in object_hypotheses:
        if not isinstance(oh, ObjectHypothesis):
            continue

        oh_roi = oh.roi.roi
        upper_left = (oh_roi.pos.x, oh_roi.pos.y)
        upper_left_text = (oh_roi.pos.x, oh_roi.pos.y - 5)

        font = cv2.FONT_HERSHEY_COMPLEX
        img = cv2.putText(
            img, text_function(oh), upper_left_text, font, 0.5, (0, 0, 255), 1, 2
        )
        img = cv2.rectangle(
            img,
            upper_left,
            (oh_roi.pos.x + oh_roi.width, oh_roi.pos.y + oh_roi.height),
            (0, 0, 255),
            2,
        )


def scale_cam_intrinsics(annotator: BaseAnnotator) -> None:
    """Scale camera intrinsics based on color-to-depth ratio.

    If the color2depth ratio is not 1,1, we will usually scale the color image to the same size
    the depth image has. This also requires an adjustment of the cam intrinsics.

    If color2depth ratio is not 1,1, we'll scale the cam intrinsics in annotator.
    You need to have a self.cam_intrinsics variable!

    :param annotator: Annotator containing camera parameters
    """
    color2depth_ratio = annotator.get_cas().get(CASViews.COLOR2DEPTH_RATIO)
    c2d_ratio_x = color2depth_ratio[0]
    c2d_ratio_y = color2depth_ratio[1]
    if color2depth_ratio != (1, 1):
        annotator.cam_intrinsics = scale_o3d_cam_intrinsics(
            annotator.cam_intrinsics, c2d_ratio_x, c2d_ratio_y
        )


def get_color_image(annotator: BaseAnnotator) -> npt.NDArray:
    """Load and potentially resize the color image from the CAS.

    This is useful, as the resolution of the RGB image might
    differ from the resolution of the depth image. This function
    resizes the color image to the resolution of the depth image.

    This is important later on when we want to create a point cloud
    from the depth image and the color image and crop the point cloud
    to the bounding box of the object hypothesis, which is defined in
    the color image coordinate system.

    :param annotator: Annotator to get image from
    :return: The resized color image.
    """
    img = annotator.get_cas().get_copy(CASViews.COLOR_IMAGE)
    resized_color = None

    if annotator.descriptor.parameters.global_with_depth:
        try:
            resized_color = get_scaled_color_image_for_depth_image(
                annotator.get_cas(), img
            )
        except RuntimeError as e:  # pylint: disable=invalid-name
            annotator.rk_logger.error(
                f"No color to depth ratio set by your camera driver! Can't preprocess: {e}"
            )
    else:
        resized_color = img

    return resized_color


def resize_mask(annotator: BaseAnnotator, mask: npt.NDArray) -> npt.NDArray:
    """Resize mask to match color image resolution.

    The mask is potentially created after the input image has been scaled down
    (see get_scaled_color_image_for_depth_image, we usually scale the RGB down).
    If that's the case, we have to bring it back to the original resolution.
    This method checks some usual error cases and decides if we need to resize at all.

    :param annotator: Annotator containing resolution info
    :param mask: A binary image.
    :return: The scaled version of mask, according to the COLOR2DEPTH_RATIO.
    """
    if not annotator.descriptor.parameters.global_with_depth:
        # nothing to do
        return mask

    color2depth_ratio = annotator.get_cas().get(CASViews.COLOR2DEPTH_RATIO)

    if not color2depth_ratio:
        raise RuntimeError("No Color to Depth Ratio set. Can't continue.")

    if color2depth_ratio == (1, 1):
        return mask
    else:
        c2d_ratio_x = color2depth_ratio[0]
        c2d_ratio_y = color2depth_ratio[1]
        resized_mask = cv2.resize(
            mask,
            None,
            fx=1 / c2d_ratio_x,
            fy=1 / c2d_ratio_y,
            interpolation=cv2.INTER_NEAREST,
        )

    return resized_mask


def generate_source_name(annotator: BaseAnnotator) -> str:
    """Helper to create the standard name for the 'source' field of Annotations.

    :param annotator: The annotator to generate the source name for
    :return: The source name as string
    """
    return type(annotator).__name__

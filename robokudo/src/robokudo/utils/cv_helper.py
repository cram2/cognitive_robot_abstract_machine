"""OpenCV helper utilities for Robokudo.

This module provides helper functions for working with OpenCV images and operations.
It supports:

* Image cropping and ROI handling
* ROS-OpenCV image conversion
* Color space conversions
* Bounding box operations
* Image scaling and resizing
* Drawing utilities
"""

from __future__ import annotations

import cv2
import numpy as np
from typing_extensions import TYPE_CHECKING, Tuple

from robokudo.cas import CASViews
from robokudo.types.cv import ImageROI

if TYPE_CHECKING:
    import cv2.typing as cv2t
    import numpy.typing as npt
    from robokudo.cas import CAS


def crop_image(image: npt.NDArray, xy: tuple, wh: tuple) -> npt.NDArray:
    """Crop region from image using coordinates and dimensions.

    Based on https://stackoverflow.com/a/67799880

    :param image: Input image
    :param xy: Top-left corner coordinates (x,y)
    :param wh: Width and height of crop region (w,h)
    :return: Cropped image region
    """
    x, y = xy
    w, h = wh

    x0, y0 = max(xy[0], 0), max(xy[1], 0)
    x1, y1 = min(x + w, image.shape[1]), min(y + h, image.shape[0])

    crop = image[y0:y1, x0:x1]
    return crop


def crop_image_roi(image: npt.NDArray, roi: ImageROI) -> npt.NDArray:
    """Crop region from image using ROI object.

    :param image: Input image
    :param roi: Region of interest
    :return: Cropped image region
    """
    return crop_image(
        image, (roi.roi.pos.x, roi.roi.pos.y), (roi.roi.width, roi.roi.height)
    )


def get_scaled_color_image_for_depth_image(
    cas: CAS, color_image: npt.NDArray
) -> npt.NDArray:
    """Scale color image to match depth image resolution.

    If the color2depth ratio is not 1,1, we will usually scale the color image to the same size
    the depth image has. This method will get check if a conversion is required and will return the
    correct size. This is usually called before o3d.geometry.RGBDImage.create_from_color_and_depth is called,
    so that depth and color image match.

    :param cas: The CAS where the COLOR2DEPTH_RATIO is stored.
    :param color_image: The color image to be adjusted
    :return: A resized copy of the color image if a resize necessary. Otherwise, the unchanged color_image
    is returned.
    :raises RuntimeError: If color-to-depth ratio not set in CAS
    """
    color2depth_ratio = cas.get(CASViews.COLOR2DEPTH_RATIO)

    if not color2depth_ratio:
        raise RuntimeError("No Color to Depth Ratio set. Can't continue.")

    if color2depth_ratio == (1, 1):
        resized_color = color_image
    else:
        c2d_ratio_x = color2depth_ratio[0]
        c2d_ratio_y = color2depth_ratio[1]
        resized_color = cv2.resize(
            color_image,
            None,
            fx=c2d_ratio_x,
            fy=c2d_ratio_y,
            interpolation=cv2.INTER_NEAREST,
        )

    return resized_color


def get_scale_coordinates(
    color2depth_ratio: Tuple[int, int], coordinates: Tuple[int, int]
) -> Tuple[int, int]:
    """Scale coordinates based on color-to-depth ratio.

    :param color2depth_ratio: Scale factors (x_scale, y_scale)
    :param coordinates: Input coordinates (x, y)
    :return: Scaled coordinates
    :raises RuntimeError: If color-to-depth ratio not provided
    """
    if not color2depth_ratio:
        raise RuntimeError("No Color to Depth Ratio set. Can't continue.")

    if color2depth_ratio == (1, 1):
        return coordinates

    return (
        color2depth_ratio[0] * coordinates[0],
        color2depth_ratio[1] * coordinates[1],
    )


def get_hsv_for_rgb_color(rgb: Tuple[int, int, int]) -> npt.NDArray:
    """Convert RGB color to HSV color space.

    :param rgb: RGB color values (r,g,b)
    :return: HSV color values
    """
    return cv2.cvtColor(np.uint8([[list(rgb)]]), cv2.COLOR_RGB2HSV_FULL)


def get_hsv_for_bgr_color(bgr: Tuple[int, int, int]) -> npt.NDArray:
    """Convert BGR color to HSV color space.

    :param bgr: BGR color values (b,g,r)
    :return: HSV color values
    """
    return cv2.cvtColor(np.uint8([[list(bgr)]]), cv2.COLOR_BGR2HSV_FULL)


def draw_rectangle_around_center(
    binary_image: npt.NDArray, x: int, y: int, width: int, height: int, value: int = 255
) -> npt.NDArray:
    """Draw filled rectangle centered at given point.

    :param binary_image: Binary image to draw on
    :param x: Center x coordinate
    :param y: Center y coordinate
    :param width: Rectangle width
    :param height: Rectangle height
    :param value: Fill value for rectangle
    :return: Image with drawn rectangle
    """
    if width <= 0 or height <= 0:
        return binary_image
    # Consider x and y as the center point of the interest and draw rectangle around it
    center = (int(x), int(y))  # Cast to int for safety
    pt1 = (center[0] - width // 2, center[1] - height // 2)
    pt2 = (center[0] + width // 2, center[1] + height // 2)
    return cv2.rectangle(
        img=binary_image, pt1=pt1, pt2=pt2, color=value, thickness=cv2.FILLED
    )


def clamp_bounding_rect(
    bounding_rect: cv2t.Rect, image_width: int, image_height: int
) -> cv2t.Rect:
    """Clamp bounding rectangle to image boundaries.

    :param bounding_rect: Input bounding rectangle (x,y,w,h)
    :param image_width: Image width
    :param image_height: Image height
    :return: Clamped bounding rectangle
    """
    x1, y1, w, h = bounding_rect
    x2, y2 = x1 + w, y1 + h

    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))

    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    return x1, y1, x2 - x1, y2 - y1


def rect_outside_image(
    bounding_rect: cv2t.Rect, image_width: int, image_height: int
) -> bool:
    """
    Check whether a bounding rectangle is completely outside an image.

    A bounding rectangle is considered outside if it has no overlapping
    area with the image. Rectangles that only touch the image border
    (zero-area intersection) are treated as outside.

    :param bounding_rect: Bounding rectangle to check (x, y, w, h)
    :param image_width: Image width
    :param image_height: Image height
    :return: True if the bounding rectangle lies completely outside
             the image, False otherwise
    """
    x, y, w, h = bounding_rect

    # Rectangle edges
    rect_left = x
    rect_right = x + w
    rect_top = y
    rect_bottom = y + h

    # Image edges
    img_left = 0
    img_right = image_width
    img_top = 0
    img_bottom = image_height

    # Completely outside if one separating axis exists
    return (
        rect_right <= img_left  # left of image
        or rect_left >= img_right  # right of image
        or rect_bottom <= img_top  # above image
        or rect_top >= img_bottom  # below image
    )


def sanity_checks_bounding_rects(
    bounding_rect: cv2t.Rect, image_width: int, image_height: int
) -> bool:
    """Check if bounding rectangle is valid for image dimensions.

    Check basic rules of a boundingRect to reject bad ones. Might happen if projections are done on objects
    that are out-of-view etc.

    :param bounding_rect: Bounding rectangle to check (x,y,w,h)
    :param image_width: Image width
    :param image_height: Image height
    :return: True if rules have been passed, False otherwise
    """
    # Check if ROI starts out of bounds - In that case, we'll skip this Object Hypothesis
    if bounding_rect[0] > image_width or bounding_rect[1] > image_height:
        return False

    # Check if the estimated Width and Height of the BB are even bigger than the image
    if bounding_rect[2] > image_width or bounding_rect[3] > image_height:
        return False

    return True


def adjust_roi(
    image: npt.NDArray, roi: Tuple[int, int, int, int], offset: int
) -> Tuple[int, int, int, int]:
    """Adjusts the ROI by a given offset while respecting image boundaries, keeping the same center point.

    :param image: The image (OpenCV format).
    :param roi: A tuple (x, y, width, height) representing the bounding box.
    :param offset: The amount by which to grow or shrink the ROI (can be negative).
    :return: A tuple (new_x, new_y, new_width, new_height) representing the adjusted ROI.
    """
    x, y, width, height = roi

    # Calculate the center of the current ROI
    center_x = x + width // 2
    center_y = y + height // 2

    # Calculate the new width and height
    new_width = max(1, width + 2 * offset)
    new_height = max(1, height + 2 * offset)

    # Calculate the new top-left position
    new_x = center_x - new_width // 2
    new_y = center_y - new_height // 2

    # Ensure the new ROI fits within the image boundaries
    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0
    if new_x + new_width > image.shape[1]:
        new_x = image.shape[1] - new_width
    if new_y + new_height > image.shape[0]:
        new_y = image.shape[0] - new_height

    # Adjust the width and height if ROI is at the boundary
    if new_x < 0:
        new_width += new_x
        new_x = 0
    if new_y < 0:
        new_height += new_y
        new_y = 0

    return new_x, new_y, new_width, new_height


def adjust_image_roi(image: npt.NDArray, image_roi: ImageROI, offset: int) -> None:
    """Adjusts the ImageROI's ROI by a given offset while respecting image boundaries,
    keeping the same center point. This method uses the adjust_roi function.

    :param image: The image this ROI is relative to. Necessary to respect the boundaries properly.
    :param image_roi: An object of type ImageROI.
    :param offset: The amount by which to grow or shrink the ROI (can be negative).
    :return: None. The ImageROI's ROI is adjusted in place.
    """
    # Extract the current ROI as a tuple (x, y, width, height)
    roi_tuple = (
        image_roi.roi.pos.x,
        image_roi.roi.pos.y,
        image_roi.roi.width,
        image_roi.roi.height,
    )

    # Use the adjust_roi function to get the new ROI
    new_x, new_y, new_width, new_height = adjust_roi(image, roi_tuple, offset)

    # Update the ImageROI object with the new values
    image_roi.roi.pos.x = new_x
    image_roi.roi.pos.y = new_y
    image_roi.roi.width = new_width
    image_roi.roi.height = new_height


def adjust_mask(mask: npt.NDArray, offset: int, fill_value: int = 0) -> npt.NDArray:
    """Adjusts the mask by a given offset. When expanding, the new areas are filled with the specified fill value.

    :param mask: The mask corresponding to the ROI (same dimensions as the ROI).
    :param offset: The amount by which to grow or shrink the mask (can be negative).
    :param fill_value: The value to fill new areas when expanding the mask.
    :return: The adjusted mask.
    """
    height, width = mask.shape

    # Calculate the new dimensions for the mask
    new_width = max(1, width + 2 * offset)
    new_height = max(1, height + 2 * offset)

    if offset > 0:
        # Growing the mask
        new_mask = np.full((new_height, new_width), fill_value, dtype=mask.dtype)
        new_mask[offset : offset + height, offset : offset + width] = mask
    elif offset < 0:
        # Shrinking the mask
        new_mask = mask[-offset : -offset + new_height, -offset : -offset + new_width]
    else:
        new_mask = mask

    return new_mask

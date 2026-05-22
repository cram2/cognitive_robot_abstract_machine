"""
OpenCV camera interface for RGB-only devices in RoboKudo.

This module provides an interface for OpenCV-compatible cameras and video sources
that only provide RGB data without depth information. It supports:

* Live camera streams
* Video file playback
* Single image loading
* Contrast/brightness normalization
* Configurable looping behavior

The module is used for:

* USB webcams
* IP cameras
* Video file playback
* Image sequence processing
"""

from __future__ import annotations
from typing_extensions import Optional, TYPE_CHECKING, Any

from robokudo.io.camera_interface import CameraInterface
from robokudo.cas import CASViews
from robokudo.annotator_parameters import AnnotatorPredefinedParameters

import cv2

if TYPE_CHECKING:
    from robokudo.cas import CAS
    import numpy.typing as npt


TYPE_VIDEO: int = 0
TYPE_CAMERA: int = 1
TYPE_IMAGE: int = 2


class OpenCVCameraWithoutDepthInterface(CameraInterface):
    """
    An openCV camera without depth information.
    An OpenCV-based interface for RGB-only cameras and video sources.

    This class handles various types of RGB input sources through OpenCV:
    * Live camera feeds (USB webcams, IP cameras)
    * Video file playback
    * Single image or image sequence loading

    Supports features like:
    * Automatic input type detection
    * Configurable frame looping
    * Image normalization
    * Static camera calibration
    """

    def __init__(self, camera_config: Any) -> None:
        """Initialize the OpenCV camera interface.

        Opens the video capture device and configures stream parameters based
        on the input type (camera/video/image).

        :raises IOError: If video stream/file cannot be opened
        """
        super().__init__(camera_config)

        # open video stream
        device = camera_config.device

        self.video_capture: cv2.VideoCapture = cv2.VideoCapture(
            device, camera_config.api_preference
        )
        """OpenCV video capture object"""

        if self.video_capture is None or not self.video_capture.isOpened():
            raise IOError("Could not open video stream/file:<{}>.".format(device))

        self.device_driver_flag: int = camera_config.device_driver_flag
        """OpenCV-specific driver flags"""

        max_frames = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

        self.stream_type: int = (
            TYPE_VIDEO
            if max_frames >= 0.0
            else (TYPE_CAMERA if max_frames == -1.0 else TYPE_IMAGE)
        )
        """Type of input (video/camera/image)"""

        self._loop_counter: int = (
            self.stream_type != TYPE_CAMERA and camera_config.loop_mode
        )
        """Number of remaining playback loops. Only works on video/image files."""

        self._backup_color: Optional[npt.NDArray] = None
        """Backup of image for single-image file mode"""

        self.rk_logger.info("OpenCVCameraWithoutDepthInterface initialized")

    def has_new_data(self) -> bool:
        """Check if new frame data is available.

        For video/camera streams, attempts to grab the next frame.
        For single images, checks if backup image is available.

        :return: True if new data is available, False otherwise
        """
        if not self._has_new_data:
            # try to grab next frame
            self._has_new_data = (
                self.video_capture.grab() or self._backup_color is not None
            )

        return self._has_new_data

    def set_data(self, cas: CAS) -> None:
        """Update the Common Analysis Structure with latest frame data.

        This method:
        * Retrieves the next frame from video/camera or backup
        * Handles looping behavior for videos and images
        * Applies optional contrast/brightness normalization
        * Updates the CAS with frame and camera data

        :param cas: Common Analysis Structure to update
        """
        if not self.has_new_data():
            # no new frame available
            return

        # read current frame
        if self._backup_color is None:
            # read ether current video/stream frame or first/only image
            retval, color = self.video_capture.retrieve(flag=self.device_driver_flag)
        else:
            # load last/only image
            color = self._backup_color
            retval = True

        self.rk_logger.debug(
            "loaded frame {} of {}".format(
                self.video_capture.get(cv2.CAP_PROP_POS_FRAMES),
                self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT),
            )
        )

        self._has_new_data = False
        self._backup_color = None

        if not retval:
            # frame not available
            return
            # raise IOError("Could not retrieve frame from stream/file.")

        # handle loop behavior
        frame_index = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)

        if self._loop_counter and frame_index >= self.video_capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        ):
            if self.stream_type == TYPE_IMAGE:
                # single image
                # store first/only image
                self._backup_color = color
            elif self.stream_type == TYPE_VIDEO:
                # video
                # reset frame position to loop the video
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                # camera
                # stream data cannot be looped
                pass

            if self._loop_counter > 0:
                # update number of remaining loops
                self._loop_counter = self._loop_counter - 1

        # stabilise contrast/brightness
        if self.camera_config.normalize_rgb:
            cv2.normalize(color, color, 0, 255, cv2.NORM_MINMAX)

        # update additional (fake) data
        depth = self.camera_config.depth
        cam_info = self.camera_config.cam_info
        cam_intrinsic = self.camera_config.cam_intrinsic
        color2depth_ratio = self.camera_config.color2depth_ratio
        viewpoint_cam_to_world = self.camera_config.viewpoint_cam_to_world

        if self.camera_config.update_global_with_depth_parameter:
            AnnotatorPredefinedParameters.global_with_depth = depth is not None

        cas.set(CASViews.COLOR_IMAGE, color)
        cas.set(CASViews.DEPTH_IMAGE, depth)
        cas.set(CASViews.CAM_INFO, cam_info)
        cas.set(CASViews.CAM_INTRINSIC, cam_intrinsic)
        cas.set(CASViews.COLOR2DEPTH_RATIO, color2depth_ratio)
        # cas.set(CASViews.VIEWPOINT_CAM_TO_WORLD, viewpoint_cam_to_world) # This shouldn't be set if it's not available

"""
This is the camera config for a openCV camera without depth information.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
from typing_extensions import TYPE_CHECKING, Optional, Dict, Any, Tuple, Union, ClassVar

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(slots=True)
class OpenCVCameraConfig(BaseCameraConfig):
    """Configuration class for OpenCV-based cameras without depth information.

    This class defines the configuration parameters for cameras that can be accessed
    through OpenCV's video capture interface. It supports various input sources
    including physical cameras, video files, image sequences, and network streams.

    .. note::
        When using image sequences as input, the first image must have a number
        between 0 and 4, and there cannot be any gaps in the sequence numbering.
    """

    registry_name: ClassVar[str] = "opencv"

    interface_type: str = "OpenCV"

    device: Union[int, str] = 0
    """Input source identifier (camera index, file path, or URL).
    
    device (integer for I/O device or path to image/video file) and flag
    integer:  id of the video capturing device to open
              Use 0 to open default camera using default backend
    string:   path to a video file
              path to image sequence (e.g. 'my/path/img_%02d.jpg').
                  Note: First image needs number between 0 and 4 and the following numbers cannot contain any gaps.
              URL of video/camera stream or image
    """

    api_preference: int = cv2.CAP_ANY
    """Preferred OpenCV capture API backend"""

    device_driver_flag: int = 0
    """Flag argument to use when retrieving/reading the frames"""

    normalize_rgb: bool = True
    """Normalize/stabilise rgb image contrast/brightness"""

    loop_mode: int = -1
    """Loop after iterating over all frames of the video file
    > 0: number of repetitions before to stop
    = 0: never loop
    < 0: infinite loop
    
    ... note::
        Only works with a file as source
    """

    depth: Optional[npt.NDArray] = None
    """Static depth image, which is used as depth image for all rgb images"""

    update_global_with_depth_parameter: bool = True
    """Flag to update global depth parameters"""

    cam_info: Dict[str, Any] = None
    """Camera config as dict"""

    cam_intrinsic = None
    """Camera intrinsic parameters"""

    color2depth_ratio: Tuple[float, float] = (1.0, 1.0)
    """If the resolution of the depth image differs from the color image, we need to define the factor for (x, y)."""

    viewpoint_cam_to_world = None
    """Camera to world transformation"""

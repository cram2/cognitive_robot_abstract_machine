"""Analysis engine for demonstrating USB camera integration.

This module provides an analysis engine that demonstrates how to read and process
video data from various sources using OpenCV, including:
- USB webcams
- Video files
- Image files
- Network video streams

The pipeline implements basic video capture and visualization functionality,
serving as a starting point for more complex video processing applications.

.. note::
    The default configuration uses a USB webcam (device 0), but can be easily
    modified to use other video sources by changing the device parameter.
"""

import robokudo.analysis_engine

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.object_hypothesis_visualizer import ObjectHypothesisVisualizer
import robokudo.descriptors.camera_configs.config_cv_camera_without_depth
import robokudo.io.camera_without_depth_interface

import robokudo.idioms
import robokudo.pipeline


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    """Analysis engine for basic video capture and visualization.

    This class implements a simple pipeline for reading video data from various
    sources and visualizing it. It supports multiple input sources through OpenCV's
    video capture interface.

    Supported input sources:
    - USB webcams (default)
    - Local video files (e.g., .webm, .mp4)
    - Image files (e.g., .jpg, .png)
    - Network video streams (via URL)

    .. note::
        The pipeline is configured to loop once through the input source by default.
        This can be modified using the loop_mode parameter in the camera config.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "demo_usb_camera"

    def implementation(self) -> robokudo.pipeline.Pipeline:
        """Basic demo to read video data from a local usb webcam.

        This method constructs a simple processing pipeline that reads video data
        from a configured source and visualizes it. The source can be configured
        by modifying the device parameter in the camera configuration.

        Example device configurations:
        - USB webcam: device = 0
        - Video file: device = "/path/to/video.webm"
        - Image file: device = "/path/to/image.jpg"
        - Network stream: device = "https://example.com/video.gif"

        :return: The configured pipeline for video processing
        """
        cv_camera_config = (
            robokudo.descriptors.camera_configs.config_cv_camera_without_depth.CameraConfig()
        )
        cv_camera_config.device = 0  # usb webcam
        # cv_camera_config.device = "/home/user_name/Videos/my_video.webm"
        # cv_camera_config.device = "/home/user_name/Pictures/my_image.jpg"
        # cv_camera_config.device = "https://raw.githubusercontent.com/cram2/pycram/refs/heads/dev/doc/images/boxy.gif"

        cv_camera_config.loop_mode = 1
        cv_camera_config.update_global_with_depth_parameter = True
        cv_config = CollectionReaderAnnotator.Descriptor(
            camera_config=cv_camera_config,
            camera_interface=robokudo.io.camera_without_depth_interface.OpenCVCameraWithoutDepthInterface(
                cv_camera_config
            ),
        )

        seq = robokudo.pipeline.Pipeline("RWPipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=cv_config),
                ObjectHypothesisVisualizer(),
            ]
        )
        return seq

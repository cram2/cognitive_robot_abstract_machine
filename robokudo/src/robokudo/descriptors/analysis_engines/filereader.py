"""Analysis engine for processing stored RGB-D data from files.

This module provides an analysis engine that demonstrates how to read and process
RGB-D data stored in the local filesystem. It implements a pipeline for reading
Kinect 360 data and performing basic object detection and segmentation.

The pipeline implements the following functionality:
- Reading RGB-D data from files
- Image preprocessing
- Point cloud cropping with restricted field of view
- Plane detection
- Point cloud cluster extraction

.. note::
    This engine requires the robokudo_test_data repository to be cloned into
    your workspace. The repository contains example data from a Kinect 360
    camera in high-resolution mode.

.. warning::
    Clone https://gitlab.informatik.uni-bremen.de/robokudo/robokudo_test_data
    before running this pipeline.
"""

from robokudo.analysis_engine import AnalysisEngineInterface

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator

from robokudo.annotators.outputs import ClearAnnotatorOutputs
from robokudo.pipeline import Pipeline
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors import CrDescriptorFactory


class AnalysisEngine(AnalysisEngineInterface):
    """Analysis engine for processing stored RGB-D data.

    This class implements a pipeline that reads RGB-D data from files and
    performs object detection and segmentation. It is designed to work with
    high-resolution Kinect 360 data stored in the robokudo_test_data repository.

    The pipeline includes:
    - Collection reader for file-based RGB-D data
    - Image preprocessing
    - Point cloud cropping (restricted FOV)
    - Plane detection
    - Point cloud cluster extraction

    .. note::
        The pipeline is configured to loop through the input data continuously
        and applies specific fixes for Kinect height data.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "filereader_from_tmp"

    def implementation(self) -> Pipeline:
        """Create a pipeline for processing stored RGB-D data.

        This method constructs a processing pipeline that reads RGB-D data from
        files and performs object detection and segmentation. The pipeline is
        configured to read data from the robokudo_test_data repository.

        Configuration details:
        - Data source: robokudo_test_data/data directory
        - Kinect height fix mode enabled
        - Color to depth ratio: 0.5, 0.5 (for high-res mode)
        - Restricted FOV: -0.3m to 0.3m in X axis

        :return: The configured pipeline for RGB-D data processing

        .. warning::
            Make sure the robokudo_test_data repository is cloned and available
            in your ROS workspace before running this pipeline.
        """

        cr_fr_config = CrDescriptorFactory.create_descriptor(
            "file_reader",
            loop=True,
            target_ros_package="robokudo_test_data",
            target_dir="data",
            kinect_height_fix_mode=True,
            color2depth_ratio=(0.5, 0.5),
        )

        # Restrict FOV of pointcloud to robustly get only one object
        pc_crop_config = PointcloudCropAnnotator.Descriptor()
        pc_crop_config.parameters.min_x = -0.3
        pc_crop_config.parameters.max_x = 0.3

        seq = Pipeline("FileReaderPipeline")
        seq.add_children(
            [
                ClearAnnotatorOutputs(),
                CollectionReaderAnnotator(descriptor=cr_fr_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(descriptor=pc_crop_config),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
            ]
        )
        return seq

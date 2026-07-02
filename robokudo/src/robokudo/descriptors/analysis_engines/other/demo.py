"""Analysis engine demonstrating basic tabletop segmentation.

This module provides a basic analysis engine that demonstrates tabletop
segmentation using a Kinect camera. It implements a straightforward pipeline
for processing point cloud data to identify objects on a table surface.

The pipeline implements the following functionality:
- Reading data from a Kinect camera (without transform lookup)
- Image preprocessing
- Point cloud cropping
- Plane detection (table surface)
- Point cloud cluster extraction (objects)

.. note::
    This is a basic demonstration pipeline that can be used as a starting point
    for more complex object detection and segmentation tasks.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.pipeline import Pipeline
from robokudo.idioms import pipeline_init
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors import CrDescriptorFactory


class AnalysisEngine(AnalysisEngineInterface):
    """Analysis engine for basic tabletop segmentation.

    This class implements a simple pipeline for tabletop segmentation using
    a Kinect camera. It processes point cloud data to identify and segment
    objects on a table surface.

    The pipeline includes:
    - Collection reader for Kinect camera data
    - Image preprocessing
    - Point cloud cropping
    - Plane detection
    - Point cloud cluster extraction

    .. note::
        The pipeline uses the Kinect configuration without transform lookup
        for simplicity. For more advanced applications, consider using the
        version with transform lookup enabled.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "demo"

    def implementation(self) -> Pipeline:
        """Create a basic pipeline for tabletop segmentation.

        This method constructs a processing pipeline that performs tabletop
        segmentation using point cloud data from a Kinect camera. The pipeline
        processes the data through several stages to identify objects on a
        table surface.

        The pipeline execution sequence is:
        1. Initialize pipeline
        2. Read frame from Kinect
        3. Preprocess image
        4. Crop point cloud to region of interest
        5. Detect table plane
        6. Extract object clusters

        :return: The configured pipeline for tabletop segmentation

        .. note::
            The pipeline includes commented-out options for adding triggers
            and slow processing simulation, which can be useful for debugging.
        """
        kinect_config = CrDescriptorFactory.create_descriptor("kinect_wo_tf")

        seq = Pipeline("RWPipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                # SlowAnnotator(sleep_in_s=3),
                PointCloudClusterExtractor(),
                # ClusterColorAnnotator(),
                # ClusterColorHistogramAnnotator(),
                # ClusterPositionAnnotator(),
                # Additional annotators (e.g., QueryAnnotator, ActionServerCheck) can be added if needed.
            ]
        )
        return seq

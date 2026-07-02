"""Analysis engine for demonstrating query processing with stored data.

This module provides an analysis engine that demonstrates query-based processing
using stored camera data from MongoDB. It implements a pipeline that combines
query handling with tabletop segmentation and color analysis.

The pipeline implements the following functionality:

* Query handling through action server
* Reading stored camera data from MongoDB
* Image preprocessing and point cloud analysis
* Object segmentation and color analysis
* Query response generation

.. note::
    This demo combines query processing capabilities with stored data analysis,
    making it useful for testing and debugging query-based object analysis
    without requiring live camera input.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import QueryAnnotator, QueryReply

from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from robokudo.behaviours.action_server_checks import ActionServerCheck
from robokudo.descriptors import CrDescriptorFactory


class AnalysisEngine(AnalysisEngineInterface):
    """Analysis engine for query-based processing of stored data.

    This class implements a pipeline that handles queries by processing stored
    camera data. It combines query handling with tabletop segmentation and
    color analysis to demonstrate complex query processing capabilities.

    The pipeline includes:

    * Query handling through action server
    * Collection reader for stored data access
    * Image preprocessing and point cloud analysis
    * Object segmentation with plane detection
    * Color analysis of detected objects
    * Query response generation

    .. note::
        The pipeline uses stored data from MongoDB, allowing repeatable
        testing of query processing without live camera dependencies.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "query_demo_from_storage"

    def implementation(self) -> Pipeline:
        """Create a pipeline for query-based processing of stored data.

        This method constructs a processing pipeline that handles queries by
        analyzing stored camera data. The pipeline performs tabletop segmentation
        and color analysis before generating query responses.

        Pipeline execution sequence:

        1. Initialize pipeline
        2. Wait for query
        3. Read stored camera data
        4. Preprocess image
        5. Crop point cloud
        6. Detect table plane
        7. Extract object clusters
        8. Analyze cluster colors
        9. Generate query response
        10. Check action server status

        :return: The configured pipeline for query processing
        """
        cr_storage_config = CrDescriptorFactory.create_descriptor("mongo")

        seq = Pipeline("StoragePipeline")
        seq.add_children(
            [
                pipeline_init(),
                QueryAnnotator(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterColorAnnotator(),
                QueryReply(),
                ActionServerCheck(),
            ]
        )

        return seq

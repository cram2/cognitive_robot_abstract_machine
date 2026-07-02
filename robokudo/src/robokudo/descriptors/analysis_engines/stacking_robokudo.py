"""Analysis engine for Tracy robot perception pipeline.

This module provides an analysis engine that demonstrates perception capabilities
using the Tracy robot's camera system. It implements a query-based pipeline for
tabletop segmentation and object pose estimation.

The pipeline implements the following functionality:

* Query-based perception control
* Tracy camera data processing
* Point cloud analysis and segmentation
* Object pose estimation using PCA
* Query result generation and response

.. note::
    This engine is specifically designed for the Tracy robot platform and uses
    its camera configuration. The pipeline can be extended with additional
    perception capabilities as needed.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_pose_pca import ClusterPosePCAAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import QueryReply, GenerateQueryResult, QueryAnnotator
from robokudo.descriptors import CrDescriptorFactory
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    """Analysis engine for Tracy robot perception.

    This class implements a pipeline that processes camera data from the Tracy
    robot to perform tabletop segmentation and object pose estimation. It uses
    a query-based approach to control perception tasks.

    The pipeline includes:

    * Query handling for perception control
    * Tracy camera data collection and preprocessing
    * Point cloud analysis and segmentation
    * Object pose estimation using PCA
    * Query result generation and response

    .. note::
        The pipeline uses PCA-based pose estimation by default, but can be
        configured to use bounding box-based estimation by uncommenting the
        relevant annotator.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "demo"

    def implementation(self) -> Pipeline:
        """Create a pipeline for Tracy robot perception.

        This method constructs a processing pipeline that handles perception
        tasks for the Tracy robot. The pipeline processes camera data to
        perform tabletop segmentation and object pose estimation.

        Pipeline execution sequence:

        1. Initialize pipeline
        2. Wait for query
        3. Read Tracy camera data
        4. Preprocess image
        5. Crop point cloud
        6. Detect table plane
        7. Extract object clusters
        8. Estimate object poses (PCA)
        9. Generate and send query response

        :return: The configured pipeline for Tracy perception
        """
        tracy_config = CrDescriptorFactory.create_descriptor("orbbec")

        # pc_crop_config = PointcloudCropAnnotator.Descriptor()
        # pc_crop_config.parameters.

        seq = Pipeline("ContPipeline")
        seq.add_children(
            [
                pipeline_init(),
                QueryAnnotator(),
                CollectionReaderAnnotator(descriptor=tracy_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                # ClusterPoseBBAnnotator(),
                ClusterPosePCAAnnotator(),
                ClusterColorAnnotator(),
                GenerateQueryResult(),
                # QueryReply(),
            ]
        )
        return seq

# test cli command: ros2 action send_goal --feedback \
#     /robokudo/query \
#     robokudo_msgs/action/Query \
#     "{obj: {type: 'block', color: ['blue']}}"
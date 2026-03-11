import py_trees.common

import robokudo.analysis_engine
import robokudo.annotators.query
import robokudo.descriptors.camera_configs.config_kinect_robot
import robokudo.idioms
import robokudo.pipeline
import robokudo.io.camera_interface
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import QueryFeedbackAndCount
from robokudo.behaviours.action_server_checks import ActionServerNoPreemptRequest


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self) -> str:
        return "query_complex"

    def implementation(self) -> robokudo.pipeline.Pipeline:
        """
        Create a pipeline which responds to a query
        """
        kinect_camera_config = (
            robokudo.descriptors.camera_configs.config_kinect_robot.CameraConfig()
        )
        kinect_config = CollectionReaderAnnotator.Descriptor(
            camera_config=kinect_camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(
                kinect_camera_config
            ),
        )

        seq = robokudo.pipeline.Pipeline("RWPipeline")

        task_sequence = py_trees.composites.Sequence(name="TaskSequence", memory=True)
        task_sequence.add_children(
            [
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterColorAnnotator(),
                # AbortGoal(),
                QueryFeedbackAndCount(
                    count_until=50, return_code=py_trees.common.Status.RUNNING
                ),
            ]
        )

        # Combine preemption handling and task execution in a selector
        conditional_selector = py_trees.composites.Selector(
            name="ConditionalSelector", memory=False
        )
        conditional_selector.add_children(
            [
                py_trees.decorators.Inverter(
                    name="Invert Preempt Request", child=ActionServerNoPreemptRequest()
                ),
                task_sequence,  # Run task sequence only if no preemption
            ]
        )

        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                robokudo.annotators.query.QueryAnnotator(),
                conditional_selector,
                robokudo.annotators.query.GenerateQueryResult(),
            ]
        )

        return seq

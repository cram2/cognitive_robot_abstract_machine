from py_trees.common import Status
from py_trees.composites import Selector, Sequence
from py_trees.decorators import Inverter

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import (
    QueryFeedbackAndCount,
    QueryAnnotator,
    GenerateQueryResult,
)
from robokudo.behaviours.action_server_checks import ActionServerNoPreemptRequest
from robokudo.descriptors import CrDescriptorFactory


class AnalysisEngine(AnalysisEngineInterface):
    def name(self) -> str:
        return "query_complex"

    def implementation(self) -> Pipeline:
        """
        Create a pipeline which responds to a query
        """
        kinect_config = CrDescriptorFactory.create_descriptor("kinect")

        seq = Pipeline("RWPipeline")

        task_sequence = Sequence(name="TaskSequence", memory=True)
        task_sequence.add_children(
            [
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterColorAnnotator(),
                # AbortGoal(),
                QueryFeedbackAndCount(count_until=50, return_code=Status.RUNNING),
            ]
        )

        # Combine preemption handling and task execution in a selector
        conditional_selector = Selector(name="ConditionalSelector", memory=False)
        conditional_selector.add_children(
            [
                Inverter(
                    name="Invert Preempt Request", child=ActionServerNoPreemptRequest()
                ),
                task_sequence,  # Run task sequence only if no preemption
            ]
        )

        seq.add_children(
            [
                pipeline_init(),
                QueryAnnotator(),
                conditional_selector,
                GenerateQueryResult(),
            ]
        )

        return seq

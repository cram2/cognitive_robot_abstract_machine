from py_trees.composites import Sequence

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outputs import ClearAnnotatorOutputs
from robokudo.annotators.testing import SlowAnnotator, EmptyAnnotator
from robokudo.descriptors import CrDescriptorFactory
from robokudo.pipeline import Pipeline
from robokudo.tree_components.task_scheduler import IterativeTaskScheduler


class AnalysisEngine(AnalysisEngineInterface):
    def name(self) -> str:
        return "job_scheduler"

    def implementation(self) -> Pipeline:
        """
        Create a pipeline which can schedule different perception tasks in a basic way
        """
        kinect_config = CrDescriptorFactory.create_descriptor("kinect")

        # Annotator definition
        collection_reader = CollectionReaderAnnotator(descriptor=kinect_config)
        image_preprocessor = ImagePreprocessorAnnotator()
        slow1 = SlowAnnotator("Slow1")
        slow2 = SlowAnnotator("Slow2")

        # Subtree construction
        # A subtree is tailored towards solving a specific perception task
        #
        # Please note that the same instance of annotators might be used in multiple trees.
        # The scheduler/planning will take care of maintaining the correct relationships for dynamic trees.
        tree1 = Sequence("Tree1")
        tree2 = Sequence("Tree1")
        tree1.add_children(
            [collection_reader, image_preprocessor, slow1, slow2],
        )
        tree2.add_children(
            [collection_reader, image_preprocessor, slow1],
        )

        # Pipeline creation
        seq = Pipeline("Pipeline")
        # The Job Scheduler needs to be the first child of a Sequence
        task_scheduling = Sequence("Task Scheduling")
        task_scheduling.add_child(IterativeTaskScheduler(tree_list=[tree1, tree2]))

        seq.add_children(
            [
                ClearAnnotatorOutputs(),
                EmptyAnnotator(),
                task_scheduling,
            ]
        )

        return seq

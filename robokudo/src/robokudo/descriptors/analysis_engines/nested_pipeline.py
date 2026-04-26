"""Analysis engine demonstrating nested pipeline functionality.

This module provides an analysis engine that demonstrates how to create and use
nested pipelines within the main processing pipeline. It implements a main pipeline
with a nested belief state pipeline that runs in parallel.

The pipeline implements the following functionality:

* Main pipeline with Kinect camera input
* Nested belief state pipeline with counting annotators
* Conditional execution based on belief state
* Visualization redraw control

.. note::
    This is a demonstration pipeline that shows how to structure complex
    processing flows using nested pipelines and conditional execution.
"""

from robokudo.annotators.outputs import ClearAnnotatorOutputs
from robokudo.descriptors import CrDescriptorFactory
from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.pipeline import Pipeline
from py_trees.behaviours import Count, SuccessEveryN
from py_trees.decorators import Condition
from py_trees.composites import Sequence

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.testing import SlowAnnotator
from robokudo.gui import SetPipelineRedraw


class AnalysisEngine(AnalysisEngineInterface):
    """Analysis engine demonstrating nested pipeline architecture.

    This class implements a pipeline that combines a main processing pipeline
    with a nested belief state pipeline. It demonstrates how to structure
    complex processing flows using nested pipelines and conditional execution.

    The pipeline includes:

    * Main pipeline with camera data processing
    * Nested belief state pipeline with counting annotators
    * Conditional execution control
    * Pipeline redraw functionality

    .. note::
        The nested pipeline uses counting annotators to simulate belief state
        processing, with configurable success/failure conditions.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        :rtype: str
        """
        return "nested_pipeline"

    def implementation(self) -> Pipeline:
        """Create a pipeline with nested belief state processing.

        This method constructs a processing pipeline that includes both a main
        pipeline for camera data processing and a nested pipeline for belief
        state management. The nested pipeline uses counting annotators to
        simulate belief state processing.

        The nested pipeline configuration:

        * Annotator A: Runs for 9 iterations, succeeds on 10th
        * Annotator B: Runs for 9 iterations, succeeds on 10th
        * Success check every 2 iterations

        :return: The configured pipeline with nested belief state processing
        """
        kinect_config = CrDescriptorFactory.create_descriptor("kinect")

        # create second 'pipeline
        second_seq = Sequence(name="BS Pipeline")
        second_seq.add_children(
            [
                Count(
                    name="Annotator A", fail_until=-1, running_until=9, success_until=10
                ),
                Count(
                    name="Annotator B", fail_until=-1, running_until=9, success_until=10
                ),
                SuccessEveryN("Repeat Done?", 2),
            ]
        )
        condition = Condition(second_seq)

        seq = Pipeline("RWPipeline")

        for annotator in [
            ClearAnnotatorOutputs(),
            CollectionReaderAnnotator(descriptor=kinect_config),
            ImagePreprocessorAnnotator("ImagePreprocessor"),
            SlowAnnotator("SlowAnnotator"),
        ]:
            seq.add_child(annotator)

        seq.add_child(SetPipelineRedraw())
        seq.add_child(condition)

        return seq

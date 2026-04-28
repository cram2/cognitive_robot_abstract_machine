"""Analysis engine demonstrating parallel processing capabilities.

This module provides an analysis engine that demonstrates how to implement
parallel processing within a pipeline. It shows how to configure and execute
multiple annotators concurrently using a parallel execution policy.

The pipeline implements the following functionality:

* Kinect camera data input
* Pipeline trigger for controlled execution
* Parallel execution of slow annotators
* Synchronized completion policy

.. note::
    This is a demonstration pipeline that uses simulated slow annotators to
    illustrate parallel processing behavior. The annotators have different
    sleep durations to show asynchronous execution.
"""

from robokudo.analysis_engine import AnalysisEngineInterface

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outputs import ClearAnnotatorOutputs
from robokudo.annotators.pipeline_trigger import PipelineTrigger
from robokudo.annotators.testing import SlowAnnotator
from robokudo.tree_components.better_parallel import Parallel, ParallelPolicy
from robokudo.pipeline import Pipeline
from robokudo.descriptors import CrDescriptorFactory


class AnalysisEngine(AnalysisEngineInterface):
    """Analysis engine demonstrating parallel processing capabilities.

    This class implements a pipeline that demonstrates parallel execution of
    annotators. It uses the BetterParallel component with a synchronized
    completion policy to ensure all parallel tasks complete before proceeding.

    The pipeline includes:

    * Pipeline trigger for controlled execution
    * Camera data collection and preprocessing
    * Fast annotator (1s processing time)
    * Parallel slow annotators (2s and 4s processing times)
    * Final slow annotator (2s processing time)

    .. note::
        The pipeline uses simulated processing times to demonstrate the
        benefits of parallel execution compared to sequential processing.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "parallel"

    def implementation(self) -> Pipeline:
        """Create a pipeline with parallel processing capabilities.

        This method constructs a processing pipeline that demonstrates parallel
        execution of annotators. It configures two slow annotators to run in
        parallel, with different processing times to show asynchronous execution.

        Pipeline execution sequence:

        1. Pipeline trigger (wait for user)
        2. Clear outputs
        3. Read camera data
        4. Preprocess image
        5. Fast annotator (1s)
        6. Parallel annotators (2s and 4s)
        7. Final slow annotator (2s)

        :return: The configured pipeline with parallel processing
        """
        kinect_config = CrDescriptorFactory.create_descriptor("kinect")

        seq = Pipeline("RWPipeline")
        # parallel = py_trees.composites.Parallel()
        parallel = Parallel(policy=ParallelPolicy.SuccessOnAll(synchronise=True))
        parallel.add_children(
            [
                # py_trees.behaviours.Count(name="Annotator A", fail_until=-1, running_until=30, success_until=1000),
                # py_trees.behaviours.Count(name="Annotator B", fail_until=-1, running_until=15, success_until=1000)
                SlowAnnotator("Slow1Annotator", sleep_in_s=4),
                SlowAnnotator("Slow2Annotator", sleep_in_s=2),
            ]
        )

        for annotator in [
            PipelineTrigger(),
            ClearAnnotatorOutputs(),
            CollectionReaderAnnotator(descriptor=kinect_config),
            ImagePreprocessorAnnotator("ImagePreprocessor"),
            SlowAnnotator("FastAnnotator", sleep_in_s=1),
            parallel,
            SlowAnnotator("Slow3Annotator", sleep_in_s=2),
        ]:
            seq.add_child(annotator)

        return seq

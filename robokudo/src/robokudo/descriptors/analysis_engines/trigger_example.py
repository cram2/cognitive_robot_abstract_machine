"""Analysis engine demonstrating pipeline trigger functionality.

This module provides an analysis engine that demonstrates how to use pipeline
triggers to control the execution flow. It implements a pipeline that waits for
user input (keypress) before processing each frame from a Kinect camera.

The pipeline implements the following functionality:
- Pipeline trigger for user-controlled execution
- Reading data from a Kinect camera
- Image preprocessing
- Simulated slow processing (for demonstration)

.. note::
    This example is particularly useful for debugging and step-by-step analysis
    of pipeline behavior, as it allows manual control over frame processing.
"""

import robokudo.analysis_engine

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.pipeline_trigger import PipelineTrigger
from robokudo.annotators.testing import SlowAnnotator

import robokudo.descriptors.camera_configs.config_kinect_robot

import robokudo.io.camera_interface
import robokudo.idioms
import robokudo.pipeline


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    """Analysis engine with pipeline trigger functionality.

    This class implements a pipeline that demonstrates the use of pipeline
    triggers for controlled execution. The pipeline waits for user input
    before processing each frame, making it useful for debugging and
    step-by-step analysis.

    The pipeline includes:
    - Pipeline trigger for user control
    - Collection reader for Kinect camera data
    - Image preprocessing
    - Simulated slow processing

    .. note::
        The pipeline uses a SlowAnnotator to simulate time-consuming processing.
        This helps demonstrate the effect of the trigger mechanism on pipeline
        execution.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "trigger_example"

    def implementation(self) -> robokudo.pipeline.Pipeline:
        """Create a pipeline with trigger-controlled execution.

        This method constructs a processing pipeline that includes a trigger
        mechanism. The pipeline will pause and wait for user input (keypress)
        before processing each frame from the Kinect camera.

        The pipeline execution sequence is:
        1. Wait for trigger (keypress)
        2. Initialize pipeline
        3. Read frame from Kinect
        4. Preprocess image
        5. Simulate slow processing
        6. Return to step 1

        :return: The configured pipeline with trigger mechanism
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

        for annotator in [
            PipelineTrigger(),
            robokudo.idioms.pipeline_init(),
            CollectionReaderAnnotator(descriptor=kinect_config),
            ImagePreprocessorAnnotator("ImagePreprocessor"),
            SlowAnnotator("SlowAnnotator"),
        ]:
            seq.add_child(annotator)

        return seq

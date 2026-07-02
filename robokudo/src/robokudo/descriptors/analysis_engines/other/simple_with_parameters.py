"""Analysis engine demonstrating parameter configuration.

This module provides an analysis engine that demonstrates how to configure
and use parameters in a pipeline. It implements a simple pipeline that uses
customized parameters for image preprocessing.

The pipeline implements the following functionality:

* Kinect camera data input
* Image preprocessing with custom depth truncation
* Simulated slow processing

.. note::
    This example shows how to configure annotator parameters through descriptors,
    which is essential for customizing pipeline behavior without modifying
    annotator code.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.testing import SlowAnnotator
from robokudo.descriptors import CrDescriptorFactory
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    """Analysis engine demonstrating parameter configuration.

    This class implements a pipeline that shows how to configure annotator
    parameters using descriptors. It uses a custom depth truncation value
    for image preprocessing to demonstrate parameter customization.

    The pipeline includes:

    * Collection reader for Kinect camera data
    * Image preprocessor with custom depth truncation
    * Slow annotator for processing simulation

    .. note::
        The image preprocessor is configured with a depth truncation value
        of 4.5 meters, which can be adjusted to suit different environments
        and requirements.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "simple_with_parameters"

    def implementation(self) -> Pipeline:
        """Create a pipeline with custom parameter configuration.

        This method constructs a processing pipeline that demonstrates how to
        configure annotator parameters. It specifically shows how to set a
        custom depth truncation value for the image preprocessor.

        Pipeline configuration:

        * Kinect camera interface setup
        * Image preprocessor with depth_trunc = 4.5m
        * Slow annotator for processing simulation

        :return: The configured pipeline with custom parameters
        """
        kinect_config = CrDescriptorFactory.create_descriptor("kinect")

        image_preprocessor_config = ImagePreprocessorAnnotator.Descriptor()
        image_preprocessor_config.parameters.depth_trunc = 4.5

        seq = Pipeline("RWPipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator(
                    "ImagePreprocessor", descriptor=image_preprocessor_config
                ),
                SlowAnnotator("SlowAnnotator"),
            ]
        )

        return seq

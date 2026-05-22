import robokudo.analysis_engine
import robokudo.behaviours.clear_errors
import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.idioms
import robokudo.io.storage_reader_interface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.location import LocationAnnotator
from robokudo.annotators.object_hypothesis_visualizer import ObjectHypothesisVisualizer
from robokudo.annotators.static_object_detector import (
    StaticObjectDetectorAnnotator,
    StaticObjectMode,
)
from robokudo.annotators.world_descriptor_bootstrap import (
    WorldDescriptorBootstrapAnnotator,
)
from robokudo.descriptors import CrDescriptorFactory


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self):
        return "static_detector_world_descriptor_from_storage"

    def implementation(self):
        cr_storage_config = CrDescriptorFactory.create_descriptor("mongo")

        bootstrap = WorldDescriptorBootstrapAnnotator.Descriptor()
        bootstrap.parameters.world_descriptor_name = "world_iai_kitchen20"

        sod = StaticObjectDetectorAnnotator.Descriptor()
        sod.parameters.mode = StaticObjectMode.WORLD_DESCRIPTOR
        sod.parameters.class_names = ["cereal", "milk"]
        sod.parameters.create_bounding_box_annotation = True
        sod.parameters.create_pose_annotation = True

        oh_vis = ObjectHypothesisVisualizer.Descriptor()
        oh_vis.parameters.visualize_full_cloud = True

        seq = robokudo.pipeline.Pipeline("StoragePipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                WorldDescriptorBootstrapAnnotator(descriptor=bootstrap),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                StaticObjectDetectorAnnotator(descriptor=sod),
                ObjectHypothesisVisualizer(descriptor=oh_vis),
            ]
        )
        return seq

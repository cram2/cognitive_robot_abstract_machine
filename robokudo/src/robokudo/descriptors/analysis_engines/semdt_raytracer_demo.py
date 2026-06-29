"""Analysis engine for simulated RGB-D input from SemDT RayTracer.

This pipeline mirrors the standard tabletop segmentation flow but uses the
`semdt_raytracer` camera descriptor, which renders color/depth images from a
world descriptor instead of reading a physical camera stream.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    def name(self) -> str:
        return "semdt_raytracer_demo"

    def implementation(self) -> Pipeline:
        raytracer_config = CollectionReaderDescriptorFactory.create_descriptor(
            "semdt_raytracer",
            world_descriptor_name="world_semdt_raytracer_tabletop",
        )
        plane_desc = PlaneAnnotator.Descriptor()
        plane_desc.parameters.distance_threshold = 0.01

        seq = Pipeline("SemDTRayTracerPipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=raytracer_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(descriptor=plane_desc),
                PointCloudClusterExtractor(),
                ClusterColorAnnotator(),
                ClusterPoseBBAnnotator(),
            ]
        )
        return seq

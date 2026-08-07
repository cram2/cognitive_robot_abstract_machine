"""
Analysis engine answering perception queries for the Stretch robot.

Localizes objects standing on the dominant plane within the apartment demo's second
shelf layer, in view of the Stretch's RealSense, and reports their poses in response to
a :class:`~robokudo_msgs.action.Query`, so a plan can correct an object's pose before
grasping it.

.. note::
    The pipeline localizes but does not recognize: it attaches no
    :class:`~robokudo.types.annotation.Classification`, so every reported object designator
    has an empty ``type`` and the caller decides what it asked for. Adding a classifying
    annotator (for example
    :class:`~robokudo.annotators.clip_annotator.ClipAnnotator` or
    :class:`~robokudo.annotators.simple_yolo_annotator.SimpleYoloAnnotator`) before
    :class:`~robokudo.annotators.query.GenerateQueryResult` fills that field in without any
    other change.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_cluster_extractor import ImageClusterExtractor
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import QueryAnnotator, GenerateQueryResult
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline

CAMERA_CONFIG_NAME = "realsense"
"""
Camera configuration this engine reads from.

The Stretch carries a RealSense D435i publishing on the stock ``realsense2_camera`` topics,
and its colour frame is named ``camera_color_optical_frame``, so the shared RealSense config
applies unchanged. Pass overrides to
:meth:`~robokudo.descriptors.factories.cr_descriptor_factory.CollectionReaderDescriptorFactory.create_descriptor`
if a particular robot publishes elsewhere.
"""

TARGET_SHELF_LAYER_MIN_WORLD_Z = 0.4565
"""
Lower world-frame height bound of the crop, in metres.

The apartment demo's shelf has layers at world heights 0.283m, 0.63m, 1.265m and 1.613m
(see ``experiments.real_stretch_apartment_demo.demo``); this engine targets the second
layer at 0.63m. The bound is the midpoint between that layer and the one below it, so
the crop keeps whatever sits on the target layer while excluding its neighbours.
"""

TARGET_SHELF_LAYER_MAX_WORLD_Z = 0.9475
"""
Upper world-frame height bound of the crop, in metres.

The midpoint between the target layer (0.63m) and the one above it (1.265m).
"""

TARGET_SHELF_MIN_WORLD_X = 0.4
TARGET_SHELF_MAX_WORLD_X = 1.35
TARGET_SHELF_MIN_WORLD_Y = -0.35
TARGET_SHELF_MAX_WORLD_Y = 0.05
"""
World-frame lateral bounds of the crop, in metres.

The shelf (``experiments.real_stretch_apartment_demo.demo``) is a 0.305m x 0.85m
footprint centred at world (0.88, -0.17) with a -90 degree yaw, which rotates its outer
footprint to world X in [0.455, 1.305] and world Y in [-0.3225, -0.0175]. These bounds
add roughly 5cm of margin on every side of that footprint, since the height bound alone
left the shelf's own case (side walls, the layer's front lip) and, without a lateral
bound at all, the rest of the room in view as further untyped candidates.
"""


class AnalysisEngine(AnalysisEngineInterface):
    """
    Query-driven tabletop localization for the Stretch.
    """

    def name(self) -> str:
        """
        Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "stretch_demo"

    def implementation(self) -> Pipeline:
        """
        Build the pipeline that answers a query with the poses of the objects in view.

        The pipeline waits for a query, reads a frame, isolates the dominant plane,
        treats what stands on it as objects, estimates a pose per object from its
        bounding box, and replies.

        :return: The configured pipeline for Stretch perception
        """
        camera_descriptor = CollectionReaderDescriptorFactory.create_descriptor(
            CAMERA_CONFIG_NAME
        )

        # Isolate the target shelf layer before clustering: without this the crop keeps
        # its class defaults (an effectively unbounded height range), so the camera's
        # wide RealSense field of view also picks up the other shelf layers, and the
        # query answers with an untyped candidate per layer instead of just the one
        # this engine is meant to find.
        crop_descriptor = PointcloudCropAnnotator.Descriptor()
        crop_descriptor.parameters.relative_to_world = True
        crop_descriptor.parameters.min_x = TARGET_SHELF_MIN_WORLD_X
        crop_descriptor.parameters.max_x = TARGET_SHELF_MAX_WORLD_X
        crop_descriptor.parameters.min_y = TARGET_SHELF_MIN_WORLD_Y
        crop_descriptor.parameters.max_y = TARGET_SHELF_MAX_WORLD_Y
        crop_descriptor.parameters.min_z = TARGET_SHELF_LAYER_MIN_WORLD_Z
        crop_descriptor.parameters.max_z = TARGET_SHELF_LAYER_MAX_WORLD_Z

        # The target object (a cereal box) is glossy enough that the RealSense returns no
        # depth on its face, and depth-based clustering has no way to bound its search to
        # just the object (it searches the whole detected shelf plane). Extracting by
        # color instead sidesteps both: the ROI comes from the RGB contour, not from
        # depth, so a hole in the box's depth just means fewer 3D points within an
        # already-correctly-shaped region. Hardcoded to red for now, matching the one
        # object this pipeline currently targets.
        cluster_descriptor = ImageClusterExtractor.Descriptor()
        cluster_descriptor.parameters.hsv_min = (
            cluster_descriptor.parameters.color_name_to_hsv_range["red"]["hsv_min"]
        )
        cluster_descriptor.parameters.hsv_max = (
            cluster_descriptor.parameters.color_name_to_hsv_range["red"]["hsv_max"]
        )
        # The depth hole leaves few valid 3D points under the color mask. Lowered to
        # ClusterPoseBBAnnotator's own floor (it needs more than 10 points to compute a
        # pose, cluster_pose_bb.py:161-165) rather than the class default of 62, so a
        # sparse-but-real detection reaches pose estimation instead of being dropped here
        # first. Outlier removal is skipped too: pruning an already-sparse, hole-riddled
        # cloud for statistical outliers risks stripping the real points further.
        cluster_descriptor.parameters.min_points_threshold = 11
        cluster_descriptor.parameters.outlier_removal = True

        # ..note:: PointcloudCropAnnotator crops CASViews.CLOUD, but
        #     ImageClusterExtractor reads CASViews.COLOR_IMAGE/DEPTH_IMAGE directly, so
        #     the crop no longer bounds what ImageClusterExtractor searches. Left in
        #     place for now to see how color-based extraction behaves before deciding
        #     whether/how to re-bound its search area.
        pipeline = Pipeline("StretchPipeline")
        pipeline.add_children(
            [
                pipeline_init(),
                QueryAnnotator(),
                CollectionReaderAnnotator(descriptor=camera_descriptor),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(descriptor=crop_descriptor),
                ImageClusterExtractor(descriptor=cluster_descriptor),
                ClusterPoseBBAnnotator(),
                # Left unconfigured so that filtering by query stays off: it compares the
                # requested type against Classification annotations, which this pipeline
                # does not produce, so filtering a typed query would discard every object.
                # ..warning:: `GenerateQueryResult.Descriptor.parameters` is a class
                #     attribute shared by every instance, so overriding it here would
                #     change the setting for other pipelines in the same process too.
                GenerateQueryResult(),
            ]
        )
        return pipeline

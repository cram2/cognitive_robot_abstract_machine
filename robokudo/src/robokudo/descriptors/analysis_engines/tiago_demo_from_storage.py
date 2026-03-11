"""Analysis engine for TIAGo robot perception using stored data.

This module provides an analysis engine that demonstrates TIAGo robot perception
capabilities using stored camera data. It implements a pipeline that combines
preprocessing, object detection, and parallel feature analysis.

The pipeline implements the following functionality:

* Reading stored TIAGo camera data from MongoDB
* Sequential preprocessing stage
* Parallel object analysis stage
* Color analysis
* PCA-based pose estimation
* Optional query handling (commented out)

.. note::
    This engine processes stored data from the TIAGo robot's camera system,
    making it useful for testing and developing perception algorithms without
    requiring direct robot access.
"""

import py_trees

import robokudo.analysis_engine

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.cluster_pose_pca import ClusterPosePCAAnnotator
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.camera_viewpoint_visualizer import CameraViewpointVisualizer


import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.io.storage_reader_interface
import robokudo.tree_components.better_parallel
import robokudo.idioms
import robokudo.pipeline

from robokudo.annotators.query import QueryReply, GenerateQueryResult


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    """Analysis engine for TIAGo robot perception with stored data.

    This class implements a pipeline that processes stored TIAGo camera data
    using a combination of sequential and parallel processing stages. It
    demonstrates advanced pipeline structuring with parallel feature analysis.

    The pipeline includes:

    * Sequential preprocessing stage
      * Collection reader for stored data
      * Image preprocessing
    * Sequential object detection stage
      * Crop point cloud
      * Detect plane
      * Extract clusters
    * Parallel feature analysis stage
      * Color analysis
      * PCA-based pose estimation

    .. note::
        The pipeline includes commented-out query handling components that
        can be enabled for interactive operation.
    """

    def name(self) -> str:
        """Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "tiago_demo_from_storage"

    def implementation(self) -> robokudo.pipeline.Pipeline:
        """Create a pipeline for TIAGo perception with stored data.

        This method constructs a processing pipeline that combines sequential
        and parallel stages for processing stored TIAGo camera data. The
        pipeline is structured to optimize processing by running compatible
        analyses in parallel.

        Pipeline structure:

        1. Preprocessing sequence
           * Read stored data
           * Preprocess images
        2. Object detection sequence
           * Crop point cloud
           * Detect plane
           * Extract clusters
        3. Parallel analysis
           * Color analysis
           * PCA pose estimation

        :return: The configured pipeline with parallel processing
        """
        cr_storage_camera_config = (
            robokudo.descriptors.camera_configs.config_mongodb_playback.CameraConfig()
        )
        cr_storage_config = CollectionReaderAnnotator.Descriptor(
            camera_config=cr_storage_camera_config,
            camera_interface=robokudo.io.storage_reader_interface.StorageReaderInterface(
                cr_storage_camera_config
            ),
        )

        pre = py_trees.composites.Sequence("Preprocessing")
        pre.add_children(
            [
                CollectionReaderAnnotator(descriptor=cr_storage_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
            ]
        )

        parallel = robokudo.tree_components.better_parallel.Parallel(
            policy=robokudo.tree_components.better_parallel.ParallelPolicy.SuccessOnAll(
                synchronise=True
            )
        )
        parallel.add_children(
            [
                ClusterColorAnnotator(),
                ClusterPosePCAAnnotator(),
            ]
        )
        annotators = py_trees.composites.Sequence("Annotators")
        annotators.add_children(
            [
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                parallel,
            ]
        )

        seq = robokudo.pipeline.Pipeline("ContPipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                # robokudo.annotators.query.QueryAnnotator(),
                pre,
                annotators,
                # GenerateQueryResult(),
                # QueryReply(),
            ]
        )
        return seq

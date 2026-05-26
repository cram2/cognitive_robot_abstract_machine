"""Pose refinement using iterative closest point (ICP).

This module provides an annotator for:

* Refining object pose estimates using ICP
* Matching against reference point cloud models
* Supporting multiple initial pose variants
* Evaluating registration quality

The module uses:

* Open3D ICP implementation
* Point-to-point registration
* Multiple pose hypotheses
* PLY model loading
* Visualization tools

.. warning::
   Currently limited to refining against a single pre-loaded model.
"""

import copy
import math
from pathlib import Path
from timeit import default_timer

import numpy as np
import open3d as o3d
from py_trees.common import Status
from typing_extensions import Optional

from robokudo.annotators.core import ThreadedAnnotator, BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import PoseAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.file_loader import FileLoader
from robokudo.utils.transform import (
    get_transform_matrix_from_q,
    get_transform_matrix_for_rotation_around_axis,
    get_translation_from_transform_matrix,
    get_quaternion_from_transform_matrix,
)


class ICPPoseRefinementAnnotator(ThreadedAnnotator):
    """Pose refinement using iterative closest point (ICP).

    This annotator:

    * Refines object pose estimates using ICP
    * Matches against reference point cloud models
    * Evaluates multiple initial pose variants
    * Selects best registration result
    * Creates refined pose annotations

    .. note::
       Point cloud models can be generated from meshes using CloudCompare's
       "Sample points on a mesh" feature.

    .. warning::
       Currently limited to refining against a single pre-loaded model.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for ICP refinement."""

        class Parameters:
            """Parameters for configuring model loading."""

            def __init__(self) -> None:
                self.ros_pkg_path: Optional[str] = None
                """If set, use use data_path as a relative path to self.ros_pkg_path"""

                self.data_path: Optional[str] = None
                """Relative Path to the folder containing the models"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "ICPPoseRefinementAnnotator",
        descriptor: "ICPPoseRefinementAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the ICP pose refiner.

        :param name: Name of this annotator instance, defaults to "ICPPoseRefinementAnnotator"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        # Data structure for available classes -> PLY files
        self.class_name_to_ply_model = dict()

        # Load PLY PointClouds (not Meshes!) for all objects
        ply_path = self.get_model_path("JeroenCup")
        if ply_path is None or not ply_path.exists():
            self.rk_logger.error("Couldn't lookup model path. Deactivating Annotator.")
            self.annotator_deactivated = True
        else:
            self.annotator_deactivated = False
            ply_model = o3d.io.read_point_cloud(str(ply_path))
            self.class_name_to_ply_model["JeroenCup"] = ply_model

    # TODO Rewrite this so that we can load multiple files from the directory at once
    def get_model_path(self, class_name: str = "") -> Optional[Path]:
        """Get path to PLY model file.

        Resolves model path using:

        * ROS package path if provided
        * Direct data path otherwise

        :param class_name: Name of object class to load model for
        :return: Path to PLY file or None if not found
        """
        if self.descriptor.parameters.ros_pkg_path is not None:
            # if using ros_pkg_path, data_path has to be set as well!
            assert self.descriptor.parameters.data_path is not None

            data_folder_path = FileLoader.get_path_to_file_in_ros_package(
                ros_pkg_name=self.descriptor.parameters.ros_pkg_path,
                relative_path=self.descriptor.parameters.data_path,
            )
        elif self.descriptor.parameters.data_path is not None:
            data_folder_path = Path(self.descriptor.parameters.data_path)

            if not data_folder_path.exists():
                self.rk_logger.error(f"'{data_folder_path}' is not existing.")
                return None

            if not data_folder_path.is_dir():
                self.rk_logger.error(f"No directory found at '{data_folder_path}.'")
                return None

        else:
            self.rk_logger.error(
                f"Can't look up necessary input data. Please check Parameters of this Annotator.'"
            )
            return None

        ply_path = data_folder_path.joinpath(f"{class_name}.ply")
        if not ply_path.exists():
            self.rk_logger.error(f"No ply model found at '{ply_path}'")
        return ply_path

    def compute(self) -> Status:
        """Process object hypotheses and refine poses.

        The method:

        * Loads point cloud and object hypotheses
        * For each hypothesis with pose:
          * Creates initial pose variants
          * Performs ICP registration for each variant
          * Selects best registration result
          * Creates refined pose annotation
          * Generates visualization markers

        :return: SUCCESS after processing, even if annotator deactivated
        """
        start_timer = default_timer()

        if self.annotator_deactivated:
            error_message = (
                f"{type(self).__name__} couldn't find model files and is deactivated."
            )
            self.rk_logger.warn(error_message)
            self.feedback_message = error_message
            return Status.SUCCESS  # Don't fail the whole pipeline

        cloud = self.get_cas().get(CASViews.CLOUD)
        annotations = self.get_cas().annotations
        geometries_to_visualize = []

        # ICP Parameters
        thresh = 0.02
        icp_transform_init = np.eye(4)

        # Iterate over everything that is a Object hypothesis and calculate the centroid
        for annotation in annotations:
            if not isinstance(annotation, ObjectHypothesis):
                continue
            # Each hypothesis must have valid points before we can proceed
            if annotation.points is None:
                continue
            if not isinstance(annotation.points, o3d.geometry.PointCloud):
                continue
            if len(annotation.points.points) == 0:
                continue

            object_hypothesis = annotation

            cluster_cloud = object_hypothesis.points

            # Check if this Object has a Pose Annotation that we can refine
            # Fetch the pose annotations first, because we can't alter the object_hypothesis.annotations list
            # while iterating!
            pose_annotations = [
                x
                for x in object_hypothesis.annotations
                if isinstance(x, PoseAnnotation)
            ]

            for pose_annotation in pose_annotations:
                # Transform PLY Model based on the PoseAnnotation for that object
                pose_t = get_transform_matrix_from_q(
                    np.asarray(pose_annotation.rotation),
                    np.asarray(pose_annotation.translation),
                )

                # Pose variants are a tuple with the actual pose to evaluate and a textual description for easier readability
                initial_pose_variants = [(pose_t, "Initial Pose")]

                # Turn model upside-down and add this as a pose variant. The z-axis might be turned upside-down
                upside_down_transform = get_transform_matrix_for_rotation_around_axis(
                    math.pi, (1, 0, 0)
                )
                initial_pose_variants.append(
                    (pose_t @ upside_down_transform, "Upside down")
                )

                # Add Initial Ply model for visualization
                ply_model = copy.deepcopy(self.class_name_to_ply_model["JeroenCup"])
                ply_model.paint_uniform_color([1, 0, 0])
                ply_model.transform(pose_t)
                geometries_to_visualize.append(
                    {"name": "Initial Pose Estimate", "geometry": ply_model}
                )

                # evaluation = o3d.pipelines.registration.evaluate_registration(
                #     ply_model, cluster_cloud, thresh, icp_transform_init)
                # self.rk_logger.info(evaluation)

                # We'll now go through our pose variants, register them after each other and then check which has the
                # highest fitness
                pose_variant_registration_results = []
                for pose_variant, description in initial_pose_variants:
                    ply_model = copy.deepcopy(self.class_name_to_ply_model["JeroenCup"])
                    ply_model.transform(pose_variant)
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        ply_model,
                        cluster_cloud,
                        thresh,
                        icp_transform_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        # PointToPlane has sometimes good results, but some of them are really off
                        # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(
                            max_iteration=100,
                        ),
                        # relative_fitness=1.000000e-06, relative_rmse=1.000000e-08 )
                    )
                    pose_variant_registration_results.append(
                        (reg_p2p, pose_variant, description)
                    )
                    # self.rk_logger.info(reg_p2p)

                best_registration_result = None
                best_registration_result_idx = 0
                i = 0
                for (
                    registration_result,
                    pose_variant,
                    description,
                ) in pose_variant_registration_results:
                    # Check if the registration result of this iteration is atleast 10% better than the previous best
                    # Or set it to the first one that we encounter
                    if (
                        best_registration_result is None
                        or registration_result.inlier_rmse * 1.1
                        < best_registration_result[0].inlier_rmse
                    ):
                        best_registration_result = (registration_result, pose_variant)
                        best_registration_result_idx = i
                    i += 1

                if best_registration_result is None:
                    continue  # Abort here if we didn't find a good fit

                self.rk_logger.info(
                    f"Selected pose {best_registration_result_idx} as the best one"
                )
                # Visualize ICP Result
                refined_ply_model = copy.deepcopy(
                    self.class_name_to_ply_model["JeroenCup"]
                )
                refined_ply_model.paint_uniform_color([0, 1, 0])
                # Transform PLY Model based on the Pose Annotation AND the ICP result
                refined_transformation = (
                    best_registration_result[0].transformation
                    @ best_registration_result[1]
                )
                refined_ply_model.transform(refined_transformation)

                geometries_to_visualize.append(
                    {"name": "Refined Pose Estimate", "geometry": refined_ply_model}
                )

                refined_pose_annotation = PoseAnnotation()
                refined_pose_annotation.translation = list(
                    get_translation_from_transform_matrix(refined_transformation)
                )
                refined_pose_annotation.rotation = list(
                    get_quaternion_from_transform_matrix(refined_transformation)
                )
                refined_pose_annotation.source = type(self).__name__

                object_hypothesis.annotations.append(refined_pose_annotation)

        # Visualization
        vis_geometries = [
            {"name": "Cloud", "geometry": cloud},
        ]
        vis_geometries.extend(geometries_to_visualize)

        self.get_annotator_output_struct().set_geometries(vis_geometries)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

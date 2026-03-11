"""Object hypothesis visualization for RoboKudo.

This module provides an annotator for visualizing object hypotheses in both
2D (image overlays) and 3D (point clouds) representations.
"""

from __future__ import annotations
import copy
from timeit import default_timer

import cv2
import open3d as o3d
import py_trees
from typing_extensions import TYPE_CHECKING, Tuple

import robokudo
import robokudo.annotators.core
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.annotator_helper
import robokudo.utils.error_handling
import robokudo.utils.o3d_helper
import robokudo.utils.type_conversion
from robokudo.cas import CASViews
from robokudo.utils.file_loader import FileLoader
from robokudo_msgs.action import Query

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class ObjectHypothesisVisualizer(robokudo.annotators.core.BaseAnnotator):
    """Annotator for visualizing object hypotheses in the CAS.

    This annotator creates visualizations of detected objects by:

    * Drawing bounding boxes and labels on the color image
    * Displaying associated point clouds in 3D
    * Optionally filtering objects based on query type
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for object hypothesis visualization."""

        class Parameters:
            """Parameters for configuring visualization behavior."""

            def __init__(self) -> None:
                self.query_aware: bool = True
                """If set to true, only visualize an Object that matches the 'object.type' from the Query."""

                self.visualize_full_cloud: bool = False
                """If set to true, the scene cloud will be shown and the individual objects will be colored."""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "ObjectHypothesisVisualizer",
        descriptor: "ObjectHypothesisVisualizer.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the object hypothesis visualizer.

        :param name: Name of the annotator instance, defaults to "ObjectHypothesisVisualizer"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        self._mesh_cache = {}
        """"""

    def draw_text_middle(
        self,
        image: npt.NDArray[np.uint8],
        text: str,
        color: Tuple[int, int, int] = (0, 0, 255),
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 1.0,
        thickness: int = 2,
    ):
        """Draw text in the middle of an image.

        :param image: Image to draw on
        :param text: Text to draw
        :param color: BGR color tuple
        :param font: OpenCV font type, defaults to cv2.FONT_HERSHEY_SIMPLEX
        :param font_scale: Font scale factor, defaults to 1
        :param thickness: Line thickness, defaults to 2
        """
        # Get the size of the text
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Calculate the position to place the text in the middle
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2

        # Draw the text on the image
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self) -> py_trees.common.Status:
        """Update the visualization with current object hypotheses.

        Creates visualizations containing:

        * Color image with bounding boxes and labels for each object
        * 3D point clouds, pose and bounding box annotations for each object
        * Optional filtering based on query type if query_aware is True

        :return: SUCCESS after creating visualizations
        """
        start_timer = default_timer()

        visualization_img = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        visualized_geometries = []

        if self.descriptor.parameters.query_aware:
            query = None  # type: Query.Goal | None
            if self.get_cas().contains(CASViews.QUERY):
                query = self.get_cas().get(CASViews.QUERY)

        object_hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.ObjectHypothesis
        )
        if len(object_hypotheses) == 0:
            self.draw_text_middle(visualization_img, "No Object Hypotheses")
        else:

            def get_box_text(oh):
                max_conf = -1
                best_classification = None

                for oh_anno in oh.annotations:
                    if isinstance(oh_anno, robokudo.types.annotation.Classification):
                        if oh_anno.confidence > max_conf:
                            max_conf = oh_anno.confidence
                            best_classification = oh_anno

                if best_classification is None:
                    return f"ROI-{oh.id}"
                else:
                    return f"{oh.id}: {best_classification.classname}, {best_classification.confidence:.2f}"

            if (
                self.descriptor.parameters.query_aware
                and query is not None
                and query.obj.type != ""
            ):
                matching_object_hypotheses = []
                for oh in object_hypotheses:
                    classifications = self.get_cas().filter_by_type_and_criteria(
                        robokudo.types.annotation.Classification,
                        oh.annotations,
                        criteria={"classname": ("==", f"{query.obj.type}")},
                    )

                    if len(classifications) > 0:
                        matching_object_hypotheses.append(oh)

                robokudo.utils.annotator_helper.draw_bounding_boxes_from_object_hypotheses(
                    visualization_img, matching_object_hypotheses, get_box_text
                )

            else:
                robokudo.utils.annotator_helper.draw_bounding_boxes_from_object_hypotheses(
                    visualization_img, object_hypotheses, get_box_text
                )

        for object_hypothesis in object_hypotheses:
            visualized_geometries.append(object_hypothesis.points)

            bb_annotations = self.get_cas().filter_by_type(
                type_to_include=robokudo.types.annotation.BoundingBox3DAnnotation,
                input_list=object_hypothesis.annotations,
            )
            for bb_annotation in bb_annotations:
                obb = robokudo.utils.type_conversion.get_o3d_obb_from_bounding_box_annotation(
                    bb_annotation
                )
                visualized_geometries.append(obb)
                robokudo.utils.o3d_helper.draw_wireframe_of_obb_into_image(
                    self.get_cas(), visualization_img, obb
                )

            pose_annotations = self.get_cas().filter_by_type(
                type_to_include=robokudo.types.annotation.PoseAnnotation,
                input_list=object_hypothesis.annotations,
            )

            for pose_annotation in pose_annotations:
                cluster_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.2
                )
                transform = robokudo.utils.type_conversion.get_transform_matrix_from_pose_annotation(
                    pose_annotation
                )
                cluster_frame.transform(transform)
                visualized_geometries.append(cluster_frame)

                # TODO Fetch mesh from Body
                # if object_hypothesis.object_knowledge is not None:
                #     ok = object_hypothesis.object_knowledge  # type:
                #
                #     if ok.mesh_ros_package == "" or ok.mesh_relative_path == "":
                #         continue
                #     mesh_path = FileLoader.get_path_to_file_in_ros_package(
                #         ros_pkg_name=ok.mesh_ros_package,
                #         relative_path=ok.mesh_relative_path)
                #
                #     if mesh_path not in self._mesh_cache:
                #         self._mesh_cache[mesh_path] = o3d.io.read_triangle_mesh(mesh_path)
                #     mesh_instance = copy.deepcopy(self._mesh_cache[mesh_path])
                #     mesh_instance.transform(transform)
                #     visualized_geometries.append(mesh_instance)

        if self.descriptor.parameters.visualize_full_cloud:
            visualized_geometries.append(self.get_cas().get(CASViews.CLOUD))

        self.get_annotator_output_struct().set_image(visualization_img)
        self.get_annotator_output_struct().set_geometries(visualized_geometries)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

"""Object knowledge visualization for RoboKudo.

This module provides an annotator for visualizing object knowledge, including
reference frames, components, and features of detected objects. It integrates
with the object knowledge base to display semantic information about objects
and their parts.
"""

import numpy as np
import numpy.typing as npt
import open3d as o3d
import py_trees
from typing_extensions import List, Optional

import robokudo.annotators
import robokudo.annotators.core
import robokudo.annotators.outputs
import robokudo.types.annotation
import robokudo.types.cv
import robokudo.types.tf
import robokudo.utils.annotator_helper
import robokudo.utils.cv_helper
import robokudo.utils.knowledge
import robokudo.utils.o3d_helper
import robokudo.utils.transform
import robokudo.utils.type_conversion
from robokudo.cas import CASViews
from robokudo.descriptors.object_knowledge.object_knowledge_iai_kitchen import (
    ObjectKnowledge,
)
from robokudo.object_knowledge_base import BaseObjectKnowledgeBase
from robokudo.types.scene import (
    ParthoodComponentHypothesis,
    ParthoodHypothesis,
    ParthoodFeatureHypothesis,
)


class ObjectKnowledgeVisualizer(robokudo.annotators.core.BaseAnnotator):
    """Annotator for visualizing object knowledge and part relationships.

    This annotator displays reference frames of objects and their components/features
    by integrating with the object knowledge base. It creates visualizations showing:
    - Object reference frames
    - Component and feature bounding boxes
    - Part relationships
    - 2D ROIs for parts in images
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for object knowledge visualization."""

        class Parameters:
            """Parameters for configuring object knowledge visualization."""

            def __init__(self) -> None:
                self.object_knowledge_base_ros_package: str = "robokudo"
                """ROS package containing knowledge base"""

                self.object_knowledge_base_name: str = "object_knowledge_iai_kitchen"
                """Name of knowledge base module"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "ObjectKnowledgeVisualizer",
        descriptor: "ObjectKnowledgeVisualizer.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the object knowledge visualizer.

        :param name: Name of the annotator instance, defaults to "ObjectKnowledgeVisualizer"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)

        self.object_kb = robokudo.utils.knowledge.load_object_knowledge_base(self)
        """Loaded object knowledge base"""

    def fill_parthood_hypothesis(
        self,
        ph: ParthoodHypothesis,
        object_knowledge: ObjectKnowledge,
        transform: npt.NDArray,
    ) -> bool:
        """Insert into a given ParthoodHypothesis the information that can be extracted from the object knowledge

        :param ph: A ParthoodHypothesis or one of its children.
        :param object_knowledge: Knowledge about the object and its parts
        :param transform: Transform between cam and parent object
        :return: bool: True if parthood hypothesis could be generated, False otherwise
        """
        ph.name = object_knowledge.name
        ph.source = robokudo.utils.annotator_helper.generate_source_name(self)

        # TODO: Cloud generation. Cut out from whole scene, but also from object pointcloud (if existing)

        obb = robokudo.utils.knowledge.get_obb_for_child_object_and_transform(
            object_knowledge, transform
        )
        corner_points = robokudo.utils.o3d_helper.get_2d_bounding_rect_from_3d_bb(
            self.get_cas(), obb
        )

        image_height = self.get_cas().get(CASViews.COLOR_IMAGE).shape[0]
        image_width = self.get_cas().get(CASViews.COLOR_IMAGE).shape[1]

        if not robokudo.utils.cv_helper.sanity_checks_bounding_rects(
            corner_points, image_width, image_height
        ):
            return False

        corner_points = robokudo.utils.cv_helper.clamp_bounding_rect(
            corner_points, image_width=image_width, image_height=image_height
        )

        roi = robokudo.types.cv.ImageROI()
        roi.roi.pos.x = corner_points[0]
        roi.roi.pos.y = corner_points[1]
        roi.roi.width = corner_points[2]
        roi.roi.height = corner_points[3]
        ph.roi = roi
        # Generate a full mask for now. There shouldn't be too much background on Parthood Images.
        # Otherwise, TODO: Use SegmentAnything?
        ph.roi.mask = np.ones((roi.roi.height, roi.roi.width), dtype=np.uint8) * 255
        return True

    def generate_parthood_hypotheses(
        self, object_knowledge: ObjectKnowledge, transform: npt.NDArray
    ) -> Optional[List[ParthoodHypothesis]]:
        """Generate hypotheses for object parts and features.

        Creates ParthoodComponentHypothesis and ParthoodFeatureHypothesis objects
        for all components and features defined in the object knowledge.

        :param object_knowledge: Knowledge about the object and its parts
        :param transform: Transform between camera and object
        :return: List of generated hypotheses or None if no parts exist
        """
        if not BaseObjectKnowledgeBase.has_parthood_childs(object_knowledge):
            return None
        annotations = []
        for component in object_knowledge.components:
            pch = ParthoodComponentHypothesis()
            if self.fill_parthood_hypothesis(pch, component, transform):
                annotations.append(pch)

        for feature in object_knowledge.features:
            pfh = ParthoodFeatureHypothesis()
            if self.fill_parthood_hypothesis(pfh, feature, transform):
                annotations.append(pfh)

        return annotations

    def update(self) -> py_trees.common.Status:
        """Update the visualization with current object knowledge.

        Creates visualizations containing:

        * Object reference frames
        * Oriented bounding boxes for objects and parts
        * 2D ROIs in the color image
        * Part relationship annotations in the CAS

        :return: SUCCESS after creating visualizations
        """
        self.object_kb = robokudo.utils.knowledge.load_object_knowledge_base(self)
        cloud = self.get_cas().get(CASViews.CLOUD)
        color = self.get_cas().get_copy(CASViews.COLOR_IMAGE)

        # Look for objects with classifications and check if we have ObjectKnowledge about them

        object_hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.ObjectHypothesis
        )

        object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        obbs_list = []

        for object_hypothesis in object_hypotheses:
            classification = None
            pose = None

            for annotation in object_hypothesis.annotations:
                # This assumes that we have a proper classification
                # and that it's the first one in the list of annotations
                if (
                    isinstance(annotation, robokudo.types.annotation.Classification)
                    and annotation.classname in self.object_kb.entries
                    and classification is None
                ):
                    classification = annotation
                    continue

                if (
                    isinstance(annotation, robokudo.types.annotation.PoseAnnotation)
                    and pose is None
                ):
                    pose = annotation
                    continue

            if classification and pose:
                object_knowledge = self.object_kb.entries[classification.classname]
                object_transform = robokudo.utils.type_conversion.get_transform_matrix_from_pose_annotation(
                    pose
                )
                object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.2
                )
                object_frame.transform(object_transform)
                obbs = robokudo.utils.knowledge.get_obbs_for_object_and_childs(
                    object_knowledge, object_transform
                )

                robokudo.utils.o3d_helper.draw_wireframe_of_obb_into_image(
                    self.get_cas(), color, obbs[classification.classname]
                )

                for o_key, o_value in obbs.items():
                    robokudo.utils.o3d_helper.get_2d_corner_points_from_3d_bb(
                        self.get_cas(), o_value
                    )
                    obbs_list.append(o_value)

                parthood_annotations = self.generate_parthood_hypotheses(
                    object_knowledge, object_transform
                )
                object_hypothesis.annotations.extend(parthood_annotations)
                # Also add the annotations to the CAS annotations such that annotators can pick them up
                self.get_cas().annotations.extend(parthood_annotations)

        # Visualization has two steps:
        # 1) Visualize the object with the estimated pose
        # 2) Visualize components and features

        visualized_geometries = [cloud, object_frame]
        visualized_geometries.extend(obbs_list)

        self.get_annotator_output_struct().set_geometries(visualized_geometries)
        self.get_annotator_output_struct().set_image(color)

        return py_trees.common.Status.SUCCESS

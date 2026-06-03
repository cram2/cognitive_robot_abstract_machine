from timeit import default_timer

import numpy as np
from py_trees.common import Status
from typing_extensions import Optional, Dict, Any

from robokudo.annotators.core import BaseAnnotator
from robokudo.io.semantic_digital_twin import SemanticDigitalTwinAdapter, Object
from robokudo.types.annotation import (
    PositionAnnotation,
    PoseAnnotation,
    Classification,
    BoundingBox3DAnnotation,
    SemanticColor,
    ColorHistogram,
)
from robokudo.types.cv import TSDFAnnotation
from robokudo.types.scene import ObjectHypothesis


class SemanticDigitalTwinConnector(BaseAnnotator):
    """An annotator that synchronizes the current state of the world with the semdt."""

    class Descriptor(BaseAnnotator.Descriptor):

        class Parameters:
            def __init__(self) -> None:
                """Initialize a new set of parameters for the descriptor."""

                self.urdf_path: Optional[str] = None
                """Optional Path to the URDF file of the world"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "WorldValidator",
        descriptor: "SemanticDigitalTwinConnector.Descriptor" = Descriptor(),
    ):
        """Default construction. Minimal one-time init!"""
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.semdt_adapter = SemanticDigitalTwinAdapter(
            self.get_cas, urdf_path=descriptor.parameters.urdf_path
        )
        """An instance of SemanticDigitalTwinConnector used to connect to the semdt."""

    def extract_data(self, oh: ObjectHypothesis) -> Dict[str, Any]:
        """Extracts data from an object hypothesis to a simple dictionary.

        :param oh: Object hypothesis to extract data from.
        :return: A dictionary containing the object hypothesis data in form of a dictionary.
        """
        data: Dict[str, Any] = {}
        cas = self.get_cas()

        if hasattr(oh, "roi"):
            data["oh_roi"] = oh.roi.roi

        positions = cas.filter_by_type(PositionAnnotation, oh.annotations)
        if len(positions) > 0:
            translation_vector = np.array(positions[0].translation)
            data["translation_vector"] = translation_vector

        poses = cas.filter_by_type(PoseAnnotation, oh.annotations)
        if len(poses) > 0:
            translation_vector = np.array(poses[0].translation)
            data["translation_vector"] = translation_vector

        classes = cas.filter_by_type(Classification, oh.annotations)
        if len(classes) > 0:
            data["class"] = classes[0]

        bboxs = cas.filter_by_type(BoundingBox3DAnnotation, oh.annotations)
        if len(bboxs) > 0:
            data["bbox"] = bboxs[0]

        semantic_colors = cas.filter_by_type(SemanticColor, oh.annotations)
        if len(semantic_colors) > 0:
            data["semantic_color"] = semantic_colors[0]

        color_histograms = cas.filter_by_type(ColorHistogram, oh.annotations)
        if len(color_histograms) > 0:
            data["color_histogram"] = color_histograms[0]

        tsdfs = cas.filter_by_type(TSDFAnnotation, oh.annotations)
        if len(tsdfs) > 0:
            data["tsdf"] = tsdfs[0]

        return data

    def update(self) -> Status:
        """Synchronise the current RoboKudo state with the current semdt state."""
        start_timer = default_timer()

        ohs: list[ObjectHypothesis] = self.get_cas().filter_annotations_by_type(
            ObjectHypothesis
        )

        # Get the best data from oh
        new_objects = [Object(data=self.extract_data(oh)) for oh in ohs]

        diffs = self.semdt_adapter.compute_diffs(new_objects)

        self.semdt_adapter.apply_diffs(diffs)

        # def string_to_type(type_string):
        #     try:
        #         module_path, class_name = type_string.rsplit('.', 1)
        #         module = importlib.import_module(module_path)
        #         return getattr(module, class_name)
        #     except (ImportError, AttributeError, ValueError) as e:
        #         raise ValueError(f"Cannot import type '{type_string}': {e}")

        # for diff in diffs:
        #     for test_dict in diff.test:
        #         self.rk_logger.info(json.dumps(test_dict))

        #         test_type = string_to_type(test_dict['type'])
        #         test_instance = test_type.from_json(test_dict)
        #         self.rk_logger.info(f"{test_instance}")

        self.rk_logger.info(
            f"SemDT \nKS Entities: {len(self.semdt_adapter.world.kinematic_structure_entities)}\nViews: {len(self.semdt_adapter.world.semantic_annotations)}\nConnections: {len(self.semdt_adapter.world.connections)}"
        )

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

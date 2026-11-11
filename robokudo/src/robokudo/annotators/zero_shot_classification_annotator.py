"""Zero shot classification annotator configuration and annotation behavior."""

from __future__ import annotations

# %% Imports
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np
from krrood.ormatic.utils import classproperty
from krrood.patterns.field_metadata import JSONMetadata
from PIL import Image
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, Callable, Generic, TypeVar

from robokudo.annotators.core import ThreadedAnnotator
from robokudo.cas import CASViews
from robokudo.datastructures.annotator_descriptor import AnnotatorDescriptor
from robokudo.datastructures.annotator_parameters import AnnotatorParameters
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis

if TYPE_CHECKING:
    import numpy.typing as npt


# %% ZeroShotClassification parameters


@dataclass(kw_only=True)
class ZeroShotClassificationParameters(AnnotatorParameters):
    """Vocabulary and processing choices for image classification."""

    vocabulary: list[str] = field(
        default_factory=lambda: [
            "Milk Container",
            "Box of Cornflakes",
            "Pancake Batter",
            "Instant Coffee",
            "Crepe Pan",
            "Dish Soap",
        ]
    )
    """The vocabulary used for object classification."""

    analysis_scope: CASViews | type[ObjectHypothesis] = CASViews.COLOR_IMAGE
    """Select the whole color image or individual object hypotheses for classification."""

    filter_function: Callable[[ObjectHypothesis], bool] | None = None
    """An optional filter function used to filter object hypotheses before analyzing them."""

    maximum_saved_classifications: int = field(
        init=False, metadata=JSONMetadata(serialize=True).as_dict()
    )
    """How many annotations to create per analyzed image or hypothesis."""

    def __post_init__(self) -> None:
        """Initialize values derived from the configured processing settings."""
        self.maximum_saved_classifications: int = min(5, len(self.vocabulary))


# %% ZeroShotClassification descriptor

ClassificationParametersType = TypeVar(
    "ClassificationParametersType", bound=ZeroShotClassificationParameters
)
"""Concrete vocabulary and model settings handled by a classification annotator."""


@dataclass(kw_only=True)
class ZeroShotClassificationDescriptor(
    AnnotatorDescriptor[ZeroShotClassificationParameters]
):
    """Declarative configuration of :class:`ZeroShotClassificationAnnotator`."""

    parameters: ZeroShotClassificationParameters = field(
        default_factory=ZeroShotClassificationParameters.from_defaults
    )
    """Effective settings initialized from application and annotator defaults."""


# %% ZeroShotClassification annotator


class ZeroShotClassificationAnnotator(
    ThreadedAnnotator[AnnotatorDescriptor[ClassificationParametersType]],
    Generic[ClassificationParametersType],
    ABC,
):
    """
    Classifies the whole image or each object hypothesis against a freely chosen
    vocabulary of class names, without training on those classes.
    """

    @classproperty
    def Descriptor(cls) -> type[AnnotatorDescriptor]:
        """Specialize :meth:`robokudo.annotators.core.BaseAnnotator.Descriptor`."""
        return ZeroShotClassificationDescriptor

    @classproperty
    def visualization_font(cls) -> int:
        """
        The OpenCV font the classifications are written onto the output image with.
        """
        return cv2.FONT_HERSHEY_COMPLEX

    @abstractmethod
    def classify(self, images: list[Image.Image]) -> list[list[Classification]]:
        """
        Classify each image against the vocabulary.

        :param images: The images to classify, in RGB channel order.
        :return: For each image, at most ``maximum_saved_classifications`` classifications, the most
            confident first.
        """

    @staticmethod
    def _rgb_image(color_image: npt.NDArray[np.uint8]) -> Image.Image:
        """
        :param color_image: A color image in the BGR channel order of the CAS.
        :return: The color image in the RGB channel order :meth:`classify` expects.
        """
        return Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

    def _analyze_scene(self, color_image: npt.NDArray[np.uint8]) -> Status:
        """
        Analyze the entire color image.

        :param color_image: The color image to analyze.
        :return: The py_trees status.
        """
        classifications = self.classify([self._rgb_image(color_image)])[0]
        self.get_cas().annotations.extend(classifications)

        visualization_img = color_image.copy()

        self.rk_logger.info("Image Classification Results:")
        self.rk_logger.info("Label \t\t Score")
        for i, classification in enumerate(classifications):
            text = f"{classification.confidence:.2f} /{classification.classname}"
            visualization_img = cv2.putText(
                visualization_img,
                text,
                (10, 40 + 20 * i),
                self.visualization_font,
                0.5,
                (0, 0, 255),
                1,
                2,
            )

            self.rk_logger.info(
                f"Top {i} prediction: Class {classification.classname} with "
                f"confidence\t {classification.confidence:.4f}"
            )

        self.get_annotator_output_struct().set_image(visualization_img)
        return Status.SUCCESS

    def _analyze_object_hypotheses(self, color_image: npt.NDArray[np.uint8]) -> Status:
        """
        Analyze the object hypothesis using the given color image.

        :param color_image: The color image to use for analyzation.
        :return: The py_trees status.
        """
        ohs: list[ObjectHypothesis] = self.get_cas().filter_annotations_by_type(
            ObjectHypothesis
        )
        if len(ohs) == 0:
            self.rk_logger.info("Found 0 object hypotheses to analyze")
            return Status.SUCCESS

        if self.descriptor.parameters.filter_function is not None:
            ohs = list(filter(self.descriptor.parameters.filter_function, ohs))

            if len(ohs) == 0:
                self.rk_logger.info("All object hypotheses were filtered out")
                return Status.SUCCESS

        # Create per-object image crop
        rgb_image = self._rgb_image(color_image)
        cropped_images = [rgb_image.crop(oh.roi.roi.get_corner_points()) for oh in ohs]

        visualization_img = color_image.copy()

        for object_hypothesis, classifications in zip(
            ohs, self.classify(cropped_images)
        ):
            self.rk_logger.info(
                f"Classification results for Object hypothesis ID: '{object_hypothesis.id}'"
            )

            best_classification = classifications[0]
            text = (
                f"{best_classification.confidence:.4f} / "
                f"{best_classification.classname}"
            )
            x1, y1, x2, y2 = object_hypothesis.roi.roi.get_corner_points()
            visualization_img = cv2.rectangle(
                visualization_img, (x1, y1), (x2, y2), (255, 255, 255), 2
            )
            visualization_img = cv2.putText(
                visualization_img,
                text,
                (x1, y1 - 20),
                self.visualization_font,
                0.5,
                (0, 0, 255),
                1,
                2,
            )

            object_hypothesis.annotations.extend(classifications)
            for i, classification in enumerate(classifications):
                self.rk_logger.info(
                    f"Top {i}: Classified as {classification.classname} "
                    f"with confidence {classification.confidence:.4f}"
                )

        self.get_annotator_output_struct().set_image(visualization_img)
        return Status.SUCCESS

    def compute(self) -> Status:
        """
        Classify the color image or its object hypotheses depending on the analysis
        scope.

        :return: Whether the computation was successfull or not.
        """
        start_time = time.perf_counter()
        analysis_scope = self.descriptor.parameters.analysis_scope

        color_image = self.get_cas().get(CASViews.COLOR_IMAGE).copy()

        if analysis_scope == CASViews.COLOR_IMAGE:
            status = self._analyze_scene(color_image)
        elif analysis_scope == ObjectHypothesis:
            status = self._analyze_object_hypotheses(color_image)
        else:
            self.feedback_message = f"Unknown analysis scope: {analysis_scope}"
            return Status.FAILURE

        self.feedback_message = (
            f"Processing took {time.perf_counter() - start_time:.2f}s"
        )
        return status

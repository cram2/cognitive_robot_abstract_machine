from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    Classification,
    ColorHistogram,
    PoseAnnotation,
)
from robokudo.types.cv import ImageROI
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.hypothesis_comparators import ObjectHypothesisComparator


@dataclass(eq=False)
class ObjectAssociator(BaseAnnotator):
    """
    An annotator that associates the objects of the current iteration with the objects
    of the last iteration.
    """

    name: str = __qualname__
    """
    Name of this node in the behaviour tree.
    """

    tracked_objects: list[ObjectHypothesis] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Object state retained between processing iterations.
    """

    object_comparator: ObjectHypothesisComparator = field(init=False, repr=False)
    """
    Compares current hypotheses with tracked objects.
    """

    def __post_init__(self) -> None:
        """
        Initialize annotator resources after
        :meth:`robokudo.annotators.core.BaseAnnotator.__post_init__`.
        """
        super().__post_init__()
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.object_comparator = (
            ObjectHypothesisComparator(self.get_cas)
            .with_comparator_for(PoseAnnotation, 0.3, max_distance=0.5)
            .with_comparator_for(Classification, 0.3)
            .with_comparator_for(BoundingBox3DAnnotation, 0.1)
            .with_comparator_for(ColorHistogram, 0.2)
            .with_comparator_for(ImageROI, 0.1)
        )

    def update(self) -> Status:
        new_ohs: list[ObjectHypothesis] = self.get_cas().filter_annotations_by_type(
            ObjectHypothesis
        )

        if len(self.tracked_objects) == 0:
            for new_oh in new_ohs:
                self.tracked_objects.append(new_oh)
            return Status.SUCCESS

        matched_new_ids: set[int] = set()

        for new_oh in new_ohs:
            best_match_id: int | None = None
            best_similarity = 0.0

            for oh_id, prev_oh in enumerate(self.tracked_objects):
                similarity = self.object_comparator.compute_similarity(prev_oh, new_oh)
                if similarity > 0.5 and similarity > best_similarity:
                    best_match_id = oh_id
                    best_similarity = similarity

            if best_match_id is not None and best_similarity > 0.5:
                self.rk_logger.info(
                    f"Associate object {new_oh.id} with {best_match_id} ({self.tracked_objects[best_match_id].id})."
                )
                matched_new_ids.add(best_match_id)
                self.tracked_objects[best_match_id] = new_oh
            else:
                self.tracked_objects.append(new_oh)
                self.rk_logger.info(
                    f"Discovered new object {len(self.tracked_objects)}"
                )
        return Status.SUCCESS

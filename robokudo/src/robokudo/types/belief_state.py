from __future__ import annotations
from robokudo.types.annotation import BoundingBox3DAnnotation
from robokudo.types.annotation import PoseAnnotation
from uuid import UUID
from collections import deque

from robokudo.types.scene import ObjectHypothesis
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, Deque, Optional, Self

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    pass


@dataclass
class ObjectBeliefState:
    body: Body
    """The semdt body associated with the belief state."""

    max_num_hypotheses: int = 1
    """The maximum number of object hypotheses to keep in the belief state."""

    hypotheses: Deque[ObjectHypothesis] = field(init=False)
    """The hypotheses associated with the belief state."""

    def __post_init__(self):
        self.hypotheses = deque(maxlen=self.max_num_hypotheses)

    @classmethod
    def create_with_new_body(cls, name: Optional[PrefixedName] = None) -> Self:
        """Create a new ObjectBeliefState with a new Body.

        :param name: Name of the body.
        :return: The new object belief state.
        """
        return ObjectBeliefState(
            body=Body(name=name) if name is not None else Body(),
        )

    @property
    def uuid(self) -> UUID:
        """The UUID of the belief state."""
        return self.body.id

    @property
    def latest_hypothesis(self) -> Optional[ObjectHypothesis]:
        """The latest hypothesis added to the belief state."""
        if len(self.hypotheses) == 0:
            return None
        return self.hypotheses[len(self.hypotheses) - 1]

    @property
    def latest_pose(self) -> Optional[PoseAnnotation]:
        """The newest pose annotation of the latest hypothesis."""
        hypothesis = self.latest_hypothesis
        if hypothesis is not None:
            for annotation in reversed(hypothesis.annotations):
                if type(annotation) == PoseAnnotation:
                    return annotation
        return None

    @property
    def latest_bbox_3d(self) -> Optional[BoundingBox3DAnnotation]:
        """The newest 3D bounding box annotation of the latest hypothesis."""
        hypothesis = self.latest_hypothesis
        if hypothesis is not None:
            for annotation in reversed(hypothesis.annotations):
                if type(annotation) == BoundingBox3DAnnotation:
                    return annotation
        return None

    def add_hypothesis(self, hypothesis: ObjectHypothesis) -> Self:
        """Add a new hypothesis to the belief state.

        :param hypothesis: The hypothesis to add.
        """
        self.hypotheses.append(hypothesis)
        return self

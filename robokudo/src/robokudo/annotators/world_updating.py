"""
Putting the objects a pass perceived into the world its pipeline perceives into.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.exceptions import CameraPoseOutsideWorldError
from robokudo.perception_beliefs import ObjectMatcher, PerceptionBelief
from robokudo.types.scene import ObjectHypothesis

logger = logging.getLogger(__name__)

# %% world updating


@dataclass(kw_only=True, eq=False)
class WorldUpdatingAnnotator(BaseAnnotator):
    """
    Puts the object hypotheses of the CAS into the world as beliefs: each belief updates
    the perceivable object it matches, as
    :meth:`PerceptionBelief.update_object_in_world` describes, or is added as a new
    object.

    The update is one modification of the world, which no other thread changes meanwhile
    and whose changes are published at its end.
    """

    minimum_class_confidence: float = 0.5
    """
    How confident a classification has to be for an object to be believed of the class
    it names; a less confident one is believed an :class:`UnknownObject`.
    """

    def __post_init__(self) -> None:
        super().__init__(name=type(self).__name__)

    def update(self) -> Status:
        """
        Put the object hypotheses the pass found so far into the world, and succeed; a
        pass without a camera pose in the world logs a warning instead.

        :raises CameraPoseOutsideWorldError: If the camera pose of the pass is measured
            against a body that is not part of the world.
        """
        cas = self.get_cas()
        observation = cas.camera_observation
        if (
            observation is None
            or observation.world_T_camera is None
            or observation.world_T_camera.reference_frame is None
        ):
            logger.warning("Leaving out a pass without a camera pose in the world.")
            return Status.SUCCESS
        world_T_camera = observation.world_T_camera
        with self.world.modify_world():
            if (
                world_T_camera.reference_frame
                not in self.world.kinematic_structure_entities
            ):
                raise CameraPoseOutsideWorldError(
                    reference_frame_name=str(world_T_camera.reference_frame.name)
                )
            beliefs = [
                belief
                for hypothesis in cas.filter_annotations_by_type(ObjectHypothesis)
                if (
                    belief := PerceptionBelief.from_hypothesis(
                        hypothesis=hypothesis,
                        world_T_camera=world_T_camera,
                        minimum_class_confidence=self.minimum_class_confidence,
                    )
                )
                is not None
            ]
            matches = ObjectMatcher().match(beliefs=beliefs, world=self.world)
            for belief in beliefs:
                object_in_world = matches.get(belief)
                if object_in_world is None:
                    belief.add_to_world(world=self.world)
                    continue
                belief.update_object_in_world(
                    object_in_world=object_in_world, world=self.world
                )
        return Status.SUCCESS

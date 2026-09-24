"""
Native designators with the metadata shown in plan snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.datastructures.enums import Arms
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.base import BaseMotion
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

# %% plan parameters


@dataclass
class ArmSelectionAction(ActionDescription):
    """
    An action parameterized by the arm that performs it.
    """

    arm: Arms
    """The selected arm."""


@dataclass
class BodyTargetMotion(BaseMotion):
    """
    A motion parameterized by the body it acts on.
    """

    target_body: Body
    """The body referenced by the motion."""


@dataclass
class MultipleArmAction(ActionDescription):
    """
    An action parameterized by several selected arms.
    """

    arms: list[Arms]
    """The arms selected to perform the action."""


@dataclass
class AnnotationTargetMotion(BaseMotion):
    """
    A motion parameterized by a semantic annotation.
    """

    target_annotation: SemanticAnnotation
    """The annotation referenced by the motion."""

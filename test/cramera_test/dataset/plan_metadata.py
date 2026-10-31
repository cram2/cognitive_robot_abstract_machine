"""
Native designators with the metadata shown in plan snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.robot_plans.motions.base import BaseMotion
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

# %% plan parameters


@dataclass
class BodyTargetMotion(BaseMotion):
    """
    A motion parameterized by the body it acts on.
    """

    target_body: Body
    """The body referenced by the motion."""


@dataclass
class AnnotationTargetMotion(BaseMotion):
    """
    A motion parameterized by a semantic annotation.
    """

    target_annotation: SemanticAnnotation
    """The annotation referenced by the motion."""

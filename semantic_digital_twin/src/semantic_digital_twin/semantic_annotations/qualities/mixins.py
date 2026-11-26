"""
Mixins for each kind of quality an annotation can have, such as the ripeness of a fruit.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass, field

from krrood.patterns.mixin import Mixin
from semantic_digital_twin.semantic_annotations.qualities.metadata import (
    IsQualityRelationship,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import Ripeness
from semantic_digital_twin.semantic_annotations.qualities.relationship import (
    QualityRelationship,
)

# %% Qualities of food


@dataclass(eq=False, kw_only=True)
class HasRipeness(QualityRelationship, Mixin):
    """
    A mixin class for semantic annotations that ripen.
    """

    ripeness: Ripeness | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    How far the semantic annotation has ripened, or ``None`` when that is unknown.
    """

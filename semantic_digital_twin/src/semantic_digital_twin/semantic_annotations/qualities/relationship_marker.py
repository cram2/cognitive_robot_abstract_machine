"""
The marker of the fields of an annotation that hold its qualities.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.semantic_relationship import (
    IsSemanticRelationship,
)

# %% Relationship marker


@dataclass(kw_only=True)
class IsQualityRelationship(IsSemanticRelationship):
    """
    Marks a field as holding a
    :class:`~semantic_digital_twin.semantic_annotations.qualities.quality.Quality` of its
    owner.
    """

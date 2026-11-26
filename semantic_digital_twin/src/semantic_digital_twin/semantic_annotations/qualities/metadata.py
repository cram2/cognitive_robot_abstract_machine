"""
Metadata for the quality relation between semantic annotations and their qualities.

Holds only the vocabulary of the relation, so code that reads or fills quality fields
can depend on it without depending on the annotation mixins.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.semantic_relationship import (
    IsSemanticRelationship,
)

# %% Relation metadata


@dataclass(kw_only=True)
class IsQualityRelationship(IsSemanticRelationship):
    """
    Marks a field as holding a
    :class:`~semantic_digital_twin.semantic_annotations.qualities.quality.Quality` of its
    owner.
    """

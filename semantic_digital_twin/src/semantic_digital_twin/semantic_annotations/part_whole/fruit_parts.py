"""
Mixins for each anatomical part of a fruit, such as its peel, flesh or seeds.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from krrood.patterns.mixin import Mixin
from semantic_digital_twin.semantic_annotations.part_whole.relationship_marker import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.semantic_annotations.part_whole.relationship import (
    HasParts,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Core,
        Flesh,
        Peel,
        Pit,
        Pith,
        SeedChamber,
        Seed,
        Stem,
    )

# %% Parts of fruits


@dataclass(eq=False, kw_only=True)
class HasPeel(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have a peel.
    """

    peel: Peel | None = field(
        default=None,
        hash=False,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The peel of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasFlesh(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have flesh.
    """

    flesh: Flesh | None = field(
        default=None,
        hash=False,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The flesh of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasCore(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have a core.
    """

    core: Core | None = field(
        default=None,
        hash=False,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The core of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasSeeds(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have separate seeds.
    """

    seeds: list[Seed] = field(
        default_factory=list,
        hash=False,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The seeds of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasStem(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have a stem.
    """

    stem: Stem | None = field(
        default=None,
        hash=False,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The stem of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasPit(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have a pit.
    """

    pit: Pit | None = field(
        default=None,
        hash=False,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The pit of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasSeedChamber(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have a seed chamber.
    """

    seed_chamber: SeedChamber | None = field(
        default=None,
        hash=False,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The seed chamber of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasPith(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have a pith.
    """

    pith: Pith | None = field(
        default=None,
        hash=False,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The pith of the semantic annotation.
    """

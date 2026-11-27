"""
Mixins for the parts of an egg: its shell, which covers nuts and seeds too, its albumen
and its yolk.
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
        Albumen,
        Shell,
        Yolk,
    )

# %% Parts of eggs


@dataclass(eq=False, kw_only=True)
class HasShell(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have a shell.
    """

    shell: Shell | None = field(
        default=None,
        hash=False,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The shell of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasAlbumen(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have an albumen.
    """

    albumen: Albumen | None = field(
        default=None,
        hash=False,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The albumen of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasYolk(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have a yolk.
    """

    yolk: Yolk | None = field(
        default=None,
        hash=False,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The yolk of the semantic annotation.
    """

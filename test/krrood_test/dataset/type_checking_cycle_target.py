"""
A module importing a name back from :mod:`type_checking_cycle_owner` at module level,
which is why that module only imports this one while type checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .type_checking_cycle_owner import HolderBindingItsContent


@dataclass
class ContainerOfBoundHolder:
    """
    A container of holders that bind their content type.
    """

    holders: list[HolderBindingItsContent] = field(default_factory=list)
    """
    The holders placed in this container.
    """

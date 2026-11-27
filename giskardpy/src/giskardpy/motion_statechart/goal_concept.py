"""
The concept of a node taking a goal, which the parameter mixins of goals share.
"""

from __future__ import annotations

from krrood.patterns.mixin import Mixin

# %% taking a goal


class HasGoal(Mixin):
    """
    Marks a node taking a goal, which the motion tasks below it work towards together.
    """

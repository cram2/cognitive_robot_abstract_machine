"""
A world with a slider moving along x and a box beside it, for statechart tests that move
something in the world.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% the world


@dataclass(kw_only=True)
class SlidingWorld:
    """
    A world with a box resting at the origin and a slider that moves along x, so that
    the box moves with the slider only once it hangs below it.
    """

    world: World
    """
    The world holding every body.
    """

    slider: Body
    """
    The body moved along x by :attr:`slider_connection`.
    """

    box: Body
    """
    The body fixed to the root, which is moved below :attr:`slider`.
    """

    slider_connection: ScrewAxisConnection
    """
    The joint moving :attr:`slider` along x.
    """

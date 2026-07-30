"""
Shared pourable-container mimic and single-cup world builder for pouring tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% shared pourable container mimic


@dataclass(eq=False)
class PourableContainer(HasFillLevel):
    """
    Minimal pourable container for testing.

    Connected to its parent via a revolute joint representing the tilt angle.
    """

    @classproperty
    def _parent_connection_type(self):
        return RevoluteConnection


# %% single-cup world builder


def build_single_cup_world(
    outflow_rate_constant: float = 1.0,
    initial_tilt: float = 0.1,
) -> tuple[World, PourableContainer]:
    """
    Builds a minimal world with one tilt-jointed pourable container filled to 100%.

    :param outflow_rate_constant: Outflow rate constant for the container's fill
        equation.
    :param initial_tilt: Initial tilt-joint position in radians.
    :return: The world and the container annotation living in it.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
    with world.modify_world():
        cup = PourableContainer.create_with_new_body_in_world(
            name=PrefixedName("cup"),
            world=world,
            active_axis=Vector3(0, 1, 0),
            connection_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=math.pi / 2, velocity=2.0),
            ),
            scale=Scale(0.4, 0.4, 1.0),
        )
    cup.initialize_fill_level(
        world=world,
        initial_fill=1.0,
        outflow_rate_constant=outflow_rate_constant,
    )
    world.set_positions_1DOF_connection({cup.root.parent_connection: initial_tilt})
    return world, cup

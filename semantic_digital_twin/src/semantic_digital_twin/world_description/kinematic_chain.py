"""
Chains through the kinematic structure of a world.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% kinematic chain


@dataclass(kw_only=True)
class KinematicChain:
    """
    The chain of connections in the kinematic structure of a world that leads from
    :attr:`root` to :attr:`tip`.

    .. note:: :attr:`root` need not be an ancestor of :attr:`tip`, as with the handle a
        gripper holds and the gripper's tool frame.
    """

    root: KinematicStructureEntity
    """
    The start of the chain, which the pose of :attr:`tip` is expressed relative to.
    """

    tip: KinematicStructureEntity
    """
    The end of the chain.
    """

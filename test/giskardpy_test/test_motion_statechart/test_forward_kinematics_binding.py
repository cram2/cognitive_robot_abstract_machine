"""
Measuring a bound chain from another root or up to another tip.
"""

from __future__ import annotations

import pytest

from giskardpy.motion_statechart.binding_policy import ForwardKinematicsBinding
from krrood.symbolic_math.float_variable_data import FloatVariableData
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain

# %% the chain the binding was given


@pytest.fixture()
def tip_only_chain(mini_world: World) -> KinematicChain:
    """
    The chain from the tip of :func:`mini_world` to itself.
    """
    tip = mini_world.get_body_by_name("tip")
    return KinematicChain(root=tip, tip=tip)


def _bound(*, chain: KinematicChain, world: World) -> ForwardKinematicsBinding:
    """
    :return: A binding of `chain`, bound to the current state of `world`.
    """
    binding = ForwardKinematicsBinding(
        float_variable_data=FloatVariableData(),
        name=PrefixedName("binding"),
        kinematic_chain=chain,
    )
    binding.bind(world=world)
    return binding


def test_changing_the_root_measures_the_chain_from_the_new_root(
    mini_world: World, tip_only_chain: KinematicChain
):
    binding = _bound(chain=tip_only_chain, world=mini_world)
    binding.change_root(root=mini_world.root, world=mini_world)
    assert binding.kinematic_chain == KinematicChain(
        root=mini_world.root, tip=tip_only_chain.tip
    )


def test_changing_the_root_leaves_the_chain_the_binding_was_given(
    mini_world: World, tip_only_chain: KinematicChain
):
    tip = tip_only_chain.tip
    binding = _bound(chain=tip_only_chain, world=mini_world)
    binding.change_root(root=mini_world.root, world=mini_world)
    assert tip_only_chain == KinematicChain(root=tip, tip=tip)


def test_changing_the_tip_leaves_the_chain_the_binding_was_given(
    mini_world: World, tip_only_chain: KinematicChain
):
    tip = tip_only_chain.tip
    binding = _bound(chain=tip_only_chain, world=mini_world)
    binding.change_tip(tip=mini_world.root, world=mini_world)
    assert binding.kinematic_chain == KinematicChain(root=tip, tip=mini_world.root)
    assert tip_only_chain == KinematicChain(root=tip, tip=tip)

# %% imports

from __future__ import annotations

from dataclasses import dataclass

from krrood.ormatic.utils import classproperty

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.effects import (
    ClosedEffect,
    OpenedEffect,
    PouringEffect,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

# %% test annotations


@dataclass(eq=False)
class _ScalarPropertyAnnotation(SemanticAnnotation):
    """Annotation stub exposing a single scalar property for boundary testing."""

    scalar_value: float = 0.0
    """The scalar the effects under test read."""


@dataclass(eq=False)
class _TiltingContainer(HasFillLevel):
    """A pourable container attached to its parent by a single tilting DOF."""

    @classproperty
    def _parent_connection_type(self):
        return RevoluteConnection


def _effect_target(scalar_value: float) -> _ScalarPropertyAnnotation:
    """Builds an annotation stub holding the given scalar property value."""
    return _ScalarPropertyAnnotation(scalar_value=scalar_value)


def _cup_with_fill_level(fill: float) -> _TiltingContainer:
    """Builds a lone tilting cup in a minimal world, filled to the given level."""
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
    with world.modify_world():
        cup = _TiltingContainer.create_with_new_body_in_world(
            name=PrefixedName("cup"),
            world=world,
            active_axis=Vector3(0, 1, 0),
            connection_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=-2.0, velocity=-1.0),
                upper=DerivativeMap(position=2.0, velocity=1.0),
            ),
            scale=Scale(0.1, 0.1, 0.2),
        )
    cup.initialize_fill_level(world=world, initial_fill=fill)
    return cup


# %% opened effect boundaries


class TestOpenedEffectBoundary:
    """Validates the monotone-increasing achievement boundary of ``OpenedEffect``."""

    def _effect(self, current: float) -> OpenedEffect:
        return OpenedEffect(
            target_object=_effect_target(current),
            property_getter=lambda annotation: annotation.scalar_value,
            goal_value=0.5,
            tolerance=0.05,
        )

    def test_achieved_exactly_at_lower_tolerance_boundary(self):
        """The effect is achieved when the value reaches ``goal - tolerance`` exactly."""
        assert self._effect(0.45).is_achieved()

    def test_not_achieved_just_below_lower_tolerance_boundary(self):
        """The effect is not achieved while the value stays below ``goal - tolerance``."""
        assert not self._effect(0.44).is_achieved()

    def test_achieved_above_goal(self):
        """The effect stays achieved for any value beyond the goal."""
        assert self._effect(0.9).is_achieved()


# %% closed effect boundaries


class TestClosedEffectBoundary:
    """Validates the monotone-decreasing achievement boundary of ``ClosedEffect``."""

    def _effect(self, current: float) -> ClosedEffect:
        return ClosedEffect(
            target_object=_effect_target(current),
            property_getter=lambda annotation: annotation.scalar_value,
            goal_value=0.5,
            tolerance=0.05,
        )

    def test_achieved_exactly_at_upper_tolerance_boundary(self):
        """The effect is achieved when the value falls to ``goal + tolerance`` exactly."""
        assert self._effect(0.55).is_achieved()

    def test_not_achieved_just_above_upper_tolerance_boundary(self):
        """The effect is not achieved while the value stays above ``goal + tolerance``."""
        assert not self._effect(0.56).is_achieved()

    def test_achieved_below_goal(self):
        """The effect stays achieved for any value beneath the goal."""
        assert self._effect(0.1).is_achieved()


# %% pouring effect boundaries


class TestPouringEffectBoundary:
    """Validates ``PouringEffect``'s fill-level reading and achievement boundary."""

    def test_default_property_getter_reads_fill_level(self):
        """Without an explicit getter the effect reads the container's fill level."""
        cup = _cup_with_fill_level(0.8)
        effect = PouringEffect(target_object=cup, goal_value=0.5)
        assert effect.current_value == cup.fill_level

    def test_achieved_exactly_at_upper_tolerance_boundary(self):
        """The effect is achieved when the fill level drops to ``goal + tolerance``."""
        cup = _cup_with_fill_level(0.55)
        effect = PouringEffect(target_object=cup, goal_value=0.5, tolerance=0.05)
        assert effect.is_achieved()

    def test_not_achieved_just_above_upper_tolerance_boundary(self):
        """The effect is not achieved while the fill level exceeds ``goal + tolerance``."""
        cup = _cup_with_fill_level(0.6)
        effect = PouringEffect(target_object=cup, goal_value=0.5, tolerance=0.05)
        assert not effect.is_achieved()

    def test_achieved_below_goal(self):
        """The effect stays achieved once the fill level falls beneath the goal."""
        cup = _cup_with_fill_level(0.2)
        effect = PouringEffect(target_object=cup, goal_value=0.5, tolerance=0.05)
        assert effect.is_achieved()

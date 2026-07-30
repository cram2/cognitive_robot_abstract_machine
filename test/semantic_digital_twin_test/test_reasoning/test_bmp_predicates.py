# %% imports

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from krrood.ormatic.utils import classproperty

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import MissingMotionTrajectoryError
from semantic_digital_twin.physics.physics_model import PhysicsModel
from semantic_digital_twin.reasoning.bmp_predicates import Causes
from semantic_digital_twin.semantic_annotations.effects import PouringEffect
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.effects import Effect
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.motion import Motion, MotionTrajectory
from semantic_digital_twin.world_description.world_entity import Body

# %% test world and physics-model mimic


@dataclass(eq=False)
class _TiltingContainer(HasFillLevel):
    """
    A pourable container attached to its parent by a single tilting DOF.
    """

    @classproperty
    def _parent_connection_type(self):
        return RevoluteConnection


@dataclass
class _RecordedTrajectoryModel(PhysicsModel):
    """
    Physics-model mimic that hands out a pre-recorded trajectory and counts runs.
    """

    trajectory: MotionTrajectory
    """
    The trajectory returned by every :meth:`run` call.
    """

    run_count: int = field(default=0)
    """
    How often :meth:`run` was invoked.
    """

    def run(self, effect: Effect, world: World) -> MotionTrajectory:
        self.run_count += 1
        return self.trajectory


def _world_with_filled_cup(
    initial_fill: float = 1.0,
) -> tuple[World, _TiltingContainer]:
    """
    Builds a minimal world holding a single tilting cup with the given fill level.
    """
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
    cup.initialize_fill_level(world=world, initial_fill=initial_fill)
    return world, cup


def _tilt_ramp(cup: _TiltingContainer, steps: int, target: float) -> MotionTrajectory:
    """
    Builds a trajectory ramping the cup's tilt linearly from zero to ``target``.
    """
    positions = [target * (step + 1) / steps for step in range(steps)]
    return MotionTrajectory(data={cup.root.parent_connection: positions})


# %% causal sufficiency


class TestCauses:
    """
    Validates the causal sufficiency check of the ``Causes`` predicate.
    """

    def test_short_circuits_when_effect_already_achieved(self):
        """
        An already-achieved effect makes the predicate false without ever consulting the
        physics model or generating a trajectory.
        """
        world, cup = _world_with_filled_cup(initial_fill=0.4)
        effect = PouringEffect(target_object=cup, goal_value=0.5)
        model = _RecordedTrajectoryModel(
            trajectory=_tilt_ramp(cup, steps=5, target=1.0)
        )
        motion = Motion(connection=cup.root.parent_connection, motion_model=model)

        assert not Causes(effect=effect, environment=world, motion=motion)()
        assert model.run_count == 0
        assert motion.motion_trajectory is None

    def test_generates_trajectory_from_physics_model(self):
        """
        A motion carrying a physics model but no trajectory gets the generated
        trajectory stored on it.
        """
        world, cup = _world_with_filled_cup()
        tilt_connection = cup.root.parent_connection
        effect = Effect(
            target_object=cup,
            property_getter=lambda container: container.root.parent_connection.position,
            goal_value=1.0,
        )
        trajectory = _tilt_ramp(cup, steps=5, target=1.0)
        model = _RecordedTrajectoryModel(trajectory=trajectory)
        motion = Motion(connection=tilt_connection, motion_model=model)

        assert Causes(effect=effect, environment=world, motion=motion)()
        assert model.run_count == 1
        assert motion.motion_trajectory is trajectory

    def test_physics_stepping_drives_coupled_fill_level(self):
        """
        With a time step set, replaying the tilt trajectory integrates the coupled fill-
        level physics, so the pour drains the cup and achieves the effect.
        """
        world, cup = _world_with_filled_cup()
        effect = PouringEffect(target_object=cup, goal_value=0.5)
        motion = Motion(
            connection=cup.root.parent_connection,
            motion_trajectory=_tilt_ramp(cup, steps=100, target=1.5),
            time_step=0.05,
        )

        assert Causes(effect=effect, environment=world, motion=motion)()

    def test_sandboxed_check_leaves_the_world_state_untouched(self):
        """
        The causal check simulates in a reset context, so neither the tilt nor the fill
        level of the live world changes.
        """
        world, cup = _world_with_filled_cup()
        effect = PouringEffect(target_object=cup, goal_value=0.5)
        motion = Motion(
            connection=cup.root.parent_connection,
            motion_trajectory=_tilt_ramp(cup, steps=100, target=1.5),
            time_step=0.05,
        )

        Causes(effect=effect, environment=world, motion=motion)()

        assert cup.root.parent_connection.position == pytest.approx(0.0)
        assert cup.fill_level == pytest.approx(1.0)

    def test_false_without_motion(self):
        """
        Without any motion there is nothing that could cause the effect.
        """
        world, cup = _world_with_filled_cup()
        effect = PouringEffect(target_object=cup, goal_value=0.5)

        assert not Causes(effect=effect, environment=world, motion=None)()


# %% trajectory replay


class TestCausesReplay:
    """
    Validates the guarded trajectory replay of the ``Causes`` predicate.
    """

    def test_replay_raises_without_motion(self):
        """
        Replaying without a motion raises instead of silently doing nothing.
        """
        world, cup = _world_with_filled_cup()
        effect = PouringEffect(target_object=cup, goal_value=0.5)
        predicate = Causes(effect=effect, environment=world, motion=None)

        with pytest.raises(MissingMotionTrajectoryError):
            predicate.replay()

    def test_replay_raises_without_trajectory(self):
        """
        Replaying a motion that carries no trajectory raises.
        """
        world, cup = _world_with_filled_cup()
        effect = PouringEffect(target_object=cup, goal_value=0.5)
        motion = Motion(connection=cup.root.parent_connection)
        predicate = Causes(effect=effect, environment=world, motion=motion)

        with pytest.raises(MissingMotionTrajectoryError):
            predicate.replay()

    def test_replay_raises_when_trajectory_does_not_track_the_actuator(self):
        """
        Replaying a trajectory that never recorded the motion's own connection raises
        instead of silently replaying nothing.
        """
        world, cup = _world_with_filled_cup()
        effect = PouringEffect(target_object=cup, goal_value=0.5)
        motion = Motion(
            connection=cup.root.parent_connection,
            motion_trajectory=MotionTrajectory(data={cup.fill_connection: [0.5]}),
        )
        predicate = Causes(effect=effect, environment=world, motion=motion)

        with pytest.raises(MissingMotionTrajectoryError):
            predicate.replay()

    def test_replay_applies_the_trajectory_to_the_world(self):
        """
        Replaying moves the actuator through the trajectory, ending at its last
        position.
        """
        world, cup = _world_with_filled_cup()
        effect = PouringEffect(target_object=cup, goal_value=0.5)
        motion = Motion(
            connection=cup.root.parent_connection,
            motion_trajectory=_tilt_ramp(cup, steps=4, target=0.8),
        )
        predicate = Causes(effect=effect, environment=world, motion=motion)

        predicate.replay(step_delay=0.0)

        assert cup.root.parent_connection.position == pytest.approx(0.8)

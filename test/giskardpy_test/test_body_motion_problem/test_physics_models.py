from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from giskardpy.body_motion_problem.container_physics import (
    ContainerManipulationPhysicsModel,
)
from giskardpy.body_motion_problem.giskard_physics_model import GiskardPhysicsModel
from giskardpy.body_motion_problem.pouring_physics import PouringMSCModel
from giskardpy.data_types.exceptions import HandleActuatorMismatchError
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.pouring import PouringTask
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.effects import Effect
from semantic_digital_twin.world_description.motion import MotionTrajectory
from semantic_digital_twin.world_description.world_entity import Body

from ..test_motion_statechart.test_pouring import PourableContainer, world_with_cup

__all__ = ["world_with_cup"]

# %% helpers


def _fill_level_effect(container: PourableContainer, tolerance: float = 0.05) -> Effect:
    """
    Build a fill-level effect targeting the given container.
    """
    return Effect(
        target_object=container,
        property_getter=lambda annotation: annotation.fill_level,
        goal_value=0.6,
        tolerance=tolerance,
    )


@dataclass(eq=False, repr=False)
class ContextRecordingTask(Task):
    """
    Task that records the build context's QP controller configuration.
    """

    recorded_qp_controller_config: Optional[QPControllerConfig] = field(
        default=None, init=False
    )
    """
    The configuration seen at build time, None until built.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Record the context's QP controller configuration and produce no constraints.
        """
        self.recorded_qp_controller_config = context.qp_controller_config
        return NodeArtifacts()


@dataclass
class ContextRecordingPhysicsModel(GiskardPhysicsModel):
    """
    Minimal physics model whose statechart records the executor's build context.
    """

    recording_task: ContextRecordingTask = field(
        default_factory=ContextRecordingTask, init=False
    )
    """
    The task capturing the QP controller configuration during compilation.
    """

    def build_motion_statechart(self, effect: Effect, world: World) -> MotionStatechart:
        """
        Build a statechart holding only the recording task.
        """
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(self.recording_task)
        return motion_statechart

    def _build_motion_trajectory(self, effect: Effect) -> MotionTrajectory:
        """
        No connections are tracked by the recording model.
        """
        return MotionTrajectory({})


# %% qp controller configuration


class TestQPControllerConfigurationReachesExecutor:
    def test_custom_config_is_seen_by_the_compiled_statechart(
        self, world_with_cup
    ) -> None:
        """
        The model's QP controller configuration must reach the executor's build context.
        """
        world, cup = world_with_cup
        custom_config = QPControllerConfig(target_frequency=40, prediction_horizon=9)
        model = ContextRecordingPhysicsModel(
            qp_controller_config=custom_config, timeout=1
        )

        model.run(_fill_level_effect(cup), world)

        assert model.recording_task.recorded_qp_controller_config is custom_config


# %% pouring tolerance


class TestPouringToleranceComesFromEffect:
    def test_fill_level_tolerance_matches_effect_tolerance(
        self, world_with_cup
    ) -> None:
        """
        The pouring task must inherit the effect's tolerance, not a hardcoded default.
        """
        world, cup = world_with_cup
        model = PouringMSCModel(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            tilt_connection=cup.root.parent_connection,
            root_link=world.root,
            tip_link=cup.root,
        )
        effect = _fill_level_effect(cup, tolerance=0.123)

        motion_statechart = model.build_motion_statechart(effect, world)

        [pouring_task] = [
            node for node in motion_statechart.nodes if isinstance(node, PouringTask)
        ]
        assert pouring_task.fill_level_tolerance == 0.123
        assert pouring_task.goal_value == effect.goal_value


# %% container actuator validation


def _drawer_world() -> tuple[World, Body, PrismaticConnection, PrismaticConnection]:
    """
    Build a world with a drawer driven by a prismatic joint, a handle fixed to the
    drawer, and an unrelated second prismatic joint.

    :return:``(world, handle, drawer_joint, unrelated_joint)``.
    """
    world = World()
    with world.modify_world():
        base = Body(name=PrefixedName("base"))
        drawer = Body(name=PrefixedName("drawer"))
        handle = Body(name=PrefixedName("handle"))
        unrelated = Body(name=PrefixedName("unrelated"))
        world.add_body(base)
        world.add_body(drawer)
        world.add_body(handle)
        world.add_body(unrelated)
        drawer_joint = PrismaticConnection.create_with_dofs(
            world=world, parent=base, child=drawer, axis=Vector3.X()
        )
        world.add_connection(drawer_joint)
        world.add_connection(
            FixedConnection.create_with_dofs(world=world, parent=drawer, child=handle)
        )
        unrelated_joint = PrismaticConnection.create_with_dofs(
            world=world, parent=base, child=unrelated, axis=Vector3.X()
        )
        world.add_connection(unrelated_joint)
    return world, handle, drawer_joint, unrelated_joint


class TestContainerActuatorValidation:
    def test_mismatching_actuator_is_rejected(self) -> None:
        """
        An actuator that does not drive the handle would record an unmoved joint.
        """
        world, handle, drawer_joint, unrelated_joint = _drawer_world()
        model = ContainerManipulationPhysicsModel(
            handle=handle, actuator=unrelated_joint, goal_joint_state=0.3
        )
        container = PourableContainer(name=PrefixedName("drawer"), root=handle)

        with pytest.raises(HandleActuatorMismatchError) as error_info:
            model.build_motion_statechart(_fill_level_effect(container), world)

        assert error_info.value.handle_name == str(handle.name)
        assert error_info.value.actuator_name == str(unrelated_joint.name)
        assert error_info.value.handle_connection_name == str(drawer_joint.name)

    def test_matching_actuator_builds_the_open_statechart(self) -> None:
        """
        The handle's own driving joint passes validation and yields the Open statechart.
        """
        world, handle, drawer_joint, _unrelated_joint = _drawer_world()
        model = ContainerManipulationPhysicsModel(
            handle=handle, actuator=drawer_joint, goal_joint_state=0.3
        )
        container = PourableContainer(name=PrefixedName("drawer"), root=handle)

        motion_statechart = model.build_motion_statechart(
            _fill_level_effect(container), world
        )

        assert len(motion_statechart.nodes) == 2

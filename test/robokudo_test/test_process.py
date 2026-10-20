from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from threading import Thread

import numpy as np
import pytest
from py_trees.behaviour import Behaviour
from py_trees.behaviours import Failure, Success
from py_trees.common import Status
from py_trees.composites import Sequence
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Optional

from robokudo.annotators.core import BaseAnnotator
from robokudo.exceptions import (
    MultipleRobotsInWorldError,
    RoboKudoAlreadyRunningError,
    SharedWorldReplacementError,
)
from robokudo.perception_belief_state_context import PerceptionBeliefStateContext
from robokudo.pipeline import Pipeline
from robokudo.process import RoboKudo

from ._object_hypotheses import make_object_hypothesis
from ._perception_world import (
    SharedWorld,
    camera_pose,
    cas_seen_from,
    place_camera_of,
    wait_until,
)

RUN_TIMEOUT = timedelta(seconds=5)
"""
How long ticking a failing engine may take to stop.
"""


@dataclass
class EngineReadingItsWorld:
    """
    An analysis engine that reads the perception world while it builds its tree, as
    engines restoring a stored scene do.
    """

    tree: Behaviour = field(default_factory=lambda: Success(name="Perceive"))
    """
    The tree the engine builds.
    """

    world_seen_while_building: Optional[World] = None
    """
    The perception world the engine found while building its tree.
    """

    def name(self) -> str:
        return type(self).__name__

    def implementation(self) -> Behaviour:
        self.world_seen_while_building = (
            PerceptionBeliefStateContext.from_blackboard().world
        )
        return self.tree


class FindsAnObjectInView(BaseAnnotator):
    """
    Finds one object in front of the camera and ends the pass with ``outcome``.
    """

    def __init__(self, outcome: Status) -> None:
        super().__init__(name="FindsAnObjectInView")
        self.outcome = outcome

    def update(self) -> Status:
        cas = self.get_cas()
        place_camera_of(cas, PerceptionBeliefStateContext.from_blackboard(), camera_pose())
        cas.annotations.append(make_object_hypothesis())
        return self.outcome


@dataclass
class EngineFindingAnObject:
    """
    An analysis engine whose pipeline finds one object per pass.
    """

    outcome: Status = Status.SUCCESS
    """
    How each pass ends.
    """

    def name(self) -> str:
        return type(self).__name__

    def implementation(self) -> Pipeline:
        pipeline = Pipeline("FindingAnObject")
        pipeline.add_child(FindsAnObjectInView(self.outcome))
        return pipeline


@pytest.fixture
def engine() -> EngineReadingItsWorld:
    return EngineReadingItsWorld()


@pytest.fixture
def following_node(ros_default: None) -> Iterator[Node]:
    """
    A node for a RoboKudo started after another one closed.

    Each behaviour tree declares its parameters on its node and never removes them, so a
    later tree needs a node of its own.
    """
    node = Node("following_robokudo")
    yield node
    node.destroy_node()


@pytest.fixture
def joined_robokudo(
    spinning_node: Node, shared_world: SharedWorld, engine: EngineReadingItsWorld
) -> Iterator[RoboKudo]:
    robokudo = RoboKudo.joining_shared_world(node=spinning_node, analysis_engine=engine)
    yield robokudo
    robokudo.close()


# %% perception world


class TestPerceptionWorld:
    def test_engine_builds_its_tree_on_the_given_world(
        self, node: Node, engine: EngineReadingItsWorld
    ) -> None:
        world = World()
        robokudo = RoboKudo(
            node=node,
            analysis_engine=engine,
            belief_state_context=PerceptionBeliefStateContext(world=world),
        )

        robokudo.start(include_gui=False)

        assert engine.world_seen_while_building is world
        robokudo.close()

    def test_without_a_world_perceives_into_an_empty_world_of_its_own(
        self, node: Node, engine: EngineReadingItsWorld
    ) -> None:
        robokudo = RoboKudo(node=node, analysis_engine=engine)

        robokudo.start(include_gui=False)

        assert engine.world_seen_while_building is robokudo.belief_state_context.world
        assert robokudo.belief_state_context.world.is_empty()
        robokudo.close()

    def test_world_of_its_own_is_not_shared(
        self, node: Node, engine: EngineReadingItsWorld
    ) -> None:
        robokudo = RoboKudo(node=node, analysis_engine=engine)

        assert (
            WorldSynchronizer.all_callbacks_of_this_type_from_world(
                robokudo.belief_state_context.world
            )
            == []
        )


# %% joining a shared world


class TestJoiningSharedWorld:
    def test_starts_from_the_shared_world(
        self, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        context = joined_robokudo.belief_state_context

        assert context.is_synchronized
        assert [body.id for body in context.world.bodies] == [
            body.id for body in shared_world.world.bodies
        ]

    def test_changes_of_the_shared_world_reach_perception(
        self, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        table = Body(name=PrefixedName("table"))
        with shared_world.world.modify_world():
            shared_world.world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=shared_world.world.root,
                    child=table,
                    world=shared_world.world,
                )
            )

        assert wait_until(
            lambda: table.id
            in [body.id for body in joined_robokudo.belief_state_context.world.bodies]
        )

    def test_perceived_object_reaches_the_shared_world_at_its_pose(
        self, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        context = joined_robokudo.belief_state_context
        cas = cas_seen_from(context, camera_pose(x=1.0))
        cas.annotations.append(make_object_hypothesis(translation=(0.0, 0.5, 2.0)))

        [body_id] = context.perceived_objects.apply(cas)

        perceived_body = context.world.get_kinematic_structure_entity_by_id(body_id)
        expected_pose = perceived_body.global_transform.to_np()
        assert wait_until(
            lambda: body_id in [body.id for body in shared_world.world.bodies]
            and np.allclose(
                shared_world.world.get_kinematic_structure_entity_by_id(
                    body_id
                ).global_transform.to_np(),
                expected_pose,
            )
        )
        shared_body = shared_world.world.get_kinematic_structure_entity_by_id(body_id)
        assert [shape.scale for shape in shared_body.visual.shapes] == [
            shape.scale for shape in perceived_body.visual.shapes
        ]

    def test_shared_world_cannot_be_replaced(self, joined_robokudo: RoboKudo) -> None:
        with pytest.raises(SharedWorldReplacementError):
            joined_robokudo.belief_state_context.replace_world(World())

    def test_closed_robokudo_no_longer_shares_its_world(
        self, joined_robokudo: RoboKudo
    ) -> None:
        joined_robokudo.close()

        assert not joined_robokudo.belief_state_context.is_synchronized


# %% one RoboKudo per process


class TestOneRoboKudoPerProcess:
    def test_second_robokudo_cannot_start(
        self, node: Node, engine: EngineReadingItsWorld
    ) -> None:
        running = RoboKudo(node=node, analysis_engine=engine)
        running.start(include_gui=False)

        with pytest.raises(RoboKudoAlreadyRunningError):
            RoboKudo(node=node, analysis_engine=EngineReadingItsWorld()).start(
                include_gui=False
            )
        running.close()

    def test_closed_robokudo_makes_room_for_the_next(
        self, node: Node, following_node: Node, engine: EngineReadingItsWorld
    ) -> None:
        RoboKudo(node=node, analysis_engine=engine).start(include_gui=False)
        RoboKudo.from_blackboard().close()
        following = RoboKudo(
            node=following_node, analysis_engine=EngineReadingItsWorld()
        )

        following.start(include_gui=False)

        assert RoboKudo.from_blackboard() is following
        assert (
            PerceptionBeliefStateContext.from_blackboard()
            is following.belief_state_context
        )
        following.close()


# %% robot camera


class TestRobotCamera:
    def test_camera_is_the_default_camera_of_the_robot(
        self, node: Node, engine: EngineReadingItsWorld, tracy_world: World
    ) -> None:
        robot = tracy_world.get_semantic_annotations_by_type(AbstractRobot)[0]

        robokudo = RoboKudo(
            node=node,
            analysis_engine=engine,
            belief_state_context=PerceptionBeliefStateContext(world=tracy_world),
        )

        assert robokudo.camera is robot.get_default_camera()

    def test_world_without_a_robot_has_no_camera(
        self, node: Node, engine: EngineReadingItsWorld
    ) -> None:
        assert RoboKudo(node=node, analysis_engine=engine).camera is None

    def test_camera_of_a_world_with_several_robots_is_ambiguous(
        self, node: Node, engine: EngineReadingItsWorld
    ) -> None:
        world = World()
        first_robot_root = Body(name=PrefixedName("first_robot"))
        second_robot_root = Body(name=PrefixedName("second_robot"))
        with world.modify_world():
            world.add_body(Body(name=PrefixedName("map")))
            for robot_root in (first_robot_root, second_robot_root):
                world.add_connection(
                    Connection6DoF.create_with_dofs(
                        parent=world.root, child=robot_root, world=world
                    )
                )
        MinimalRobot.from_branch_in_world(first_robot_root)
        MinimalRobot.from_branch_in_world(second_robot_root)

        with pytest.raises(MultipleRobotsInWorldError):
            RoboKudo(
                node=node,
                analysis_engine=engine,
                belief_state_context=PerceptionBeliefStateContext(world=world),
            ).camera


# %% ticking


class TestTicking:
    def test_runs_until_the_engine_fails(self, node: Node) -> None:
        failing_tree = Sequence(name="FailingPipeline", memory=True)
        failing_tree.add_child(Failure(name="Fail"))
        robokudo = RoboKudo(
            node=node, analysis_engine=EngineReadingItsWorld(tree=failing_tree)
        )
        robokudo.start(include_gui=False)
        running = Thread(target=robokudo.run, daemon=True)

        running.start()
        running.join(timeout=RUN_TIMEOUT.total_seconds())

        assert not running.is_alive()
        assert failing_tree.children[0].status is Status.FAILURE
        robokudo.close()


# %% objects of a finished pass


class TestObjectsOfAFinishedPass:
    def test_objects_of_a_finished_pass_reach_the_world(self, node: Node) -> None:
        robokudo = RoboKudo(node=node, analysis_engine=EngineFindingAnObject())
        robokudo.start(include_gui=False)

        robokudo.tick()

        context = robokudo.belief_state_context
        [body_id] = context.perceived_objects.beliefs
        assert body_id in [body.id for body in context.world.bodies]
        robokudo.close()

    def test_failed_pass_puts_nothing_into_the_world(self, node: Node) -> None:
        robokudo = RoboKudo(
            node=node, analysis_engine=EngineFindingAnObject(outcome=Status.FAILURE)
        )
        robokudo.start(include_gui=False)

        robokudo.tick()

        assert robokudo.belief_state_context.perceived_objects.beliefs == {}
        robokudo.close()

    def test_engine_without_a_pipeline_puts_nothing_into_the_world(
        self, node: Node, engine: EngineReadingItsWorld
    ) -> None:
        robokudo = RoboKudo(node=node, analysis_engine=engine)
        robokudo.start(include_gui=False)

        robokudo.tick()

        assert robokudo.belief_state_context.perceived_objects.beliefs == {}
        robokudo.close()

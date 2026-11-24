"""
Tests of :class:`robokudo.annotators.world_updating.WorldUpdatingAnnotator`, which puts
the objects of a pass into the world.

How the beliefs of a pass are matched to the objects of the world is tested in
:mod:`test_perception_beliefs`.
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID

import pytest
from py_trees.common import Status
from py_trees.composites import Sequence
from rclpy.node import Node
from semantic_digital_twin.semantic_annotations.mixins import IsPerceivable
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    UnknownObject,
)

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.world_updating import WorldUpdatingAnnotator
from robokudo.cas import CAS
from robokudo.process import RoboKudo

from ._object_hypotheses import make_object_hypothesis
from ._perception_process import (
    EngineRunningABehaviour,
    FindsAnObjectInView,
    put_pass_into_world,
    robokudo_of_its_own,
)
from ._perception_world import camera_pose, cas_seen_from

# %% fixtures


@pytest.fixture
def robokudo(node: Node) -> Iterator[RoboKudo]:
    """
    A RoboKudo perceiving into a world of its own.
    """
    robokudo = robokudo_of_its_own(node, EngineRunningABehaviour())
    yield robokudo
    robokudo.close()


class NotesThePerceivableObjectsOfTheWorld(BaseAnnotator):
    """
    Notes the bodies of the perceivable objects of the world each time it runs.
    """

    def __init__(self) -> None:
        super().__init__(name="NotesThePerceivableObjectsOfTheWorld")
        self.noted_body_ids: list[list[UUID]] = []
        """
        The ids of the bodies of the perceivable objects, once per run.
        """

    def update(self) -> Status:
        self.noted_body_ids.append(
            [
                annotation.root.id
                for annotation in self.world.get_semantic_annotations_by_type(
                    IsPerceivable
                )
            ]
        )
        return Status.SUCCESS


# %% updating the world in the middle of a pass


class TestUpdatingTheWorldInTheMiddleOfAPass:
    """
    A pass whose pipeline goes on after it updated the world.
    """

    def test_objects_reach_the_world_before_the_rest_of_the_pass_runs(
        self, *, node: Node
    ) -> None:
        notes = NotesThePerceivableObjectsOfTheWorld()
        robokudo = robokudo_of_its_own(
            node,
            EngineRunningABehaviour(
                behaviour=Sequence(
                    name="FindUpdateAndNote",
                    memory=True,
                    children=[
                        FindsAnObjectInView(outcome=Status.SUCCESS),
                        WorldUpdatingAnnotator(),
                        notes,
                    ],
                )
            ),
        )
        robokudo.start(include_gui=False)

        robokudo.tick_tree()

        [unknown_object] = robokudo.world.get_semantic_annotations_by_type(
            UnknownObject
        )
        assert notes.noted_body_ids == [[unknown_object.root.id]]
        robokudo.close()

    def test_pass_without_a_camera_pose_goes_on(self, *, robokudo: RoboKudo) -> None:
        status = put_pass_into_world(robokudo, CAS(), make_object_hypothesis())

        assert status == Status.SUCCESS


# %% class of a new object


class TestClassOfANewObject:
    """
    The class a new object of the world gets from the classification of its hypothesis.
    """

    def test_object_classified_less_confidently_than_required_becomes_an_unknown_object(
        self, *, robokudo: RoboKudo
    ) -> None:
        world_updating = WorldUpdatingAnnotator(minimum_class_confidence=0.9)

        put_pass_into_world(
            robokudo,
            cas_seen_from(robokudo.world, camera_pose()),
            make_object_hypothesis(class_label="cup", class_confidence=0.8),
            world_updating=world_updating,
        )

        assert [
            type(annotation)
            for annotation in robokudo.world.get_semantic_annotations_by_type(
                IsPerceivable
            )
        ] == [UnknownObject]

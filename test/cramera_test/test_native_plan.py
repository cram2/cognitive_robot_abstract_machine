"""
Native plan lifecycle values at the live and recording boundaries.
"""

from __future__ import annotations

import json

import pytest

from coraplex.datastructures.enums import Arms
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import ActionNode, MotionNode, PlanNode
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from giskardpy.motion_statechart.data_types import LifeCycleValues
from krrood.entity_query_language.factories import inference
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live.bridge import Bridge
from cramera.recording_fields import SceneField

from .dataset.plan_metadata import AnnotationTargetMotion

# %% lifecycle publication


@pytest.mark.parametrize("status", list(LifeCycleValues))
def test_plan_snapshot_keeps_the_native_lifecycle(status: LifeCycleValues) -> None:
    """
    Keep native states in memory and their names in serialized snapshots.

    :param status: The native lifecycle state to publish.
    """
    plan = Plan()
    node = PlanNode(status=status)
    plan.add_node(node)
    bridge = Bridge()

    bridge.begin_plan(plan)

    assert bridge.plan_state.nodes[0].status is status
    assert (
        json.loads(json.dumps(bridge.get_plan()))["nodes"][0]["status"] == status.name
    )
    assert (
        json.loads(json.dumps(bridge.plan_state.recorded_trees()))[0]["status"]
        == status.name
    )


@pytest.mark.parametrize("status", list(LifeCycleValues))
def test_parent_lifecycle_is_independent_of_finished_children(
    status: LifeCycleValues,
) -> None:
    """
    A parent's lifecycle remains its own when children have completed.

    :param status: The parent's current lifecycle, including a reset.
    """
    plan = Plan()
    parent = PlanNode(status=LifeCycleValues.RUNNING)
    plan.add_edge(parent, PlanNode(status=LifeCycleValues.SUCCEEDED))
    bridge = Bridge()
    bridge.begin_plan(plan)

    parent.status = status
    bridge.snapshot_plan()

    assert bridge.plan_state.nodes[0].status is status
    assert bridge.plan_state.nodes[0].derived is False


# %% native designator metadata


@pytest.mark.parametrize("gripper", list(Arms))
def test_designator_description_uses_native_parameter_verbalization(
    gripper: Arms,
) -> None:
    """
    Publish native wording for the selected grippers and their requested state.

    :param gripper: The native gripper selection to describe.
    """
    action = SetGripperAction(gripper=gripper, motion=GripperState.CLOSE)
    plan = Plan()
    plan.add_node(ActionNode(designator=action))
    bridge = Bridge()

    bridge.begin_plan(plan)

    [entry] = bridge.get_plan()["nodes"]
    assert entry[SceneField.DESCRIPTION] == verbalize_expression(
        inference(type(action))(**action.designator_parameter)
    )
    assert "arm" not in entry


@pytest.mark.parametrize("arm", list(Arms))
def test_native_arm_selection_is_verbalized(arm: Arms) -> None:
    """
    Describe every native arm selection, including the left arm and both arms.

    :param arm: The native arm selection to publish.
    """
    plan = Plan()
    action = ParkArmsAction(arm=arm)
    plan.add_node(ActionNode(designator=action))
    bridge = Bridge()

    bridge.begin_plan(plan)

    assert bridge.plan_state.nodes[0].description == verbalize_expression(
        inference(type(action))(**action.designator_parameter)
    )


def test_arm_enum_is_not_mistaken_for_a_target_body() -> None:
    """
    An enum's name does not create an object reference with the same name.
    """
    arm = Arms.RIGHT
    body = Body(name=PrefixedName(arm.name))
    plan = Plan()
    plan.add_node(ActionNode(designator=ParkArmsAction(arm=arm)))
    bridge = Bridge()
    bridge.publish_bodies({str(body.name): body})

    bridge.begin_plan(plan)

    assert bridge.plan_state.nodes[0].target is None


def test_native_annotation_resolves_its_published_body_name() -> None:
    """
    A semantic annotation retains the target match through its native name.
    """
    body = Body(name=PrefixedName("handle", prefix="world"))
    annotation = Handle(root=body, name=body.name)
    plan = Plan()
    plan.add_node(
        MotionNode(designator=AnnotationTargetMotion(target_annotation=annotation))
    )
    bridge = Bridge()
    bridge.publish_bodies({body.name.name: body})

    bridge.begin_plan(plan)

    assert bridge.plan_state.nodes[0].target == body.name.name

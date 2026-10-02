"""
Keep recorded plan inspection separate from the active execution stream.
"""

import json
from pathlib import Path

from coraplex.datastructures.enums import Arms
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import ActionNode
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from krrood.entity_query_language.factories import inference
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression

from cramera import paths
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.knowledge.views.plan_tree import PlanViewPayload
from cramera.live.bridge import Bridge
from cramera.recording_fields import SceneField


# %% recorded and live plan sources
def test_recorded_plan_does_not_subscribe_to_another_run(fixture_scene: Path) -> None:
    """
    A saved scene retains its plan while another live session is available.

    :param fixture_scene: Isolated recorded scene and architecture fixture.
    """
    payload = PlanViewPayload.of_tab(EpisodeKnowledgeBase.of_scene("fixture"))

    assert "live" not in payload.panel_options()


def test_live_plan_subscribes_to_the_execution_stream(fixture_scene: Path) -> None:
    """
    The reserved live scene requests plan updates from the bridge.

    :param fixture_scene: Isolated data directory and architecture fixture.
    """
    knowledge = EpisodeKnowledgeBase(scene_name=paths.LIVE_SCENE_NAME)

    assert PlanViewPayload.of_tab(knowledge).panel_options()["live"] == "plan"


def test_recorded_plan_keeps_native_designator_description(fixture_scene: Path) -> None:
    """
    Display the native parameter wording after saving and reopening a plan tree.

    :param fixture_scene: Isolated recorded scene and architecture fixture.
    """
    action = ParkArmsAction(arm=Arms.BOTH)
    plan = Plan()
    plan.add_node(ActionNode(designator=action))
    bridge = Bridge()
    bridge.begin_plan(plan)
    scene_file = fixture_scene / "scenes" / "fixture" / "scene.json"
    scene = json.loads(scene_file.read_text())
    scene[SceneField.PLAN_TREES] = bridge.plan_state.recorded_trees()
    scene_file.write_text(json.dumps(scene))

    payload = PlanViewPayload.of_tab(EpisodeKnowledgeBase.of_scene("fixture"))

    [detail] = payload.details.values()
    description = verbalize_expression(
        inference(type(action))(**action.designator_parameter)
    )
    assert description in detail.lines
    [node] = payload.nodes
    assert description in node.title
    assert node.label == type(action).__name__.removesuffix("Action")

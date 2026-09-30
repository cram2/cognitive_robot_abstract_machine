"""
Keep recorded plan inspection separate from the active execution stream.
"""

from pathlib import Path

from cramera import paths
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.knowledge.views.plan_tree import PlanViewPayload


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

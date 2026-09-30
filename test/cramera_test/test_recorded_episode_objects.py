"""
Recorded transport objects remain queryable by their serialized references.
"""

from cramera import paths
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.live.bridge import Bridge

from .test_recording_save_transaction import recorded_bridge


# %% recorded object references


def test_action_episode_resolves_the_recorded_object_key(
    recorded_bridge: Bridge,
) -> None:
    """
    A real recording's mesh key resolves to the same object its scene describes.

    :param recorded_bridge: A finalized recording carrying a moving milk object.
    """
    knowledge = EpisodeKnowledgeBase.of_scene(paths.RECORDING_SCENE_NAME)
    [object_entry] = recorded_bridge.object_metadata
    recorded_object = next(
        item for item in knowledge.objects if item.name == object_entry.id
    )

    assert knowledge.episodes[-1].picks is recorded_object

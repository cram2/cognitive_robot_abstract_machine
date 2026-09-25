"""
Native body answers retain their names while selecting existing viewer objects.
"""

from pathlib import Path

import pytest

from krrood.entity_query_language.factories import an, entity, variable
from semantic_digital_twin.world_description.world_entity import Body

from cramera.knowledge.enums import SceneEntityPrefix
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.knowledge.views.kinematics import UrdfViewPayload
from cramera.live.bridge import Bridge
from cramera.live.live_bundle import build_live_scene
from cramera.onboard.world_to_urdf import UrdfDocument

from .test_live_bundle import attached_bridge
from .test_live_query import GrowingRecordSource, make_record


# %% scene identifiers
def test_prefixed_loose_body_highlights_its_published_key() -> None:
    """
    An object's canonical name stays visible while its published mesh key glows.
    """
    bridge = attached_bridge()
    [key] = bridge.object_keys()
    body = bridge.object_body(key)
    answer = bridge.run_query(an(entity(variable(Body, domain=[body]))))

    assert answer.rows[0]["__entity__"] == str(body.name)
    assert answer.highlight == [key]


def test_native_robot_body_highlights_its_urdf_link() -> None:
    """
    A native robot body's canonical name selects the existing URDF link node.
    """
    bridge = attached_bridge(with_robot=True)
    body = bridge.robot.root
    answer = bridge.run_query(an(entity(variable(Body, domain=[body]))))

    assert answer.rows[0]["__entity__"] == str(body.name)
    assert answer.highlight == [SceneEntityPrefix.URDF_LINK + str(body.name)]


def test_legacy_highlight_identifiers_remain_unchanged() -> None:
    """
    Existing object keys, display names, and URDF link identifiers pass through.
    """
    bridge = attached_bridge(with_robot=True)
    [key] = bridge.object_keys()
    identifiers = [
        key,
        Path(key).stem,
        SceneEntityPrefix.URDF_LINK + str(bridge.robot.root.name),
    ]
    source = GrowingRecordSource(records=[make_record(name) for name in identifiers])
    bridge.register_query_source(source.knowledge, source.title(), source.presets)

    answer = bridge.run_query(source.presets()[0].code)

    assert answer.highlight == sorted(identifiers)


def test_live_body_highlight_matches_the_exported_robot_link(
    fixture_scene: Path,
) -> None:
    """
    A native body query selects the same node as the generated URDF graph.

    :param fixture_scene: The existing scene fixture's isolated storage and metadata.
    """
    bridge = attached_bridge(with_robot=True)
    scene_name = build_live_scene(bridge)
    view = UrdfViewPayload.of_tab(EpisodeKnowledgeBase.of_scene(scene_name))
    body = bridge.robot.root

    answer = bridge.run_query(an(entity(variable(Body, domain=[body]))))

    link_node = next(node for node in view.nodes if node.label == str(body.name))
    assert answer.highlight == [link_node.id]
    assert link_node.id in view.details
    assert any(node.label == UrdfDocument.SYNTHESIZED_ROOT_LINK for node in view.nodes)


@pytest.mark.parametrize("attached", [False, True])
def test_unknown_highlights_survive_without_duplicate_identifiers(
    attached: bool,
) -> None:
    """
    Unrecognized names survive conversion whether a world is attached or not.

    :param attached: Whether a world and robot are available for conversion.
    """
    bridge = attached_bridge(with_robot=True) if attached else Bridge()
    names = ["unpublished/body", "another/body", "unpublished/body"]

    assert bridge._resolve_highlights(names) == sorted(set(names))

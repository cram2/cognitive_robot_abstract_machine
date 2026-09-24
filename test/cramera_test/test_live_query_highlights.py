"""
Native body answers retain their names while selecting existing viewer objects.
"""

from pathlib import Path

from krrood.entity_query_language.factories import an, entity, variable
from semantic_digital_twin.world_description.world_entity import Body

from cramera.knowledge.views.kinematics import UrdfViewPayload

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
    assert answer.highlight == [UrdfViewPayload.link_id(str(body.name))]


def test_legacy_highlight_identifiers_remain_unchanged() -> None:
    """
    Existing object keys, display names, and URDF link identifiers pass through.
    """
    bridge = attached_bridge(with_robot=True)
    [key] = bridge.object_keys()
    identifiers = [
        key,
        Path(key).stem,
        UrdfViewPayload.link_id(str(bridge.robot.root.name)),
    ]
    source = GrowingRecordSource(records=[make_record(name) for name in identifiers])
    bridge.register_query_source(source.knowledge, source.title(), source.presets)

    answer = bridge.run_query(source.presets()[0].code)

    assert answer.highlight == sorted(identifiers)

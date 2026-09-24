"""
Session configuration controls publication without changing other viewers.
"""

from unittest.mock import patch

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

from cramera.config import CrameraConfig
from cramera.live.bridge import Bridge


# %% session defaults
def test_configured_robot_root_is_excluded_from_loose_objects() -> None:
    """
    Use the reserved key consistently in the catalog and public status.
    """
    configuration = CrameraConfig(robot_base_key="robot-root")
    bridge = Bridge(configuration=configuration)
    root = Body(name=PrefixedName("base_link"))
    movable = Body(name=PrefixedName("object"))

    bridge.publish_bodies(
        {configuration.robot_base_key: root, str(movable.name): movable}
    )

    assert bridge.object_keys() == [str(movable.name)]
    assert bridge.status()["objects"] == [str(movable.name)]
    assert bridge.overlay_bodies() == [movable]
    assert [entry["key"] for entry in bridge.object_catalog()] == [str(movable.name)]


def test_configured_placeholder_geometry_is_isolated_to_its_session() -> None:
    """
    Render each shapeless body with its own session's configured dimensions.
    """
    configuration = CrameraConfig(default_object_size=(0.2, 0.3, 0.4))
    configured = Bridge(configuration=configuration)
    default = Bridge()
    body = Body(name=PrefixedName("object"))

    for bridge in (configured, default):
        bridge.publish_bodies({str(body.name): body})
        [shape] = bridge.object_metadata[0].shapes
        assert shape.scale == Scale(*bridge.configuration.default_object_size)
        assert bridge.object_catalog()[0]["shapes"][0]["size"] == list(
            bridge.configuration.default_object_size
        )

    assert configured.configuration != default.configuration


@pytest.mark.parametrize("interval, expected_bind_count", [(1.0, 1), (3.0, 0)])
def test_world_discovery_respects_the_configured_interval(
    interval: float, expected_bind_count: int
) -> None:
    """
    Refresh world bindings only after the configured interval has elapsed.

    :param interval: Time between periodic world discoveries in seconds.
    :param expected_bind_count: Discoveries expected two seconds after the last bind.
    """
    bridge = Bridge(configuration=CrameraConfig(rebind_interval_seconds=interval))
    bridge.attach(World())
    next_snapshot_time = bridge._last_bind_time + 2.0

    with (
        patch("cramera.live.bridge.time.time", return_value=next_snapshot_time),
        patch.object(bridge, "bind", wraps=bridge.bind) as bind,
    ):
        bridge.snapshot()

    assert bind.call_count == expected_bind_count

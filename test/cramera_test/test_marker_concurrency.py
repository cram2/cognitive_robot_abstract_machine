"""
Marker snapshots remain coherent while ROS and HTTP threads change the stores.
"""

from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from typing_extensions import Any

import pytest

from cramera.live.bridge import Bridge
from cramera.live.markers import MarkerEntry

from .test_live_markers import MimicMarker, RecordingListener


# %% marker topics
class MarkerTopic(StrEnum):
    """
    Independent topics whose updates may overlap one snapshot.
    """

    FIRST = "/first"
    SECOND = "/second"


# %% concurrent publication
@pytest.mark.parametrize("remove_topic", [False, True])
def test_changes_during_marker_publication_reach_next_snapshot(
    monkeypatch: pytest.MonkeyPatch, remove_topic: bool
) -> None:
    """
    Concurrent inserts and topic removals cannot invalidate snapshot iterators.

    :param monkeypatch: Temporarily schedule a writer during pose resolution.
    :param remove_topic: Whether the HTTP thread drops a topic instead of ROS adding.
    """
    bridge = Bridge(marker_listener=RecordingListener(topics=list(MarkerTopic)))
    first = MimicMarker(id=1)
    second = MimicMarker(id=2)
    bridge.observe_ros_markers(MarkerTopic.FIRST, [first])
    bridge.observe_ros_markers(MarkerTopic.SECOND, [second])
    build_payload = bridge._marker_payload

    with ThreadPoolExecutor(max_workers=1) as writer:

        def publish_with_concurrent_change(
            topic: str, entry: MarkerEntry
        ) -> dict[str, Any]:
            """
            Finish an independent writer before the snapshot continues iterating.
            """
            if topic == MarkerTopic.FIRST:
                if remove_topic:
                    writer.submit(
                        bridge.set_marker_topic, MarkerTopic.SECOND, False
                    ).result(timeout=5)
                else:
                    writer.submit(
                        bridge.observe_ros_markers, MarkerTopic.FIRST, [second]
                    ).result(timeout=5)
            return build_payload(topic, entry)

        monkeypatch.setattr(bridge, "_marker_payload", publish_with_concurrent_change)
        bridge._refresh_marker_state()

    first_snapshot = bridge.get_markers()
    assert [entry["id"] for entry in first_snapshot["markers"]] == [first.id, second.id]
    monkeypatch.setattr(bridge, "_marker_payload", build_payload)
    bridge._refresh_marker_state()
    expected = [first.id] if remove_topic else [first.id, second.id, second.id]
    assert [entry["id"] for entry in bridge.get_markers()["markers"]] == expected
    assert bridge.get_markers()["version"] == first_snapshot["version"] + 1


def test_replacing_topic_during_publication_does_not_lose_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Removing and replacing a topic cannot reuse a previously published revision.

    :param monkeypatch: Schedule the topic replacement during pose resolution.
    """
    bridge = Bridge(marker_listener=RecordingListener(topics=list(MarkerTopic)))
    first = MimicMarker(id=1)
    replacement = MimicMarker(id=2)
    bridge.observe_ros_markers(MarkerTopic.FIRST, [first])
    build_payload = bridge._marker_payload

    with ThreadPoolExecutor(max_workers=1) as writer:

        def publish_with_replacement(topic: str, entry: MarkerEntry) -> dict[str, Any]:
            """
            Replace the only topic while its previous payload is being built.
            """
            writer.submit(bridge.set_marker_topic, MarkerTopic.FIRST, False).result(
                timeout=5
            )
            writer.submit(
                bridge.observe_ros_markers, MarkerTopic.SECOND, [replacement]
            ).result(timeout=5)
            return build_payload(topic, entry)

        monkeypatch.setattr(bridge, "_marker_payload", publish_with_replacement)
        bridge._refresh_marker_state()

    monkeypatch.setattr(bridge, "_marker_payload", build_payload)
    bridge._refresh_marker_state()
    assert [entry["id"] for entry in bridge.get_markers()["markers"]] == [
        replacement.id
    ]

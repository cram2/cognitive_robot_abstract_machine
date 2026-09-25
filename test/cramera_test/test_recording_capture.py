"""
Recorded poses remain independent of mutable world snapshots.
"""

from cramera.live.recording import Recording

from .test_live_recording import snapshot


# %% snapshot ownership


def test_captured_base_pose_is_independent_of_source_mutations() -> None:
    """
    Editing a published snapshot cannot change an already captured robot pose.
    """
    recording = Recording()
    recording.start()
    base = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    expected = list(base)
    recording.append(snapshot(base=base))

    base[0] += 1.0

    [frame] = recording.stop()
    assert frame.base == expected

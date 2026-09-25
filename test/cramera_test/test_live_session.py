"""
Recording ownership and cleanup across live visualization sessions.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from semantic_digital_twin.world import World

from cramera.live import visualization
from cramera.live.recording import RecordingState
from cramera.live.visualization import LiveVisualization


# %% session fixtures


@pytest.fixture()
def session_dependencies(monkeypatch):
    """
    Replace external listeners and capture recording finalization.
    """
    server = Mock()
    finalize = Mock()
    register = Mock()
    unregister = Mock()
    monkeypatch.setattr(visualization, "serve", Mock(return_value=server))
    monkeypatch.setattr(visualization, "finalize_recording", finalize)
    monkeypatch.setattr(visualization.atexit, "register", register)
    monkeypatch.setattr(visualization.atexit, "unregister", unregister)
    monkeypatch.setattr(
        visualization.RosMarkerListener, "start_if_available", Mock(return_value=None)
    )
    return server, finalize, register, unregister


# %% independent session ownership


class TestSessionOwnership:
    """
    Each visualization owns the callbacks and capture of its world.
    """

    def test_default_sessions_have_independent_bridges(self):
        first = LiveVisualization(world=World())
        second = LiveVisualization(world=World())

        assert first.bridge is not second.bridge

    def test_starting_twice_preserves_the_active_capture(self, session_dependencies):
        session = LiveVisualization(world=World()).start()
        recording = session.bridge.recording
        callback = session.state_sync

        session.start()

        assert session.bridge.recording is recording
        assert session.state_sync is callback
        assert visualization.serve.call_count == 1

    def test_stop_finalizes_the_owned_capture_and_closes_server(
        self, session_dependencies
    ):
        server, finalize, _, unregister = session_dependencies
        session = LiveVisualization(world=World()).start()
        recording = session.bridge.recording
        callback = session.state_sync

        session.stop()

        finalize.assert_called_once_with(session.bridge, recording)
        server.shutdown.assert_called_once_with()
        server.server_close.assert_called_once_with()
        assert callback not in session.world.state.state_change_callbacks
        unregister.assert_called_once()

    def test_stopping_twice_finalizes_only_once(self, session_dependencies):
        _, finalize, _, _ = session_dependencies
        session = LiveVisualization(world=World()).start()

        session.stop()
        session.stop()

        assert finalize.call_count == 1

    def test_restart_begins_a_new_capture_after_finalizing_previous(
        self, session_dependencies
    ):
        _, finalize, _, _ = session_dependencies
        session = LiveVisualization(world=World()).start()
        previous = session.bridge.recording
        session.stop()

        session.start()

        assert session.bridge.recording is not previous
        assert session.bridge.recording.state is RecordingState.RECORDING
        finalize.assert_called_once_with(session.bridge, previous)

    def test_exit_callback_retains_its_original_capture(self, session_dependencies):
        _, finalize, register, _ = session_dependencies
        session = LiveVisualization(world=World()).start()
        recording = session.bridge.recording
        exit_callback, *arguments = register.call_args.args

        session.bridge.recording = None
        exit_callback(*arguments)

        finalize.assert_called_once_with(session.bridge, recording)

    def test_failed_http_start_releases_resources_and_can_retry(
        self, session_dependencies, monkeypatch
    ):
        server, _, _, unregister = session_dependencies
        listener = Mock()
        monkeypatch.setattr(
            visualization.RosMarkerListener,
            "start_if_available",
            Mock(return_value=listener),
        )
        visualization.serve.side_effect = OSError("address in use")
        session = LiveVisualization(world=World())
        previous_callbacks = list(session.world.state.state_change_callbacks)

        with pytest.raises(OSError):
            session.start()

        assert session.world.state.state_change_callbacks == previous_callbacks
        assert session.bridge.recording is None
        assert session.state_sync is None
        assert session.model_sync is None
        listener.stop.assert_called_once_with()
        unregister.assert_called_once()
        visualization.serve.side_effect = None

        session.start()

        assert session.bridge.live_server is server

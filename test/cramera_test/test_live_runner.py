"""
Unit tests for :func:`cramera.live.runner.start`'s control flow.

``hooks.install_*`` monkey-patches coraplex/giskardpy classes process-globally with no
uninstall (see :meth:`cramera.live.bridge.Bridge.claim_hook`), so these tests replace
them with no-ops rather than calling them for real, and substitute a fresh
:class:`Bridge` for :data:`cramera.live.runner.BRIDGE` so no test touches or dirties
the real process singleton.
"""

from __future__ import annotations

from semantic_digital_twin.world import World

from cramera.live import runner
from cramera.live.bridge import Bridge


def install_no_op_hooks(monkeypatch):
    """
    Replace every hook-installing call ``start()`` makes with a no-op.
    """
    monkeypatch.setattr(runner.hooks, "install_mesh_hook", lambda: None)
    monkeypatch.setattr(runner.hooks, "install_plan_hooks", lambda: None)
    monkeypatch.setattr(runner.hooks, "install_tick_hook", lambda: None)


class TestStart:
    def test_reuses_the_running_server_without_calling_serve_again(self, monkeypatch):
        bridge = Bridge()
        sentinel_server = object()
        bridge.live_server = sentinel_server
        monkeypatch.setattr(runner, "BRIDGE", bridge)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("serve() must not be called when already running")

        monkeypatch.setattr(runner, "serve", fail_if_called)

        assert runner.start() is sentinel_server

    def test_start_installs_hooks_and_serves_the_bound_bridge(self, monkeypatch):
        bridge = Bridge()
        monkeypatch.setattr(runner, "BRIDGE", bridge)
        install_no_op_hooks(monkeypatch)
        sentinel_server = object()
        serve_calls = []

        def fake_serve(passed_bridge, port):
            serve_calls.append((passed_bridge, port))
            return sentinel_server

        monkeypatch.setattr(runner, "serve", fake_serve)

        result = runner.start(port=1234)

        assert serve_calls == [(bridge, 1234)]
        assert result is sentinel_server
        assert bridge.live_server is sentinel_server

    def test_start_binds_and_snapshots_the_given_world(self, monkeypatch):
        bridge = Bridge()
        monkeypatch.setattr(runner, "BRIDGE", bridge)
        install_no_op_hooks(monkeypatch)
        monkeypatch.setattr(runner, "serve", lambda passed_bridge, port: object())
        world = World()

        runner.start(world=world)

        assert bridge.world is world
        assert bridge.robot is None  # an empty world has no AbstractRobot

    def test_start_without_a_world_leaves_the_bridge_unbound(self, monkeypatch):
        """
        ``world=None`` is a real reachable state: the bridge attaches to the executing
        world on the first executor tick instead.
        """
        bridge = Bridge()
        monkeypatch.setattr(runner, "BRIDGE", bridge)
        install_no_op_hooks(monkeypatch)
        monkeypatch.setattr(runner, "serve", lambda passed_bridge, port: object())

        runner.start()

        assert bridge.world is None

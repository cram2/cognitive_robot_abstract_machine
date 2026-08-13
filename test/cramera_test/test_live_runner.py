"""
Unit tests for the live runner: the library entry point and the demo wrapper CLI.
"""

from __future__ import annotations

import os
import runpy
import signal
import sys
from dataclasses import dataclass, field

from semantic_digital_twin.world import World
from typing_extensions import Any, Dict, List

from cramera.live import runner, visualization


@dataclass
class StartedVisualization:
    """
    A visualization recorder standing in for ``LiveVisualization``.
    """

    world: World
    port: int
    started: bool = False

    def start(self):
        self.started = True
        return self


class TestStart:
    def test_start_serves_the_given_world(self, monkeypatch):
        monkeypatch.setattr(visualization, "LiveVisualization", StartedVisualization)
        world = World()

        result = runner.start(world, port=1234)

        assert isinstance(result, StartedVisualization)
        assert result.world is world
        assert result.port == 1234
        assert result.started is True


class TestMain:
    def test_main_preselects_the_cramera_backend_and_runs_the_demo(self, monkeypatch):
        run_calls: List[Dict[str, Any]] = []
        monkeypatch.delenv(runner.VISUALIZATION_BACKEND_VARIABLE, raising=False)
        monkeypatch.setattr(sys, "argv", ["cramera-live", "/demos/demo.py"])
        monkeypatch.setattr(
            runpy,
            "run_path",
            lambda path, run_name: run_calls.append(
                {"path": path, "run_name": run_name}
            ),
        )
        monkeypatch.setattr(signal, "pause", lambda: None)

        runner.main()

        assert os.environ[runner.VISUALIZATION_BACKEND_VARIABLE] == "cramera"
        assert run_calls == [{"path": "/demos/demo.py", "run_name": "__main__"}]
        assert "/demos" in sys.path

    def test_main_keeps_an_explicit_backend_choice(self, monkeypatch):
        """
        ``CORAPLEX_VISUALIZATION=rerun cramera-live demo.py`` is a deliberate override
        and must survive.
        """
        monkeypatch.setenv(runner.VISUALIZATION_BACKEND_VARIABLE, "rerun")
        monkeypatch.setattr(sys, "argv", ["cramera-live", "/demos/demo.py"])
        monkeypatch.setattr(runpy, "run_path", lambda path, run_name: None)
        monkeypatch.setattr(signal, "pause", lambda: None)

        runner.main()

        assert os.environ[runner.VISUALIZATION_BACKEND_VARIABLE] == "rerun"

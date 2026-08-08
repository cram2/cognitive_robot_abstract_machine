"""
Unit tests for the live hooks' wrapper methods.

Exercised against mimics of the CRAM interfaces and of the bridge itself, so no coraplex
or giskardpy import is needed and no real world binding happens. What is covered is each
wrapper's own contract: when it forwards to the bridge, when it falls through to the
original call, and how it behaves when the bridge itself misbehaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from typing_extensions import Any, List, Optional, Tuple

from cram_viz.live.bridge import TaskStatusName
from cram_viz.live.hooks import LiveHooks


# %% mimics of the interfaces the hooks read
@dataclass
class FakeBridge:
    """
    Records what a hook forwards to it, standing in for a real :class:`Bridge`.
    """

    world: Optional[Any] = None
    attached: List[Any] = field(default_factory=list)
    observed_charts: List[Any] = field(default_factory=list)
    began_plans: List[Any] = field(default_factory=list)
    bound_motion_groups: List[Any] = field(default_factory=list)
    frozen_motion_groups: List[Tuple[Any, Any]] = field(default_factory=list)
    remembered_mesh_files: List[str] = field(default_factory=list)
    raise_on_observe_tick: bool = False

    def attach(self, world: Any) -> None:
        self.attached.append(world)
        self.world = world

    def observe_tick(self, chart: Any) -> None:
        if self.raise_on_observe_tick:
            raise RuntimeError("bridge misbehaved")
        self.observed_charts.append(chart)

    def begin_plan(self, plan: Any) -> None:
        self.began_plans.append(plan)

    def bind_motion_group(self, executable: Any) -> None:
        self.bound_motion_groups.append(executable)

    def freeze_motion_group(self, executable: Any, status: Any) -> None:
        self.frozen_motion_groups.append((executable, status))

    def remember_mesh_file(self, file_path: str) -> None:
        self.remembered_mesh_files.append(file_path)


@dataclass
class FakeExecutorContext:
    """
    The part of ``Executor.context`` the tick hook reads.
    """

    world: Any


@dataclass
class FakeExecutor:
    """
    A giskardpy executor, of which the tick hook reads only its context and chart.
    """

    context: FakeExecutorContext
    motion_statechart: Any = None


@dataclass
class FakeMeshParser:
    """
    A mesh parser, of which the mesh hook reads only its file path.
    """

    file_path: str


# %% tick hook
class TestObserveTick:
    def test_the_first_tick_attaches_the_world(self):
        bridge = FakeBridge()
        hooks = LiveHooks(bridge=bridge)
        executor = FakeExecutor(
            context=FakeExecutorContext(world="the-world"), motion_statechart="chart"
        )

        result = hooks._observe_tick(lambda executor: "ticked", executor)

        assert result == "ticked"
        assert bridge.attached == ["the-world"]
        assert bridge.observed_charts == ["chart"]

    def test_an_already_bound_world_is_not_reattached(self):
        bridge = FakeBridge(world="already-bound")
        hooks = LiveHooks(bridge=bridge)
        executor = FakeExecutor(
            context=FakeExecutorContext(world="the-world"), motion_statechart="chart"
        )

        hooks._observe_tick(lambda executor: None, executor)

        assert bridge.attached == []

    def test_a_bridge_failure_does_not_stop_the_tick(self):
        """
        A visualization bug must never take the robot demo down.
        """
        bridge = FakeBridge(world="already-bound", raise_on_observe_tick=True)
        hooks = LiveHooks(bridge=bridge)
        executor = FakeExecutor(
            context=FakeExecutorContext(world="the-world"), motion_statechart="chart"
        )

        result = hooks._observe_tick(lambda executor: "ticked", executor)

        assert result == "ticked"


# %% plan hook
class TestBeginPlan:
    def test_the_plan_is_captured_before_it_performs(self):
        bridge = FakeBridge()
        hooks = LiveHooks(bridge=bridge)
        order = []

        def original(plan: Any) -> str:
            order.append(plan)
            return "performed"

        result = hooks._begin_plan(original, "the-plan")

        assert bridge.began_plans == ["the-plan"]
        assert order == ["the-plan"]
        assert result == "performed"


# %% motion-group hook
class TestTrackMotionGroup:
    def test_a_successful_execution_freezes_succeeded(self):
        bridge = FakeBridge()
        hooks = LiveHooks(bridge=bridge)

        result = hooks._track_motion_group(lambda executable: "ok", "the-executable")

        assert bridge.bound_motion_groups == ["the-executable"]
        assert bridge.frozen_motion_groups == [
            ("the-executable", TaskStatusName.SUCCEEDED)
        ]
        assert result == "ok"

    def test_a_failed_execution_freezes_failed_and_reraises(self):
        bridge = FakeBridge()
        hooks = LiveHooks(bridge=bridge)

        def original(executable: Any) -> None:
            raise RuntimeError("motion group failed")

        with pytest.raises(RuntimeError):
            hooks._track_motion_group(original, "the-executable")

        assert bridge.frozen_motion_groups == [
            ("the-executable", TaskStatusName.FAILED)
        ]


# %% mesh hook
class TestRememberMeshFile:
    def test_the_mesh_source_is_remembered_before_parsing(self):
        bridge = FakeBridge()
        hooks = LiveHooks(bridge=bridge)

        result = hooks._remember_mesh_file(
            lambda parser: "a-world", FakeMeshParser(file_path="cup.stl")
        )

        assert bridge.remembered_mesh_files == ["cup.stl"]
        assert result == "a-world"

    def test_an_empty_file_path_is_not_remembered(self):
        bridge = FakeBridge()
        hooks = LiveHooks(bridge=bridge)

        hooks._remember_mesh_file(
            lambda parser: "a-world", FakeMeshParser(file_path="")
        )

        assert bridge.remembered_mesh_files == []

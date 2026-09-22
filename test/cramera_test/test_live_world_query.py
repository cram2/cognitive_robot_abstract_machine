from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pytest
from typing_extensions import TYPE_CHECKING

from cramera.knowledge.query_runner import RowRenderer
from cramera.knowledge.queryable_knowledge import QueryScope
from cramera.live import visualization as visualization_module
from cramera.live.bridge import Bridge
from cramera.live.query import NoQuerySourceRegistered
from cramera.live.visualization import LiveVisualization
from cramera.live.world_query import WorldQuerySource
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World

from .test_live_http import bridge, get_json, post, server
from .test_live_query import CurrentStateOnlySource, GrowingRecordSource
from .test_live_visualization import ServerRecorder, world

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing_extensions import Any
    from cramera.knowledge.presets import Preset


# %% source selection
class ResponseField(StrEnum):
    QUERY = "query"
    PRESETS = "presets"
    CODE = "code"
    SCOPE = "scope"
    OK = "ok"


@dataclass
class CountsReads(CurrentStateOnlySource):
    reads: int = 0
    """
    Number of operations that entered this source's read scope.
    """

    @contextmanager
    def read_scope(self) -> Iterator[None]:
        self.reads += 1
        yield


class TestAutomaticWorldQueries:
    def test_attach_enables_current_state_queries(self, world: World) -> None:
        bridge = Bridge()
        bridge.attach(world)

        assert bridge.status()[ResponseField.QUERY] is True
        assert bridge.query_scopes() == [QueryScope.CURRENT_STATE]
        assert bridge.query_source is None

    def test_every_default_preset_runs_without_a_robot(self, world: World) -> None:
        bridge = Bridge()
        bridge.attach(world)

        presets = bridge.query_presets()

        assert presets
        for preset in presets:
            answer = bridge.run_query(preset.code, preset.scope)
            assert answer.ok, preset.text
            assert preset.verbalization is not None
            assert bridge.match_question(preset.text).preset == preset

    def test_reattach_replaces_default_domains(self, world: World) -> None:
        bridge = Bridge()
        bridge.attach(world)
        first_count = len(bridge.query_vocabulary().domains[0].objects)

        bridge.attach(World())

        assert first_count == len(world.bodies)
        assert bridge.query_vocabulary().domains[0].objects == []

    @pytest.mark.parametrize("register_first", [True, False])
    def test_explicit_source_keeps_precedence(
        self, world: World, register_first: bool
    ) -> None:
        source = GrowingRecordSource()
        bridge = Bridge()
        if register_first:
            bridge.register_query_source(source)
        bridge.attach(world)
        if not register_first:
            bridge.register_query_source(source)

        bridge.attach(World())

        assert bridge.query_title() == source.title()
        assert bridge.query_scopes() == [
            knowledge.scope for knowledge in source.knowledge()
        ]
        assert bridge.query_source is source

    def test_an_operation_keeps_its_source_when_registration_changes(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = Bridge()
        bridge.attach(world)
        source = bridge._registered_query_source()
        expected = bridge.query_presets()
        original_presets = source.presets

        def change_source() -> list[Preset]:
            bridge.register_query_source(CurrentStateOnlySource())
            return original_presets()

        monkeypatch.setattr(source, "presets", change_source)

        assert bridge.query_presets() == expected
        assert bridge.query_title() == CurrentStateOnlySource().title()

    def test_custom_sources_participate_in_read_scopes(self) -> None:
        source = CountsReads()
        bridge = Bridge(query_source=source)

        bridge.query_title()
        bridge.query_scopes()
        bridge.query_variables()
        bridge.query_vocabulary()
        bridge.query_presets()
        bridge.match_question(source.title())

        assert source.reads == 6

    def test_explicit_world_source_is_not_replaced(self, world: World) -> None:
        source = WorldQuerySource(world)
        bridge = Bridge(query_source=source)

        bridge.attach(World())

        assert bridge.query_vocabulary().domains[0].objects == world.bodies
        assert bridge.query_source is source

    def test_queries_read_current_poses_without_changing_world_versions(
        self, world: World
    ) -> None:
        bridge = Bridge()
        bridge.attach(world)
        code = (
            Path(__file__).parent / "dataset" / "world_body_positions.eql"
        ).read_text()
        before = bridge.run_query(code)
        connection = world.connections[0]
        connection.origin = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=Point3(1.0, 2.0, 3.0, reference_frame=connection.parent)
        )
        model_version = world.get_world_model_manager().version
        state_version = world.state.version

        after = bridge.run_query(code)

        assert before.ok and after.ok
        assert after.rows != before.rows
        assert world.get_world_model_manager().version == model_version
        assert world.state.version == state_version


# %% concurrent world updates
class TestWorldQueryLocking:
    def test_native_world_stays_locked_through_result_rendering(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = Bridge()
        bridge.attach(world)
        original_render = RowRenderer.rows_of
        competing_reads: list[bool] = []

        def try_read() -> bool:
            acquired = world.state.world_lock.acquire(blocking=False)
            if acquired:
                world.state.world_lock.release()
            return acquired

        with ThreadPoolExecutor(max_workers=1) as executor:

            def render(renderer: RowRenderer, result: Any) -> Any:
                competing_reads.append(executor.submit(try_read).result(timeout=10))
                return original_render(renderer, result)

            monkeypatch.setattr(RowRenderer, "rows_of", render)
            preset = bridge.query_presets()[0]
            answer = bridge.run_query(preset.code)

        assert answer.ok
        assert competing_reads == [False]


# %% visualization lifetime
@pytest.fixture()
def visualization(
    world: World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[LiveVisualization]:
    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
    monkeypatch.setattr(
        visualization_module, "serve", lambda passed_bridge, port: ServerRecorder()
    )
    live = LiveVisualization(world=world)
    yield live
    live.stop()


class TestWorldQueryLifetime:
    def test_stop_releases_default_queries(
        self, visualization: LiveVisualization
    ) -> None:
        visualization.start()
        visualization.stop()

        assert visualization.bridge.status()[ResponseField.QUERY] is False
        with pytest.raises(NoQuerySourceRegistered):
            visualization.bridge.query_presets()

    def test_restart_restores_queries(self, visualization: LiveVisualization) -> None:
        visualization.start()
        visualization.stop()
        visualization.start()

        assert visualization.bridge.status()[ResponseField.QUERY] is True

    def test_stop_preserves_explicit_source(
        self, visualization: LiveVisualization
    ) -> None:
        source = CurrentStateOnlySource()
        visualization.bridge.register_query_source(source)
        visualization.start()
        visualization.stop()

        assert visualization.bridge.query_title() == source.title()

    def test_stop_preserves_a_newer_world_attachment(
        self, visualization: LiveVisualization
    ) -> None:
        visualization.start()
        visualization.bridge.attach(visualization.world)
        visualization.stop()

        assert visualization.bridge.status()[ResponseField.QUERY] is True

    def test_failed_start_releases_default_queries(
        self, visualization: LiveVisualization, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_start(bridge: Bridge, port: int) -> None:
            raise OSError()

        monkeypatch.setattr(visualization_module, "serve", fail_start)

        with pytest.raises(OSError):
            visualization.start()

        assert visualization.bridge.status()[ResponseField.QUERY] is False

    def test_failed_finalization_still_releases_default_queries(
        self, visualization: LiveVisualization, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        visualization.start()

        def fail_finalize(bridge: Bridge, recording: Any) -> None:
            raise OSError()

        with monkeypatch.context() as context:
            context.setattr(visualization_module, "finalize_recording", fail_finalize)
            with pytest.raises(OSError):
                visualization.stop()

        assert visualization.bridge.status()[ResponseField.QUERY] is False


# %% browser endpoint contract
class TestWorldQueriesOverHttp:
    def test_presets_execute_over_the_existing_endpoint(
        self, world: World, bridge: Bridge, server: str
    ) -> None:
        bridge.attach(world)

        payload = get_json(server + "/presets")

        assert payload[ResponseField.OK] is True
        assert payload[ResponseField.PRESETS]
        for preset in payload[ResponseField.PRESETS]:
            status, answer = post(
                server + "/eql",
                {
                    ResponseField.CODE: preset[ResponseField.CODE],
                    ResponseField.SCOPE: preset[ResponseField.SCOPE],
                },
            )
            assert status == 200
            assert answer[ResponseField.OK] is True

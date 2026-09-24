"""
Live world queries preserve source selection, locking and attachment ownership.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path
from threading import Event

import pytest
from typing_extensions import TYPE_CHECKING

from krrood.entity_query_language.factories import an, entity, flat_variable, variable
from cramera.knowledge.presets import Preset
from cramera.knowledge.query_runner import EqlQueryRunner, RowRenderer
from cramera.knowledge.queryable_knowledge import QueryScope, QueryableKnowledge
from cramera.live import visualization as visualization_module
from cramera.live.bridge import Bridge
from cramera.live.query import NoQuerySourceRegistered
from cramera.live.visualization import LiveVisualization
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World

from .test_live_http import bridge, get_json, post, server
from .test_live_query import CurrentStateOnlySource, GrowingRecordSource
from .test_live_visualization import ServerRecorder, world

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing_extensions import Any


# %% source selection
class ResponseField(StrEnum):
    """
    Response fields describing live query availability and execution.
    """

    QUERY = "query"
    """
    Whether the bridge can answer live queries.
    """

    PRESETS = "presets"
    """
    Queries offered by the selected source.
    """

    CODE = "code"
    """
    The executable expression submitted for a query.
    """

    SCOPE = "scope"
    """
    The body of knowledge selected for an expression.
    """

    OK = "ok"
    """
    Whether the request completed successfully.
    """


class TestAutomaticWorldQueries:
    """
    World attachments supply defaults while explicit sources retain precedence.
    """

    def test_attach_enables_current_state_queries(self, world: World) -> None:
        """
        Attaching a world enables queries without registering an explicit source.

        :param world: The robotless scene attached to the bridge.
        """
        bridge = Bridge()
        bridge.attach(world)

        assert bridge.status()[ResponseField.QUERY] is True
        assert bridge.query_scopes() == [QueryScope.CURRENT_STATE]
        assert bridge.query_knowledge is None

    def test_every_default_preset_runs_without_a_robot(self, world: World) -> None:
        """
        Robotless scenes support execution and wording of every default preset.

        :param world: The scene supplying the default query domains.
        """
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
        """
        A new attachment replaces the bodies exposed by automatic queries.

        :param world: The populated scene attached before an empty replacement.
        """
        bridge = Bridge()
        bridge.attach(world)
        first = bridge.query_vocabulary().extra_names[World.__name__.lower()]
        replacement = World()

        bridge.attach(replacement)

        assert first is world
        assert (
            bridge.query_vocabulary().extra_names[World.__name__.lower()] is replacement
        )

    @pytest.mark.parametrize("register_first", [True, False])
    def test_explicit_source_keeps_precedence(
        self, world: World, register_first: bool
    ) -> None:
        """
        Explicit sources override default queries regardless of attachment order.

        :param world: The scene providing automatic queries.
        :param register_first: Whether to register the explicit source before attaching.
        """
        source = GrowingRecordSource()
        bridge = Bridge()
        if register_first:
            bridge.register_query_source(
                source.knowledge(), source.title(), source.presets()
            )
        bridge.attach(world)
        if not register_first:
            bridge.register_query_source(
                source.knowledge(), source.title(), source.presets()
            )

        bridge.attach(World())

        assert bridge.query_title() == source.title()
        assert bridge.query_scopes() == [
            knowledge.scope for knowledge in source.knowledge()
        ]
        assert bridge.query_knowledge == source.knowledge()

    def test_an_operation_keeps_its_knowledge_when_registration_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A replacement registration affects subsequent queries, not an active request.

        :param monkeypatch: Registers new knowledge during preset verbalization.
        """
        source = GrowingRecordSource()
        bridge = Bridge()
        bridge.register_query_source(
            source.knowledge(), source.title(), source.presets()
        )
        expected = bridge.query_presets()
        original_worded = Preset.worded
        replacement = CurrentStateOnlySource()

        def replace_knowledge(preset: Preset, runner: EqlQueryRunner) -> Preset:
            """
            Replace future query knowledge while wording the current preset.

            :param preset: The preset selected before registration changes.
            :param runner: The runner retaining the original query knowledge.
            :return: The original preset worded by its original scope.
            """
            bridge.register_query_source(
                replacement.knowledge(), replacement.title(), replacement.presets()
            )
            return original_worded(preset, runner)

        monkeypatch.setattr(Preset, "worded", replace_knowledge)

        assert bridge.query_presets() == expected
        assert bridge.query_title() == replacement.title()

    def test_explicit_world_knowledge_is_not_replaced(self, world: World) -> None:
        """
        Manually registered native world knowledge survives another attachment.

        :param world: The world retained by the explicitly registered knowledge.
        """
        name = World.__name__.lower()
        knowledge = [
            QueryableKnowledge(
                QueryScope.CURRENT_STATE, domains=[], extra_names={name: world}
            )
        ]
        bridge = Bridge()
        bridge.register_query_source(
            knowledge, type(world).__name__, Preset.of_world(world, name)
        )

        bridge.attach(World())

        assert bridge.query_vocabulary().extra_names[name] is world
        assert bridge.query_knowledge is knowledge

    def test_queries_read_current_poses_without_changing_world_versions(
        self, world: World
    ) -> None:
        """
        Pose queries observe world changes without modifying model or state versions.

        :param world: The scene whose connection pose changes between queries.
        """
        bridge = Bridge()
        bridge.attach(world)
        body = flat_variable(variable(World, domain=[world]).bodies)
        query = an(entity(body.global_pose))
        before = bridge.run_query(query)
        connection = world.connections[0]
        connection.origin = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=Point3(1.0, 2.0, 3.0, reference_frame=connection.parent)
        )
        model_version = world.get_world_model_manager().version
        state_version = world.state.version

        after = bridge.run_query(query)

        assert before.ok and after.ok
        assert after.rows != before.rows
        assert world.get_world_model_manager().version == model_version
        assert world.state.version == state_version


# %% concurrent world updates
class TestWorldQueryLocking:
    """
    Query results stay consistent with the locked native world state.
    """

    def test_registration_can_change_while_waiting_for_a_world_lock(
        self, world: World
    ) -> None:
        """
        Waiting queries use one complete replacement configuration without deadlocking.

        :param world: The explicitly registered world held by the competing thread.
        """
        name = World.__name__.lower()
        selected = Event()
        knowledge = [
            QueryableKnowledge(
                QueryScope.CURRENT_STATE, domains=[], extra_names={name: world}
            )
        ]

        def current_knowledge() -> list[QueryableKnowledge]:
            """
            Signal that the operation selected the original world's knowledge.
            """
            selected.set()
            return knowledge

        bridge = Bridge()
        bridge.register_query_source(
            current_knowledge, type(world).__name__, Preset.of_world(world, name)
        )
        replacement = CurrentStateOnlySource()
        with ThreadPoolExecutor(max_workers=2) as executor:
            with world.state.world_lock:
                answer = executor.submit(bridge.query_presets)
                assert selected.wait(timeout=10)
                registration = executor.submit(
                    bridge.register_query_source,
                    replacement.knowledge(),
                    replacement.title(),
                    replacement.presets(),
                )
                registration.result(timeout=10)
            presets = answer.result(timeout=10)

        assert presets == bridge.query_presets()
        assert bridge.query_title() == replacement.title()

    @pytest.mark.parametrize("attach_another_world", [False, True])
    @pytest.mark.parametrize("knowledge_factory", [False, True])
    @pytest.mark.parametrize("native_expression", [False, True])
    def test_explicit_world_stays_locked_through_result_rendering(
        self,
        world: World,
        monkeypatch: pytest.MonkeyPatch,
        attach_another_world: bool,
        knowledge_factory: bool,
        native_expression: bool,
    ) -> None:
        """
        Explicit world queries prevent concurrent state changes during rendering.

        :param world: The world exposed by the registered query knowledge.
        :param monkeypatch: Attempts a competing state change during rendering.
        :param attach_another_world: Whether the bridge visualizes an unrelated world.
        :param knowledge_factory: Whether registered knowledge comes from a factory.
        :param native_expression: Whether the query is supplied as a native expression.
        """
        name = World.__name__.lower()
        knowledge = [
            QueryableKnowledge(
                QueryScope.CURRENT_STATE, domains=[], extra_names={name: world}
            )
        ]

        def current_knowledge() -> list[QueryableKnowledge]:
            """
            Return the explicit native world knowledge for this operation.
            """
            return knowledge

        bridge = Bridge()
        bridge.register_query_source(
            current_knowledge if knowledge_factory else knowledge,
            type(world).__name__,
            Preset.of_world(world, name),
        )
        if attach_another_world:
            bridge.attach(World())
        connection = world.connections[0]
        original_pose = connection.origin.to_np().copy()
        original_render = RowRenderer.rows_of
        competing_updates: list[bool] = []

        def try_update() -> bool:
            """
            Change the world pose only if its state lock is immediately available.
            """
            acquired = world.state.world_lock.acquire(blocking=False)
            if acquired:
                try:
                    connection.origin = (
                        HomogeneousTransformationMatrix.from_point_rotation_matrix(
                            point=Point3(
                                1.0, 2.0, 3.0, reference_frame=connection.parent
                            )
                        )
                    )
                finally:
                    world.state.world_lock.release()
            return acquired

        with ThreadPoolExecutor(max_workers=1) as executor:

            def render(renderer: RowRenderer, result: Any) -> Any:
                """
                Attempt a competing update before rendering the evaluated result.

                :param renderer: The renderer producing response rows.
                :param result: The query result awaiting rendering.
                :return: The rows produced by the original renderer.
                """
                competing_updates.append(executor.submit(try_update).result(timeout=10))
                return original_render(renderer, result)

            monkeypatch.setattr(RowRenderer, "rows_of", render)
            query = (
                an(entity(flat_variable(variable(World, domain=[world]).bodies)))
                if native_expression
                else bridge.query_presets()[0].code
            )
            answer = bridge.run_query(query)

        assert answer.ok
        assert competing_updates == [False]
        assert (connection.origin.to_np() == original_pose).all()

    @pytest.mark.parametrize("native_expression", [False, True])
    def test_native_world_stays_locked_through_result_rendering(
        self, world: World, monkeypatch: pytest.MonkeyPatch, native_expression: bool
    ) -> None:
        """
        Rendering holds the world lock against acquisition from another thread.

        :param world: The scene whose lock protects the query results.
        :param monkeypatch: Adds a competing lock attempt during result rendering.
        :param native_expression: Whether the query is supplied as a native expression.
        """
        bridge = Bridge()
        bridge.attach(world)
        original_render = RowRenderer.rows_of
        competing_reads: list[bool] = []

        def try_read() -> bool:
            """
            Return whether the world lock can be acquired without waiting.
            """
            acquired = world.state.world_lock.acquire(blocking=False)
            if acquired:
                world.state.world_lock.release()
            return acquired

        with ThreadPoolExecutor(max_workers=1) as executor:

            def render(renderer: RowRenderer, result: Any) -> Any:
                """
                Check lock ownership before rendering the query result.

                :param renderer: The renderer producing response rows.
                :param result: The evaluated query result to render.
                :return: The rows produced by the original renderer.
                """
                competing_reads.append(executor.submit(try_read).result(timeout=10))
                return original_render(renderer, result)

            monkeypatch.setattr(RowRenderer, "rows_of", render)
            preset = bridge.query_presets()[0]
            query = (
                an(entity(flat_variable(variable(World, domain=[world]).bodies)))
                if native_expression
                else preset.code
            )
            answer = bridge.run_query(query)

        assert answer.ok
        assert competing_reads == [False]


# %% visualization lifetime
@pytest.fixture()
def visualization(
    world: World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[LiveVisualization]:
    """
    Provide a visualization with isolated recordings and a replaceable server.

    :param world: The scene presented by the visualization.
    :param monkeypatch: Redirects storage and replaces the server with a recorder.
    :param tmp_path: The temporary directory holding recordings.
    :yield: The visualization, stopped again during fixture cleanup.
    """
    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
    monkeypatch.setattr(
        visualization_module, "serve", lambda passed_bridge, port: ServerRecorder()
    )
    live = LiveVisualization(world=world)
    yield live
    live.stop()


class TestWorldQueryLifetime:
    """
    Automatic queries follow the lifetime of their owning world attachment.
    """

    def test_stop_releases_default_queries(
        self, visualization: LiveVisualization
    ) -> None:
        """
        Stopping the owning visualization removes its automatic query source.

        :param visualization: The visualization acquiring the default queries.
        """
        visualization.start()
        visualization.stop()

        assert visualization.bridge.status()[ResponseField.QUERY] is False
        with pytest.raises(NoQuerySourceRegistered):
            visualization.bridge.query_presets()

    def test_restart_restores_queries(self, visualization: LiveVisualization) -> None:
        """
        Restarting a stopped visualization restores automatic query availability.

        :param visualization: The visualization stopped and started again.
        """
        visualization.start()
        visualization.stop()
        visualization.start()

        assert visualization.bridge.status()[ResponseField.QUERY] is True

    def test_stop_preserves_explicit_source(
        self, visualization: LiveVisualization
    ) -> None:
        """
        Stopping a visualization leaves an explicitly registered source available.

        :param visualization: The visualization whose bridge receives a custom source.
        """
        source = CurrentStateOnlySource()
        visualization.bridge.register_query_source(
            source.knowledge(), source.title(), source.presets()
        )
        visualization.start()
        visualization.stop()

        assert visualization.bridge.query_title() == source.title()

    def test_stop_preserves_a_newer_world_attachment(
        self, visualization: LiveVisualization
    ) -> None:
        """
        Cleanup of an older attachment does not remove a newer attachment's queries.

        :param visualization: The visualization whose bridge is reattached after
            startup.
        """
        visualization.start()
        visualization.bridge.attach(visualization.world)
        visualization.stop()

        assert visualization.bridge.status()[ResponseField.QUERY] is True

    def test_model_updates_do_not_change_query_ownership(
        self, visualization: LiveVisualization
    ) -> None:
        """
        Model updates leave cleanup ownership with the original attachment.

        :param visualization: The session whose model revision advances before stopping.
        """
        visualization.start()
        visualization.bridge.observe_model_change()

        visualization.stop()

        assert visualization.bridge.status()[ResponseField.QUERY] is False

    def test_failed_start_releases_default_queries(
        self, visualization: LiveVisualization, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Failure to start the server releases automatic queries acquired earlier.

        :param visualization: The visualization whose server startup fails.
        :param monkeypatch: Replaces server startup with a failing implementation.
        """

        def fail_start(bridge: Bridge, port: int) -> None:
            """
            Reject server startup after the world has been attached.

            :param bridge: The bridge offered for serving.
            :param port: The requested listening port.
            :raises OSError: Always, to represent a server startup failure.
            """
            raise OSError()

        monkeypatch.setattr(visualization_module, "serve", fail_start)

        with pytest.raises(OSError):
            visualization.start()

        assert visualization.bridge.status()[ResponseField.QUERY] is False

    def test_failed_finalization_still_releases_default_queries(
        self, visualization: LiveVisualization, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Recording finalization errors cannot retain the automatic query source.

        :param visualization: The visualization whose recording cannot be finalized.
        :param monkeypatch: Replaces recording finalization with a failing
            implementation.
        """
        visualization.start()

        def fail_finalize(bridge: Bridge, recording: Any) -> None:
            """
            Reject finalization of an otherwise active recording.

            :param bridge: The bridge holding the recording.
            :param recording: The capture offered for finalization.
            :raises OSError: Always, to represent a finalization failure.
            """
            raise OSError()

        with monkeypatch.context() as context:
            context.setattr(visualization_module, "finalize_recording", fail_finalize)
            with pytest.raises(OSError):
                visualization.stop()

        assert visualization.bridge.status()[ResponseField.QUERY] is False


# %% browser endpoint contract
class TestWorldQueriesOverHttp:
    """
    Automatic world queries use the existing preset and query endpoints.
    """

    def test_native_world_members_are_available_over_http(
        self, world: World, bridge: Bridge, server: str
    ) -> None:
        """
        The world value offers its native members through the completion endpoint.

        :param world: The native world exposed to the query editor.
        :param bridge: The bridge served by the local endpoint.
        :param server: The base URL for query completion requests.
        """
        bridge.attach(world)

        payload = get_json(server + "/members?name=" + World.__name__.lower())

        assert payload[ResponseField.OK] is True
        assert World.bodies.fget.__name__ in {
            member["name"] for member in payload["members"]
        }

    def test_presets_execute_over_the_existing_endpoint(
        self, world: World, bridge: Bridge, server: str
    ) -> None:
        """
        Every advertised world preset executes successfully over HTTP.

        :param world: The scene supplying automatic presets.
        :param bridge: The bridge served by the local endpoint.
        :param server: The base URL for preset discovery and query execution.
        """
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

"""
End-to-end tests of the live bridge's HTTP layer.

Every endpoint is served against an explicitly injected :class:`Bridge`, never the
module-level ``BRIDGE`` singleton — the concrete proof that
:class:`BridgeRequestHandler` and :func:`serve` no longer read a shared global.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cramera import paths
from cramera.live.bridge import Bridge
from cramera.live.http import serve
from cramera.live.recording import Recording

from .test_live_bridge import shaped_body, world_with
from .test_server import get, get_json


def post(url, payload=None, timeout=10):
    """
    POST a JSON body (or none) and return ``(status, decoded json body)``.
    """
    request = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(payload).encode() if payload is not None else b"{}",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.fixture()
def bridge():
    return Bridge()


@pytest.fixture()
def server(bridge):
    """
    A real server on an ephemeral port, bound to ``bridge``.
    """
    httpd = serve(bridge, 0)
    yield "http://localhost:%d" % httpd.server_address[1]
    httpd.shutdown()


def publish_mesh_object(
    bridge, tmp_path, key="milk.stl", content=b"solid milk endsolid"
):
    """
    Publish one mesh-backed object on ``bridge``, with a real file behind it.

    The object's mesh is served shape by shape, so its serve key is ``<key>#0``.
    """
    mesh_file = tmp_path / key
    mesh_file.write_bytes(content)
    body = Body(
        name=PrefixedName(key, prefix="world"),
        visual=ShapeCollection(shapes=[Mesh(filename=str(mesh_file))]),
    )
    bridge.publish_bodies({key: body})
    return mesh_file


class TestReadOnlyEndpoints:
    def test_state_reflects_a_fresh_bridge(self, server):
        assert get_json(server + "/state") == {
            "sequenceNumber": 0,
            "frames": {},
            "base": None,
            "objects": {},
            "modelBases": {},
            "markersVersion": 0,
        }

    def test_plan_reflects_a_fresh_bridge(self, server):
        assert get_json(server + "/plan")["nodes"] == []

    def test_chart_reflects_a_fresh_bridge(self, server):
        chart = get_json(server + "/chart")
        assert chart["nodes"] == []
        assert chart["edges"] == []

    def test_transforms_reflects_a_fresh_bridge(self, server):
        assert get_json(server + "/transforms")["connections"] == []

    def test_transforms_serves_the_attached_worlds_connection_graph(
        self, server, bridge
    ):
        bridge.world = world_with(shaped_body("kitchen", "shelf"))
        bridge.bind()
        bridge.snapshot()

        payload = get_json(server + "/transforms")

        assert [entry["child"] for entry in payload["connections"]] == ["kitchen/shelf"]
        assert payload["signature"] == bridge.transform_state.signature

    def test_objects_reflects_the_injected_bridges_catalog(
        self, server, bridge, tmp_path
    ):
        publish_mesh_object(bridge, tmp_path)
        payload = get_json(server + "/objects")
        assert [entry["key"] for entry in payload["objects"]] == ["milk.stl"]

    def test_info_reflects_a_fresh_bridge(self, server):
        assert get_json(server + "/info") == {
            "running": False,
            "robot": None,
            "objects": [],
            "movable": True,
            "plan": False,
            "chart": False,
            "sequenceNumber": 0,
            "modelVersion": 0,
            "bundleSignature": Bridge().bundle_signature(),
            "partAnnotations": [],
        }

    def test_unknown_get_path_is_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as error:
            get(server + "/nope")
        assert error.value.code == 404


class TestMesh:
    def test_a_published_meshs_bytes_are_served(self, server, bridge, tmp_path):
        publish_mesh_object(bridge, tmp_path, content=b"solid milk endsolid")
        status, body = get(server + "/mesh?key=milk.stl%230")
        assert status == 200
        assert body == b"solid milk endsolid"

    def test_an_unknown_mesh_key_is_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as error:
            get(server + "/mesh?key=nope.stl")
        assert error.value.code == 404

    def test_a_side_asset_is_served_from_the_meshs_directory(
        self, server, bridge, tmp_path
    ):
        publish_mesh_object(bridge, tmp_path, key="board.obj", content=b"o board")
        (tmp_path / "board.mtl").write_bytes(b"newmtl paint")

        status, body = get(server + "/mesh?key=board.obj%230&side=board.mtl")

        assert status == 200
        assert body == b"newmtl paint"

    def test_a_side_asset_outside_the_meshs_directory_is_refused(
        self, server, bridge, tmp_path
    ):
        publish_mesh_object(bridge, tmp_path, key="board.obj", content=b"o board")

        with pytest.raises(urllib.error.HTTPError) as error:
            get(server + "/mesh?key=board.obj%230&side=..%2Fsecret.txt")

        assert error.value.code == 403


class TestMarkers:
    def test_a_fresh_bridge_serves_an_empty_overlay(self, server):
        assert get_json(server + "/markers") == {"version": 0, "markers": []}


class TestLiveScene:
    def test_live_scene_reflects_a_fresh_bridge(self, server):
        assert get_json(server + "/live_scene") == {"scene": None}

    def test_live_scene_bundles_the_attached_world(
        self, server, bridge, tmp_path, monkeypatch
    ):
        scenes = tmp_path / "scenes"
        monkeypatch.setenv("CRAMERA_SCENES", str(scenes))
        bridge.attach(world_with(shaped_body("laboratory", "bench")))

        assert get_json(server + "/live_scene") == {"scene": paths.LIVE_SCENE_NAME}
        scene_directory = scenes / paths.LIVE_SCENE_NAME
        assert (scene_directory / "scene.json").is_file()
        assert (scene_directory / "environment.urdf").is_file()


class TestRecordingEndpoints:
    def start_recording(self, bridge, ticks=2) -> None:
        """
        Attach ``bridge`` to a real world, start a recording, and feed it a few ticks.
        """
        bridge.attach(world_with(shaped_body("laboratory", "bench")))
        bridge.recording = Recording()
        bridge.recording.start()
        for _ in range(ticks):
            bridge.snapshot()
            bridge.recording.append(bridge.state)

    def test_a_fresh_bridge_is_idle(self, server):
        assert get_json(server + "/recording") == {
            "state": "idle",
            "frameCount": 0,
            "durationSeconds": 0.0,
            "sceneName": None,
        }

    def test_stop_without_a_recording_is_rejected(self, server):
        status, body = post(server + "/recording/stop")
        assert status == 400
        assert body["ok"] is False

    def test_stop_bundles_the_recording_under_the_local_data_directory_only(
        self, server, bridge, tmp_path, monkeypatch
    ):
        shared = tmp_path / "shared"
        monkeypatch.setenv("CRAMERA_SCENES", str(shared))
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        self.start_recording(bridge)

        status, body = post(server + "/recording/stop")

        assert status == 200
        assert body["ok"] is True
        assert body["scene"] == paths.RECORDING_SCENE_NAME
        local_bundle = tmp_path / "data" / "scenes" / paths.RECORDING_SCENE_NAME
        assert (local_bundle / "scene.json").is_file()
        assert not shared.exists()

    def test_recording_status_reflects_the_finalized_bundle(
        self, server, bridge, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        self.start_recording(bridge)
        post(server + "/recording/stop")

        status = get_json(server + "/recording")

        assert status["state"] == "finalized"
        assert status["sceneName"] == paths.RECORDING_SCENE_NAME
        assert status["frameCount"] == 2

    def test_stopping_twice_does_not_rebundle(
        self, server, bridge, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        self.start_recording(bridge)
        post(server + "/recording/stop")
        local_bundle = tmp_path / "data" / "scenes" / paths.RECORDING_SCENE_NAME
        marker = local_bundle / "marker"
        marker.touch()

        status, body = post(server + "/recording/stop")

        assert status == 200
        assert marker.exists()

    def test_stopping_an_empty_recording_is_rejected(
        self, server, bridge, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        bridge.attach(world_with(shaped_body("laboratory", "bench")))
        bridge.recording = Recording()
        bridge.recording.start()  # no ticks appended

        status, body = post(server + "/recording/stop")

        assert status == 400
        assert body["ok"] is False

    def test_discard_removes_the_bundle_and_returns_to_idle(
        self, server, bridge, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        self.start_recording(bridge)
        post(server + "/recording/stop")
        local_bundle = tmp_path / "data" / "scenes" / paths.RECORDING_SCENE_NAME

        status, body = post(server + "/recording/discard")

        assert status == 200
        assert body["ok"] is True
        assert not local_bundle.exists()
        assert get_json(server + "/recording")["state"] == "idle"

    def test_discard_without_a_recording_is_harmless(self, server):
        status, body = post(server + "/recording/discard")
        assert status == 200
        assert body["ok"] is True

    def test_save_moves_the_bundle_and_updates_the_local_index(
        self, server, bridge, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        self.start_recording(bridge)
        post(server + "/recording/stop")

        status, body = post(server + "/recording/save", {"name": "my_run"})

        assert status == 200
        assert body == {"ok": True, "scene": "my_run"}
        saved = tmp_path / "data" / "scenes" / "my_run"
        assert json.loads((saved / "scene.json").read_text())["name"] == "my_run"
        assert not (tmp_path / "data" / "scenes" / paths.RECORDING_SCENE_NAME).exists()
        index = json.loads((tmp_path / "data" / "scenes" / "index.json").read_text())
        assert any(entry["name"] == "my_run" for entry in index["scenes"])

    def test_save_frees_the_slot_for_another_recording(
        self, server, bridge, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        self.start_recording(bridge)
        post(server + "/recording/stop")
        post(server + "/recording/save", {"name": "my_run"})

        assert get_json(server + "/recording")["state"] == "idle"

    def test_save_without_a_finalized_recording_is_rejected(self, server):
        status, body = post(server + "/recording/save", {"name": "my_run"})
        assert status == 400
        assert body["ok"] is False

    def test_save_rejects_an_unsafe_name(self, server, bridge, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        self.start_recording(bridge)
        post(server + "/recording/stop")

        status, body = post(server + "/recording/save", {"name": "../escape"})

        assert status == 400
        assert body["ok"] is False

    def test_save_rejects_a_name_collision(self, server, bridge, tmp_path, monkeypatch):
        shared = tmp_path / "shared"
        (shared / "kitchen").mkdir(parents=True)
        (shared / "kitchen" / "scene.json").write_text("{}")
        monkeypatch.setenv("CRAMERA_SCENES", str(shared))
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        self.start_recording(bridge)
        post(server + "/recording/stop")

        status, body = post(server + "/recording/save", {"name": "kitchen"})

        assert status == 409
        assert body["ok"] is False


class TestMove:
    def test_a_valid_move_is_queued_on_the_injected_bridge(self, server, bridge):
        request = urllib.request.Request(
            server + "/move",
            method="POST",
            data=json.dumps(
                {"object": "milk.stl", "position": [1.0, 2.0, 3.0]}
            ).encode(),
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 200
            assert json.loads(response.read()) == {"ok": True}
        assert [move.object_key for move in bridge._moves] == ["milk.stl"]

    def test_a_malformed_move_is_rejected_without_touching_the_bridge(
        self, server, bridge
    ):
        request = urllib.request.Request(
            server + "/move",
            method="POST",
            data=json.dumps({"object": "milk.stl"}).encode(),
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, body = error.code, error.read()
        assert status == 400
        assert json.loads(body)["ok"] is False
        assert bridge._moves == []

    def test_unknown_post_path_is_404(self, server):
        request = urllib.request.Request(server + "/nope", method="POST", data=b"{}")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = response.status
        except urllib.error.HTTPError as error:
            status = error.code
        assert status == 404


class TestOptions:
    def test_preflight_returns_cors_headers(self, server):
        request = urllib.request.Request(server + "/move", method="OPTIONS")
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == "*"
            assert "POST" in response.headers["Access-Control-Allow-Methods"]


class TestTwoIndependentBridges:
    def test_each_server_reflects_only_its_own_bridge(self, tmp_path):
        first_bridge, second_bridge = Bridge(), Bridge()
        publish_mesh_object(first_bridge, tmp_path, key="milk.stl")
        publish_mesh_object(second_bridge, tmp_path, key="cup.stl")
        first_server = serve(first_bridge, 0)
        second_server = serve(second_bridge, 0)
        try:
            first_url = "http://localhost:%d" % first_server.server_address[1]
            second_url = "http://localhost:%d" % second_server.server_address[1]
            first_objects = get_json(first_url + "/objects")["objects"]
            second_objects = get_json(second_url + "/objects")["objects"]
            assert [entry["key"] for entry in first_objects] == ["milk.stl"]
            assert [entry["key"] for entry in second_objects] == ["cup.stl"]
        finally:
            first_server.shutdown()
            second_server.shutdown()


class TestClientDisconnects:
    """
    A browser aborts requests as a matter of course: the viewer polls on an interval and
    navigating away cancels whatever is in flight.

    The socket is then gone before the response is written, which surfaces as a
    ``BrokenPipeError`` out of the write. That is the client's normal behaviour, not a
    server fault, and printing a traceback per occurrence buries the log a real fault
    would show up in.
    """

    def raise_and_report(self, server, failure: BaseException) -> None:
        """
        Report ``failure`` to the server the way ``socketserver`` does, from inside an
        active exception handler.

        :param server: The running server whose error reporting is exercised.
        :param failure: The exception to report.
        """
        try:
            raise failure
        except type(failure):
            server.handle_error(None, ("127.0.0.1", 36976))

    def test_a_client_that_hung_up_is_not_reported(self, bridge, capsys):
        httpd = serve(bridge, 0)
        try:
            self.raise_and_report(httpd, BrokenPipeError(32, "Broken pipe"))
            self.raise_and_report(httpd, ConnectionResetError(104, "Connection reset"))
        finally:
            httpd.shutdown()

        assert capsys.readouterr().err == ""

    def test_a_real_fault_is_still_reported(self, bridge, capsys):
        """
        The quiet path must stay narrow: anything that is not the client hanging up has
        to keep reaching the log.
        """
        httpd = serve(bridge, 0)
        try:
            self.raise_and_report(httpd, ValueError("a real bug"))
        finally:
            httpd.shutdown()

        assert "ValueError: a real bug" in capsys.readouterr().err

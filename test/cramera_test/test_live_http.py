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

from cramera.live.bridge import Bridge
from cramera.live.http import serve

from .test_live_bridge import PublishedBody
from .test_server import get, get_json


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
    """
    mesh_file = tmp_path / key
    mesh_file.write_bytes(content)
    bridge.remember_mesh_file(str(mesh_file))
    bridge.publish_bodies({key: PublishedBody(name="world/" + key)})
    return mesh_file


class TestReadOnlyEndpoints:
    def test_state_reflects_a_fresh_bridge(self, server):
        assert get_json(server + "/state") == {
            "sequenceNumber": 0,
            "frames": {},
            "base": None,
            "objects": {},
        }

    def test_plan_reflects_a_fresh_bridge(self, server):
        assert get_json(server + "/plan")["nodes"] == []

    def test_chart_reflects_a_fresh_bridge(self, server):
        chart = get_json(server + "/chart")
        assert chart["nodes"] == []
        assert chart["edges"] == []

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
            "partAnnotations": [],
        }

    def test_unknown_get_path_is_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as error:
            get(server + "/nope")
        assert error.value.code == 404


class TestMesh:
    def test_a_published_meshs_bytes_are_served(self, server, bridge, tmp_path):
        publish_mesh_object(bridge, tmp_path, content=b"solid milk endsolid")
        status, body = get(server + "/mesh?key=milk.stl")
        assert status == 200
        assert body == b"solid milk endsolid"

    def test_an_unknown_mesh_key_is_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as error:
            get(server + "/mesh?key=nope.stl")
        assert error.value.code == 404


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

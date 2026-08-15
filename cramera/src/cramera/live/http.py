"""
HTTP endpoints of the live bridge (default port 8765).

::

    GET /info    {running, robot, objects, plan, chart, sequenceNumber,
                  partAnnotations}
    GET /state   {sequenceNumber, frames: {prefixed_joint: position},
                  base: pose, objects: {mesh_key: pose}}
    GET /objects geometry catalog (mesh served via /mesh?key=)
    GET /markers {version, markers: [{topic, ns, id, kind, pose, scale, color,
                  opacity, points, text}]}  the CRAM debug-marker overlay
    GET /live_scene  {scene}  bundles the running demo's *current* world into a
                      throwaway scene (see :mod:`cramera.live.live_bundle`) and names
                      it, or {scene: null} when nothing is tracked yet; the viewer
                      loads the named scene exactly like any other
    GET /plan    {signature, nodes: [{id, parent, kind, label, status, derived}]}
    GET /chart   {signature, title,
                  nodes: [{id, parent, name, class_name, life_cycle, observation}],
                  edges: [{from, to, kind}]}
    GET /transforms  {signature, connections: [{name, parent, child, kind, writer,
                      freshness, ageSeconds}]}  the world's connection graph and how
                      recently each connection moved (see
                      :mod:`cramera.live.transforms`)
    GET /recording  {state: idle|recording|finalized, frameCount, durationSeconds,
                      sceneName}  see :mod:`cramera.live.recording`
    POST /move   queue an object move (applied on the simulation thread)
    POST /recording/stop     finalize the current recording into a scene bundle under
                              :func:`cramera.paths.local_scenes_directory`
    POST /recording/discard  drop the current recording and its bundle, if any
    POST /recording/save     {name} -> promote the finalized bundle to a permanent,
                              locally saved scene

Every ``pose`` above is ``[x, y, z, qx, qy, qz, qw]``.

Handlers only ever read finished snapshot dicts — never the world; the bridge's
snapshots are produced on the simulation thread by the world and plan callbacks of
:mod:`cramera.live.visualization`.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from typing_extensions import Any, ClassVar, Dict, Optional, Tuple, Type

from cramera.live.bridge import Bridge, MalformedMoveRequest, MoveRequest
from cramera.live.live_bundle import build_live_scene
from cramera.live.recording import Recording, RecordingState
from cramera.live.recording_bundle import finalize_recording
from cramera.live.recording_storage import (
    NoSavedRecording,
    SceneNameTaken,
    discard_recording_bundle,
    save_recording_bundle,
)
from cramera.logging_setup import get_logger
from cramera.onboard.scene_index import InvalidSceneName

logger = get_logger(__name__)

DEFAULT_PORT = int(os.environ.get("LIVE_VIZ_PORT", "8765"))


class BridgeRequestHandler(BaseHTTPRequestHandler):
    """
    Serves the bridge's snapshots and accepts viewer moves.
    """

    def __init__(self, *args: Any, bridge: Bridge, **kwargs: Any) -> None:
        """
        Capture the bridge before delegating, since the base constructor already
        dispatches the request synchronously.

        :param args: Positional arguments forwarded to the base handler.
        :param bridge: The bridge this handler serves.
        :param kwargs: Keyword arguments forwarded to the base handler.
        """
        self.bridge = bridge
        super().__init__(*args, **kwargs)

    def _send_json(self, payload: Dict[str, Any], code: int = 200) -> None:
        """
        Send a JSON payload with the CORS headers the viewer needs.

        :param payload: The JSON-serializable payload to send.
        :param code: HTTP status code to respond with.
        """
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """
        Entry point :class:`~http.server.BaseHTTPRequestHandler` dispatches a ``GET``
        request to, found by name as ``"do_" + self.command``.
        """
        self.route_snapshot_request()

    def route_snapshot_request(self) -> None:
        """
        Route the read-only snapshot endpoints.
        """
        if self.path.startswith("/state"):
            return self._send_json(self.bridge.get_state())
        if self.path.startswith("/plan"):
            return self._send_json(self.bridge.get_plan())
        if self.path.startswith("/chart"):
            return self._send_json(self.bridge.get_chart())
        if self.path.startswith("/transforms"):
            return self._send_json(self.bridge.get_transforms())
        if self.path.startswith("/objects"):
            return self._send_json({"objects": self.bridge.object_catalog()})
        if self.path.startswith("/mesh"):
            return self._send_mesh()
        if self.path.startswith("/live_scene"):
            return self._send_json({"scene": build_live_scene(self.bridge)})
        if self.path.startswith("/markers"):
            return self._send_json(self.bridge.get_markers())
        if self.path.startswith("/info"):
            return self._send_json(self.bridge.status())
        if self.path.startswith("/recording"):
            return self._send_json(self._recording_status())
        self.send_response(404)
        self.end_headers()

    def _recording_status(self) -> Dict[str, Any]:
        """
        What the viewer polls to show recording/playback controls.
        """
        if self.bridge.recording is None:
            return Recording().status_payload()
        return self.bridge.recording.status_payload()

    def _query_value(self, name: str) -> Optional[str]:
        """
        One query-string parameter's value, or None if it is absent.

        :param name: The parameter's name.
        """
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        values = query.get(name)
        return values[0] if values else None

    def _send_mesh(self) -> None:
        """
        Serve one object's mesh file, or one of its side assets (plain file IO, no world
        access).

        A ``side`` query parameter names a file relative to the mesh's own directory —
        its ``.mtl`` companion or a texture it references. Requests that resolve outside
        that directory are refused.
        """
        mesh_path = self.bridge.mesh_path(self._query_value("key") or "")
        side = self._query_value("side")
        if mesh_path is None or not side:
            return self._send_file(mesh_path)
        mesh_directory = Path(mesh_path).parent.resolve()
        side_path = (mesh_directory / side).resolve()
        if not side_path.is_relative_to(mesh_directory):
            return self._send_json({"error": "side asset outside mesh directory"}, 403)
        return self._send_file(str(side_path))

    def _send_file(self, path: Optional[str]) -> None:
        """
        Stream an absolute path's bytes, or 404 when it does not resolve to a file.

        :param path: The absolute path to stream, or None/empty when nothing resolved.
        """
        if not path or not Path(path).is_file():
            self.send_response(404)
            self.end_headers()
            return
        data = Path(path).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        """
        Entry point :class:`~http.server.BaseHTTPRequestHandler` dispatches a ``POST``
        request to, found by name as ``"do_" + self.command``.
        """
        if self.path.startswith("/marker_topics"):
            return self.set_marker_topic()
        if self.path == "/recording/stop":
            return self._stop_recording()
        if self.path == "/recording/discard":
            return self._discard_recording()
        if self.path == "/recording/save":
            return self._save_recording()
        self.queue_requested_move()

    def _stop_recording(self) -> None:
        """
        Finalize the current recording, bundling it on first stop.

        Idempotent: stopping an already-finalized recording just re-reports it, so a
        browser tab that already finalized a recording can ask again safely.
        """
        recording = self.bridge.recording
        if recording is None or recording.state is RecordingState.IDLE:
            return self._send_json({"ok": False, "error": "no active recording"}, 400)
        scene_name = finalize_recording(self.bridge, recording)
        if scene_name is None:
            return self._send_json(
                {"ok": False, "error": "the recording has no frames"}, 400
            )
        payload = recording.status_payload()
        payload.update({"ok": True, "scene": scene_name})
        self._send_json(payload)

    def _discard_recording(self) -> None:
        """
        Drop the current recording and its unsaved bundle, if any.
        """
        if self.bridge.recording is not None:
            self.bridge.recording.discard()
        discard_recording_bundle()
        self._send_json({"ok": True})

    def _save_recording(self) -> None:
        """
        Promote the finalized recording to a permanent, locally saved scene.
        """
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        recording = self.bridge.recording
        if recording is None or recording.state is not RecordingState.FINALIZED:
            return self._send_json(
                {"ok": False, "error": "nothing finalized to save"}, 400
            )
        try:
            name = save_recording_bundle(str(payload.get("name") or ""))
        except InvalidSceneName as error:
            return self._send_json({"ok": False, "error": str(error)}, 400)
        except NoSavedRecording as error:
            return self._send_json({"ok": False, "error": str(error)}, 400)
        except SceneNameTaken as error:
            return self._send_json({"ok": False, "error": str(error)}, 409)
        recording.discard()
        self._send_json({"ok": True, "scene": name})

    def set_marker_topic(self) -> None:
        """
        Watch or drop a marker topic, as the viewer's marker settings ask.
        """
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        self._send_json(
            self.bridge.set_marker_topic(
                str(payload.get("topic") or ""), bool(payload.get("subscribed", True))
            )
        )

    def queue_requested_move(self) -> None:
        """
        Queue an object move requested by the viewer.

        The payload is validated here so that malformed input is rejected on the HTTP
        thread, rather than raising later inside the simulation tick.
        """
        if not self.path.startswith("/move"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            return self._send_json({"ok": False, "error": str(error)}, code=400)
        if not isinstance(payload, dict):
            return self._send_json(
                {"ok": False, "error": "body must be a JSON object"}, code=400
            )
        try:
            move = MoveRequest.from_payload(payload)
        except MalformedMoveRequest as error:
            return self._send_json({"ok": False, "error": str(error)}, code=400)
        self.bridge.queue_move(move)
        return self._send_json({"ok": True})

    def do_OPTIONS(self) -> None:
        """
        Entry point :class:`~http.server.BaseHTTPRequestHandler` dispatches an
        ``OPTIONS`` request to, found by name as ``"do_" + self.command``.
        """
        self.answer_preflight()

    def answer_preflight(self) -> None:
        """
        CORS preflight for the viewer's cross-origin POSTs.
        """
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """
        Route the per-request access log to debug (15 Hz polling is noisy).

        :param format:``printf``-style log message format.
        :param args: Values to interpolate into ``format``.
        """
        logger.debug(format, *args)


class BridgeServer(ThreadingHTTPServer):
    """
    The bridge's HTTP server, which treats a client hanging up as the end of a request.

    A browser aborts requests routinely -- the viewer polls on an interval, and
    navigating away cancels whatever is in flight -- so the socket is regularly gone
    before the response is written. ``socketserver`` reports that like any other fault,
    one traceback per occurrence, which buries the log a real fault would have to be
    found in.
    """

    CLIENT_HUNG_UP: ClassVar[Tuple[Type[BaseException], ...]] = (
        BrokenPipeError,
        ConnectionResetError,
    )
    """
    Exceptions that mean the client is gone rather than that the server misbehaved.
    """

    def handle_error(self, request: Any, client_address: Any) -> None:
        """
        Report a failed request, unless it failed because the client hung up.

        :param request: The request being handled, as ``socketserver`` passes it.
        :param client_address: Address of the client whose request failed.
        """
        if isinstance(sys.exc_info()[1], self.CLIENT_HUNG_UP):
            return
        super().handle_error(request, client_address)


def serve(bridge: Bridge, port: int = DEFAULT_PORT) -> BridgeServer:
    """
    Start an HTTP server on a daemon thread, serving ``bridge``.

    :param bridge: The bridge every request handler on this server reads and writes.
    :param port: Port to listen on (all interfaces).
    :return: The running server.
    """
    handler = functools.partial(BridgeRequestHandler, bridge=bridge)
    server = BridgeServer(("0.0.0.0", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

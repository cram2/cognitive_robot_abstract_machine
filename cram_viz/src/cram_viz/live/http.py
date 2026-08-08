"""
HTTP endpoints of the live bridge (default port 8765).

::

    GET /info    {running, robot, objects, plan, chart, seq}
    GET /state   {seq, frames: {prefixed_joint: pos}, base: pose7,
                  objects: {mesh_key: pose7}}
    GET /objects geometry catalog (mesh served via /mesh?key=)
    GET /plan    {signature, nodes: [{id, parent, kind, label, status, derived}]}
    GET /chart   {signature, title, nodes: [{id, parent, name, cls, life, obs}],
                  edges: [{from, to, kind}]}
    POST /move   queue an object move (applied on the simulation thread)

Handlers only ever read finished snapshot dicts — never the world (see
:mod:`cram_viz.live.hooks`).
"""

from __future__ import annotations

import functools
import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from typing_extensions import Any, Dict

from cram_viz import get_logger
from cram_viz.live.bridge import Bridge, MalformedMoveRequest, MoveRequest

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
        Route the read-only snapshot endpoints.
        """
        if self.path.startswith("/state"):
            return self._send_json(self.bridge.get_state())
        if self.path.startswith("/plan"):
            return self._send_json(self.bridge.get_plan())
        if self.path.startswith("/chart"):
            return self._send_json(self.bridge.get_chart())
        if self.path.startswith("/objects"):
            return self._send_json({"objects": self.bridge.object_catalog()})
        if self.path.startswith("/mesh"):
            return self._send_mesh()
        if self.path.startswith("/info"):
            return self._send_json(self.bridge.status())
        self.send_response(404)
        self.end_headers()

    def _send_mesh(self) -> None:
        """
        Serve one object's mesh file (plain file IO, no world access).
        """
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        key = (query.get("key") or [""])[0]
        path = self.bridge.mesh_path(key)
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

        :param format: ``printf``-style log message format.
        :param args: Values to interpolate into ``format``.
        """
        logger.debug(format, *args)


def serve(bridge: Bridge, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """
    Start an HTTP server on a daemon thread, serving ``bridge``.

    :param bridge: The bridge every request handler on this server reads and writes.
    :param port: Port to listen on (all interfaces).
    :return: The running server.
    """
    handler = functools.partial(BridgeRequestHandler, bridge=bridge)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

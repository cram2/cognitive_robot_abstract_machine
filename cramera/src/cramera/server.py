"""
The cramera HTTP server: static frontend + JSON API.

Serves three things from one port (default 8711):

  * the packaged web frontend (``cramera/web`` — panels, vendored libs)
  * scene bundles from :func:`cramera.paths.scenes_directory` under ``/scenes/``
  * the JSON API the panels talk to:

      GET  /api/knowledge              the knowledge-graph overview payload
      GET  /api/knowledge/view?name=   one graph tab (knowledge/kinematics/plan/chart)
      GET  /api/knowledge/expand?node= drill-down subgraph for one node
      POST /api/eql             run an EQL query string

The API needs krrood (EQL). Without it the server still serves the viewer and
answers API calls with ``{"ok": false, "error": ...}`` so the frontend can say
why the knowledge panel is empty.

    cramera            # console script
    python -m cramera.server [port]
"""

from __future__ import annotations

import http.server
import json
import logging
import mimetypes
import os
import socketserver
import sys
import threading
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Any, Callable, ClassVar, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from cramera import paths
from cramera.logging_setup import get_logger
from cramera.models_workbench import (
    ModelWorkbench,
    NO_MODELS_MESSAGE,
    PROBABILISTIC_MODELS_AVAILABLE,
)
from cramera.payload import CrameraPayload

logger = get_logger(__name__)

DEFAULT_PORT = 8711

try:
    import krrood  # noqa: F401  (the EQL engine)

    from cramera.knowledge.eql_session import EqlSession
    from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
    from cramera.knowledge.views.dispatcher import GraphPanelViews

    EQL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    EQL_AVAILABLE = False
    logger.warning("krrood not importable — serving the viewer without the EQL API")
except (
    Exception
):  # pragma: no cover - a broken knowledge base should not kill the viewer
    EQL_AVAILABLE = False
    traceback.print_exc()


_EQL_LOCK = threading.Lock()
"""
Krrood's SymbolGraph singleton is not threadsafe; queries are serialized.
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    """
    Static files from the packaged web root, plus the JSON API routes.
    """

    NO_EQL_MESSAGE: ClassVar[str] = "krrood/EQL not available in this environment"
    """
    What every API route answers with when krrood is not importable.
    """

    def __init__(self, *args, **kwargs):
        """
        Delegate to the base handler, serving from the packaged web root.

        :param args: Positional arguments forwarded to the base handler.
        :param kwargs: Keyword arguments forwarded to the base handler.
        """
        super().__init__(*args, directory=str(paths.WEB_ROOT), **kwargs)

    def end_headers(self) -> None:
        """
        Disable caching so a rebuilt scene/frontend is never served stale.
        """
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        """
        Route the per-request access log through logging.

        The page polls for the live scene while no demo runs; those misses would
        flood the console every second and are not news, so they stay out of the log.

        :param format:``printf``-style log message format.
        :param args: Values to interpolate into ``format``.
        """
        message = format % args if args else format
        if paths.LIVE_SCENE_NAME in message and " 404 " in message:
            return
        logger.info("  %s", message)

    # %% helpers
    def _send_json(self, payload: Any, code: int = 200) -> None:
        """
        Send a payload as JSON with the given status code.

        ``payload`` may be a plain JSON-able dict or a :class:`CrameraPayload` — either
        is serialized the same way.

        :param payload: The payload to serialize and send.
        :param code: HTTP status code to respond with.
        """
        if isinstance(payload, CrameraPayload):
            payload = payload.to_payload()
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query_parameters(self) -> Dict[str, List[str]]:
        """
        The parsed query-string parameters of the current request.
        """
        return parse_qs(urlparse(self.path).query)

    def _requested_scene(self) -> Optional[str]:
        """
        The scene the request targets, or None to let the server pick the active one.

        The frontend switches scenes by reloading with a ``?scene=`` parameter, so every
        API route has to honour it or the panels would disagree about what is on screen.
        """
        requested = self._query_parameters().get("scene")
        return requested[0] if requested else None

    def _guarded(self, handler: Callable[[], Any]) -> None:
        """
        Run an API handler; report exceptions as a JSON error payload.

        :param handler: The handler to run, returning the payload to send on success.
        """
        if not EQL_AVAILABLE:
            return self._send_error(self.NO_EQL_MESSAGE)
        try:
            return self._send_json(handler())
        except Exception as error:
            return self._send_exception(error)

    # %% scene bundles (generated data, lives outside the package)
    def _serve_scene_file(self, url_path: str) -> None:
        """
        Serve one file of a scene bundle, with path-traversal protection.

        :param url_path: The request path, starting with ``/scenes/``.
        """
        relative_path = url_path[len("/scenes/") :]
        base = paths.scenes_directory().resolve()
        target = (base / relative_path).resolve()
        if not str(target).startswith(str(base) + os.sep) and target != base:
            self.send_response(403)
            self.end_headers()
            return
        if not target.is_file():
            self.send_response(404)
            self.end_headers()
            return
        content_type = (
            mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        )
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # %% routes
    def do_GET(self) -> None:
        """
        Route static files, scene bundles and the read-only API.
        """
        route = self.path.split("?")[0]
        if route.startswith("/scenes/"):
            return self._serve_scene_file(route)
        scene = self._requested_scene()
        if route == "/api/knowledge":
            return self._guarded(
                lambda: GraphPanelViews.of_scene(scene).for_tab("knowledge")
            )
        if route == "/api/knowledge/view":
            name = (self._query_parameters().get("name") or ["knowledge"])[0]
            return self._guarded(lambda: GraphPanelViews.of_scene(scene).for_tab(name))
        if route == "/api/knowledge/expand":
            node = (self._query_parameters().get("node") or [""])[0]
            return self._guarded(lambda: self._expanded_node(node, scene))
        if route == "/api/models/state":
            if not PROBABILISTIC_MODELS_AVAILABLE:
                return self._send_error(NO_MODELS_MESSAGE)
            return self._guarded(lambda: ModelWorkbench.active().state())
        return super().do_GET()

    @staticmethod
    def _expanded_node(node: str, scene: Optional[str]) -> Any:
        """
        The node's subgraph, or a "not drillable" error if it has none.

        :param node: Id of the double-clicked node to expand.
        :param scene: Name of the scene the node belongs to, or None for the active one.
        """
        payload = GraphPanelViews.of_scene(scene).for_node(node)
        return payload if payload else {"ok": False, "error": "not drillable"}

    def do_POST(self) -> None:
        """
        Route the write-ish endpoints: EQL queries and the models workbench.
        """
        route = self.path.split("?")[0]
        if route == "/api/eql":
            return self._run_eql()
        if route.startswith("/api/models/"):
            return self._run_models_request(route)
        return self._send_error("unknown endpoint", 404)

    def _run_eql(self) -> None:
        """
        Execute an EQL query.
        """
        if not EQL_AVAILABLE:
            return self._send_error(self.NO_EQL_MESSAGE)
        try:
            request_body = self._request_body()
            code = (request_body.get("code") or "").strip()
            if not code:
                return self._send_error("empty query")
            with _EQL_LOCK:
                session = EqlSession.of_scene(self._requested_scene())
                return self._send_json(session.run(code))
        except Exception as error:
            # a SyntaxError from the query is named by its own type, like any other
            return self._send_exception(error)

    def _run_models_request(self, route: str) -> None:
        """
        Answer one models-workbench request.

        :param route: The request path below ``/api/models/``.
        """
        if not PROBABILISTIC_MODELS_AVAILABLE:
            return self._send_error(NO_MODELS_MESSAGE)
        try:
            body = self._request_body()
            workbench = ModelWorkbench.active()
            if route == "/api/models/load":
                # ``model_text`` is the uploaded file verbatim: python's JSON reader
                # accepts the Infinity/NaN literals circuit files carry, which the
                # browser's own parser would reject
                model_data = body.get("model")
                if model_data is None:
                    model_data = json.loads(body.get("model_text") or "{}")
                return self._send_json(
                    workbench.load_model(model_data, name=body.get("name") or "")
                )
            if route == "/api/models/probability":
                return self._send_json(
                    workbench.probability(
                        body.get("query") or [], body.get("evidence") or []
                    )
                )
            if route == "/api/models/posterior":
                return self._send_json(
                    workbench.posterior(
                        body.get("variables") or [], body.get("evidence") or []
                    )
                )
            if route == "/api/models/mode":
                return self._send_json(workbench.mode(body.get("evidence") or []))
            return self._send_error("unknown endpoint", 404)
        except Exception as error:
            return self._send_exception(error)

    def _request_body(self) -> Dict[str, Any]:
        """
        The request's JSON body, or an empty mapping without one.
        """
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _send_error(self, message: str, code: int = 200) -> None:
        """
        Send an error payload; the panels render ``error`` as their message.

        :param message: What went wrong, in the panel's own words.
        :param code: HTTP status code to respond with.
        """
        self._send_json({"ok": False, "error": message}, code)

    def _send_exception(self, error: Exception) -> None:
        """
        Send an exception as an error payload, named by its type.

        :param error: The exception to report.
        """
        self._send_error("%s: %s" % (type(error).__name__, error))


def make_server(port: int = 0) -> socketserver.ThreadingTCPServer:
    """
    A ready-to-serve ThreadingTCPServer (port 0 = ephemeral, for tests).

    :param port: Port to listen on, or 0 for an ephemeral port.
    """
    socketserver.TCPServer.allow_reuse_address = True
    return socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)


NO_BROWSER_FLAG = "--no-browser"
"""
CLI flag that keeps the server from opening the viewer page on start.
"""


@dataclass(frozen=True)
class ServerOptions:
    """
    What the ``cramera`` command line asks for.
    """

    port: int = DEFAULT_PORT
    """
    Port the server listens on.
    """

    open_browser: bool = True
    """
    Whether the viewer page is opened in the default browser on start.
    """


def parse_arguments(arguments: List[str]) -> ServerOptions:
    """
    Read the ``cramera`` command line: an optional port and ``--no-browser``.

    :param arguments: The command-line arguments, without the program name.
    """
    open_browser = NO_BROWSER_FLAG not in arguments
    ports = [argument for argument in arguments if argument != NO_BROWSER_FLAG]
    port = int(ports[0]) if ports else DEFAULT_PORT
    return ServerOptions(port=port, open_browser=open_browser)


def main(arguments: Optional[List[str]] = None) -> None:
    """
    ``cramera`` — serve the viewer, the scenes and the JSON API.

    Opens the viewer page in the default browser once the server is up; demos only
    ever connect to it, so this is the one deliberate moment a page appears. Pass
    ``--no-browser`` to skip it (a headless or remote server).

    :param arguments: Command-line arguments, or None to use ``sys.argv``.
    """
    # force: an imported CRAM package may already have configured the root logger,
    # which would otherwise make this call a no-op and swallow the startup output
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    options = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    port = options.port
    if EQL_AVAILABLE:  # build the knowledge base once, before the first query
        EpisodeKnowledgeBase.of_active_scene()
    with make_server(port) as server:
        eql = "EQL ready (krrood)" if EQL_AVAILABLE else "EQL unavailable — static only"
        scenes = paths.scenes_directory()
        logger.info("cramera running at http://localhost:%d/ (%s)", port, eql)
        logger.info(
            "scene bundles: %s%s",
            scenes,
            "" if Path(scenes).is_dir() else "  (missing — run cramera-onboard)",
        )
        if options.open_browser:
            webbrowser.open("http://localhost:%d/" % port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

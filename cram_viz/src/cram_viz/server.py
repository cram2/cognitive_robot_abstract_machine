"""
The cram_viz HTTP server: static frontend + JSON API.

Serves three things from one port (default 8711):

  * the packaged web frontend (``cram_viz/web`` — panels, vendored libs)
  * scene bundles from :func:`cram_viz.paths.scenes_dir` under ``/scenes/``
  * the JSON API the panels talk to:

      GET  /api/kb              the knowledge-graph overview payload
      GET  /api/kb/view?name=   one graph tab (knowledge/kinematics/plan/chart)
      GET  /api/kb/expand?node= drill-down subgraph for one node
      POST /api/eql             run an EQL query string

The API needs krrood (EQL). Without it the server still serves the viewer and
answers API calls with ``{"ok": false, "error": ...}`` so the frontend can say
why the knowledge panel is empty.

    cram-viz            # console script
    python -m cram_viz.server [port]
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
from pathlib import Path
from typing_extensions import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from cram_viz import get_logger, paths

logger = get_logger(__name__)

DEFAULT_PORT = 8711

try:
    import krrood  # noqa: F401  (the EQL engine)

    from cram_viz import knowledge as knowledge_module
except ImportError:  # pragma: no cover - depends on the environment
    knowledge_module = None
    logger.warning("krrood not importable — serving the viewer without the EQL API")
except Exception:  # pragma: no cover - a broken KB should not kill the viewer
    knowledge_module = None
    traceback.print_exc()


#: krrood's SymbolGraph singleton is not threadsafe; queries are serialized
_EQL_LOCK = threading.Lock()


def _no_eql_error() -> Dict[str, Any]:
    """
    The standard error payload for any API route when krrood isn't importable.
    """
    return {"ok": False, "error": "krrood/EQL not available in this environment"}


class Handler(http.server.SimpleHTTPRequestHandler):
    """
    Static files from the packaged web root, plus the JSON API routes.
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

        :param format: ``printf``-style log message format.
        :param args: Values to interpolate into ``format``.
        """
        logger.info("  " + format, *args)

    # %% helpers
    def _json(self, payload: Any, code: int = 200) -> None:
        """
        Send a payload as JSON with the given status code.

        ``payload`` may be a plain JSON-able dict, or a knowledge-package payload
        dataclass exposing ``to_payload()`` — either is serialized the same way.

        :param payload: The payload to serialize and send.
        :param code: HTTP status code to respond with.
        """
        if hasattr(payload, "to_payload"):
            payload = payload.to_payload()
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query(self) -> Dict[str, List[str]]:
        """
        The parsed query-string parameters of the current request.
        """
        return parse_qs(urlparse(self.path).query)

    def _guarded(self, fn: Callable[[], Any]) -> None:
        """
        Run an API handler; report exceptions as a JSON error payload.

        :param fn: The handler to run, returning the payload to send on success.
        """
        if knowledge_module is None:
            return self._json(_no_eql_error())
        try:
            return self._json(fn())
        except Exception as ex:
            return self._json(
                {"ok": False, "error": "%s: %s" % (type(ex).__name__, ex)}
            )

    # %% scene bundles (generated data, lives outside the package)
    def _serve_scene_file(self, url_path: str) -> None:
        """
        Serve one file of a scene bundle, with path-traversal protection.

        :param url_path: The request path, starting with ``/scenes/``.
        """
        rel = url_path[len("/scenes/") :]
        base = paths.scenes_dir().resolve()
        target = (base / rel).resolve()
        if not str(target).startswith(str(base) + os.sep) and target != base:
            self.send_response(403)
            self.end_headers()
            return
        if not target.is_file():
            self.send_response(404)
            self.end_headers()
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
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
        if route == "/api/kb":
            return self._guarded(lambda: knowledge_module.graph_payload())
        if route == "/api/kb/view":
            name = (self._query().get("name") or ["knowledge"])[0]
            return self._guarded(lambda: knowledge_module.view_payload(name))
        if route == "/api/kb/expand":
            node = (self._query().get("node") or [""])[0]

            def expand() -> Dict[str, Any]:
                """
                The node's subgraph, or a "not drillable" error if it has none.
                """
                payload = knowledge_module.expand_node(node)
                return payload if payload else {"ok": False, "error": "not drillable"}

            return self._guarded(expand)
        return super().do_GET()

    def do_POST(self) -> None:
        """
        Execute an EQL query (the only write-ish endpoint).
        """
        if self.path.split("?")[0] != "/api/eql":
            return self._json({"ok": False, "error": "unknown endpoint"}, 404)
        if knowledge_module is None:
            return self._json(_no_eql_error())
        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length) or b"{}")
            code = (req.get("code") or "").strip()
            if not code:
                return self._json({"ok": False, "error": "empty query"})
            with _EQL_LOCK:
                return self._json(knowledge_module.run_query(code))
        except SyntaxError as ex:
            return self._json({"ok": False, "error": "SyntaxError: %s" % ex})
        except Exception as ex:
            return self._json(
                {"ok": False, "error": "%s: %s" % (type(ex).__name__, ex)}
            )


def make_server(port: int = 0) -> socketserver.ThreadingTCPServer:
    """
    A ready-to-serve ThreadingTCPServer (port 0 = ephemeral, for tests).

    :param port: Port to listen on, or 0 for an ephemeral port.
    """
    socketserver.TCPServer.allow_reuse_address = True
    return socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)


def main(argv: Optional[List[str]] = None) -> None:
    """
    ``cram-viz`` — serve the viewer, the scenes and the JSON API.

    :param argv: Command-line arguments, or None to use ``sys.argv``.
    """
    # force: an imported CRAM package may already have configured the root logger,
    # which would otherwise make this call a no-op and swallow the startup output
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    argv = sys.argv[1:] if argv is None else argv
    port = int(argv[0]) if argv else DEFAULT_PORT
    if (
        knowledge_module is not None
    ):  # build the knowledge base once, before the first query
        knowledge_module.get_knowledge_base()
    with make_server(port) as httpd:
        eql = (
            "EQL ready (krrood)"
            if knowledge_module is not None
            else "EQL unavailable — static only"
        )
        scenes = paths.scenes_dir()
        logger.info("cram_viz running at http://localhost:%d/ (%s)", port, eql)
        logger.info(
            "scene bundles: %s%s",
            scenes,
            "" if Path(scenes).is_dir() else "  (missing — run cram-viz-onboard)",
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

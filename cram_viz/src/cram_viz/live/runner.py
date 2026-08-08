"""
Starting the live bridge: as a library call or as the demo wrapper CLI.
"""

from __future__ import annotations

import logging
import runpy
import signal
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

from typing_extensions import Optional, TYPE_CHECKING

from cram_viz import get_logger
from cram_viz.live import hooks
from cram_viz.live.bridge import BRIDGE
from cram_viz.live.http import DEFAULT_PORT, serve

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

logger = get_logger(__name__)


def start(
    world: Optional[World] = None, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """
    Start the live bridge, or return the already running one.

    Safe to call more than once: ``cram-viz-live demo.py`` starts the bridge and the
    demo may call this again itself, which must not re-patch the hooks or bind the
    port a second time.

    :param world: Bind to this world immediately; without it the bridge attaches to the
        executing world on the first executor tick.
    :param port: Port of the bridge's HTTP endpoints.
    :return: The running HTTP server (a daemon thread).
    """
    bridge = BRIDGE
    if bridge.live_server is not None:
        logger.info("live bridge is already running — reusing it")
        return bridge.live_server
    hooks.install_mesh_hook()  # before the demo parses its objects
    hooks.install_plan_hooks()
    if world is not None:
        bridge.world = world
        bridge.bind()
        bridge.snapshot()  # single-threaded here, before execution starts
    hooks.install_tick_hook()
    bridge.live_server = serve(bridge, port)
    logger.info(
        "bridge on http://localhost:%d (the viewer shows a Live button "
        "while this runs)",
        port,
    )
    return bridge.live_server


def main() -> None:
    """
    ``cram-viz-live path/to/demo.py`` — run a demo with the live bridge.

    The demo's own directory is put on ``sys.path`` so the demo can import its local
    helper modules, exactly as if it were started directly. After the demo finishes the
    bridge stays up for inspection until Ctrl-C.
    """
    # force: importing the hooks pulls in coraplex, which configures the root logger
    logging.basicConfig(level=logging.INFO, force=True)
    if len(sys.argv) < 2:
        sys.exit("usage: cram-viz-live path/to/demo.py")
    demo = Path(sys.argv[1]).resolve()
    start()
    sys.path.insert(0, str(demo.parent))
    logger.info("running demo: %s", demo)
    runpy.run_path(str(demo), run_name="__main__")
    logger.info("demo finished — bridge stays up for inspection (Ctrl-C to quit)")
    try:
        signal.pause()  # wait for Ctrl-C (SIGINT) instead of a sleep loop
    except KeyboardInterrupt:
        pass

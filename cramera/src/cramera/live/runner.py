"""
Starting the live viewer: as a library call or as the demo wrapper CLI.
"""

from __future__ import annotations

import logging
import os
import runpy
import signal
import sys
from pathlib import Path

from typing_extensions import TYPE_CHECKING

from cramera.live.http import DEFAULT_PORT
from cramera.logging_setup import get_logger

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

    from cramera.live.visualization import LiveVisualization

logger = get_logger(__name__)

VISUALIZATION_BACKEND_VARIABLE = "CORAPLEX_VISUALIZATION"
"""
The coraplex environment variable selecting the visualization backend.
"""


def start(world: World, port: int = DEFAULT_PORT) -> LiveVisualization:
    """
    Serve a world to the browser viewer.

    Prefer :func:`coraplex.testing.start_visualization` (or
    :class:`coraplex.visualization.WorldVisualization` directly) inside demos — this is
    the cramera-side entry point they delegate to.

    :param world: The world to serve.
    :param port: Port of the bridge's HTTP endpoints.
    :return: The started visualization.
    """
    from cramera.live.visualization import LiveVisualization

    return LiveVisualization(world=world, port=port).start()


def main() -> None:
    """
    ``cramera-live path/to/demo.py`` — run a demo with the browser viewer.

    Selects the cramera backend through ``CORAPLEX_VISUALIZATION`` and runs the demo
    unchanged: the demo's own ``start_visualization`` call picks the backend up. The
    demo's directory is put on ``sys.path`` so it can import its local helper modules,
    exactly as if it were started directly. After the demo finishes the bridge stays
    up for inspection until Ctrl-C.
    """
    logging.basicConfig(level=logging.INFO, force=True)
    if len(sys.argv) < 2:
        sys.exit("usage: cramera-live path/to/demo.py")
    demo = Path(sys.argv[1]).resolve()
    os.environ.setdefault(VISUALIZATION_BACKEND_VARIABLE, "cramera")
    sys.path.insert(0, str(demo.parent))
    logger.info("running demo: %s", demo)
    runpy.run_path(str(demo), run_name="__main__")
    logger.info("demo finished — bridge stays up for inspection (Ctrl-C to quit)")
    try:
        signal.pause()  # wait for Ctrl-C (SIGINT) instead of a sleep loop
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

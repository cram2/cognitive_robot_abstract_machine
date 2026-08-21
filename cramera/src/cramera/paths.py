"""
Filesystem locations for cramera, all overridable via environment.

The frontend (``web/``) ships inside the package. Scene bundles are *generated*
artifacts (tens to hundreds of MB per scene, produced by ``cramera-onboard``)
and are deliberately not part of this repository — they are versioned in
https://github.com/cram2/cram-scenes, wired in as the *optional* submodule
``cramera/scenes`` (live visualization and freshly onboarded scenes work
without it). :func:`scenes_directory` looks in this order:

    1. CRAMERA_SCENES=/path/to/scenes        explicit override
    2. cramera/scenes                        the submodule, if initialized
                                              (git submodule update --init cramera/scenes)
    3. ~/.cramera/scenes                     default data directory

    CRAMERA_ARCHITECTURE=/path/to/repo       CRAM repo scanned by the knowledge graph
"""

from __future__ import annotations

import os
from pathlib import Path

from typing_extensions import Optional

WEB_ROOT = Path(__file__).resolve().parent / "web"
"""
The packaged frontend: index.html, panels, vendored libraries.
"""


def _configured_path(variable: str) -> Optional[Path]:
    """
    The path an environment variable overrides a default with, if it is set.

    :param variable: Name of the environment variable to read.
    """
    value = os.environ.get(variable)
    return Path(value).expanduser() if value else None


def data_directory() -> Path:
    """
    Writable per-user data directory (architecture scan cache, defaults).
    """
    return _configured_path("CRAMERA_DATA") or Path.home() / ".cramera"


SCENES_SUBMODULE = WEB_ROOT.parents[2] / "scenes"
"""
The optional cram-scenes submodule checkout (``<member dir>/scenes``).
"""


def scenes_directory() -> Path:
    """
    Directory holding the onboarded scene bundles (``<name>/scene.json``).

    Search order: the ``CRAMERA_SCENES`` environment variable, then the initialized
    cram-scenes submodule, then ``~/.cramera/scenes``. An un-initialized
    submodule is an empty directory and is skipped (index.json is the marker).
    """
    configured = _configured_path("CRAMERA_SCENES")
    if configured:
        return configured
    if (SCENES_SUBMODULE / "index.json").is_file():
        return SCENES_SUBMODULE
    return data_directory() / "scenes"


def architecture_root() -> Path:
    """
    The CRAM repository whose packages/classes the knowledge graph shows.

    Defaults to the repository this package is checked out in, which is the common case
    inside the workspace; falls back to the conventional clone location otherwise.
    """
    configured = _configured_path("CRAMERA_ARCHITECTURE")
    if configured:
        return configured
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "coraplex").is_dir() and (parent / "krrood").is_dir():
            return parent
    return Path.home() / "cognitive_robot_abstract_machine"

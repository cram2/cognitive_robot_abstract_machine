"""
Filesystem locations for cram_viz, all overridable via environment.

The frontend (``web/``) ships inside the package. Scene bundles are *generated*
artifacts (tens to hundreds of MB per scene, produced by ``cram-viz-onboard``)
and are deliberately not part of this repository — they are versioned in
https://github.com/cram2/cram-scenes, wired here as the *optional* submodule
``cram_viz/scenes`` (live visualization and freshly onboarded scenes work
without it). :func:`scenes_dir` looks in this order:

    1. CRAM_VIZ_SCENES=/path/to/scenes        explicit override
    2. cram_viz/scenes                        the submodule, if initialized
                                              (git submodule update --init cram_viz/scenes)
    3. ~/.cram_viz/scenes                     default data directory

    CRAM_VIZ_ARCHITECTURE=/path/to/repo       CRAM repo scanned by the KB graph
"""

from __future__ import annotations

import os
from pathlib import Path

#: the packaged frontend: index.html, panels, vendored libraries
WEB_ROOT = Path(__file__).resolve().parent / "web"


def data_dir() -> Path:
    """
    Writable per-user data directory (architecture scan cache, defaults).
    """
    env = os.environ.get("CRAM_VIZ_DATA")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cram_viz"


#: the optional cram-scenes submodule checkout (``<member dir>/scenes``)
SCENES_SUBMODULE = WEB_ROOT.parents[2] / "scenes"


def scenes_dir() -> Path:
    """
    Directory holding the onboarded scene bundles (``<name>/scene.json``).

    Search order: ``CRAM_VIZ_SCENES`` env var, then the initialized
    cram-scenes submodule, then ``~/.cram_viz/scenes``. An un-initialized
    submodule is an empty directory and is skipped (index.json is the marker).
    """
    env = os.environ.get("CRAM_VIZ_SCENES")
    if env:
        return Path(env).expanduser()
    if (SCENES_SUBMODULE / "index.json").is_file():
        return SCENES_SUBMODULE
    return data_dir() / "scenes"


def architecture_root() -> Path:
    """
    The CRAM repository whose packages/classes the knowledge graph shows.

    Defaults to the repository this package is checked out in, which is the common case
    inside the workspace; falls back to the conventional clone location otherwise.
    """
    env = os.environ.get("CRAM_VIZ_ARCHITECTURE")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "coraplex").is_dir() and (parent / "krrood").is_dir():
            return parent
    return Path.home() / "cognitive_robot_abstract_machine"

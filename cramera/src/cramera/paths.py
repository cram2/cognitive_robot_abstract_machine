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

from typing_extensions import List, Optional

WEB_ROOT = Path(__file__).resolve().parent / "web"
"""
The packaged frontend: index.html, panels, vendored libraries.
"""

LIVE_SCENE_NAME = "__live__"
"""
Reserved scene name a live-attach snapshot is bundled under (see
:mod:`cramera.live.live_bundle`), rebuilt from the running demo's current world on every
attach.

Excluded from the real scene index — never a bundle a user onboarded.
"""

RECORDING_SCENE_NAME = "__recording__"
"""
Reserved scene name a captured live run is bundled under while unsaved (see
:mod:`cramera.live.recording_bundle`), analogous to :data:`LIVE_SCENE_NAME`. Always
written under :func:`local_scenes_directory`, never inside a shared scenes root, so
saving or discarding it never touches a git-tracked ``cram-scenes`` checkout.

Excluded from the real scene index — never a bundle a user onboarded.
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
    return local_scenes_directory()


def local_scenes_directory() -> Path:
    """
    Writable, local-only root for live recordings (temporary and saved).

    Deliberately ignores ``CRAMERA_SCENES`` and the cram-scenes submodule: a recording
    must never land inside a shared, git-tracked scenes root, even when one is checked
    out — saving a captured live run is a local action, not a contribution to it.
    """
    return data_directory() / "scenes"


def scene_roots() -> List[Path]:
    """
    Every directory a named scene may be bundled under, local root first.

    A local recording shadows a shared scene of the same name. Returns one entry when
    :func:`scenes_directory` already resolves to :func:`local_scenes_directory` (the
    common case, no shared submodule checked out), else both.
    """
    shared = scenes_directory()
    local = local_scenes_directory()
    return [shared] if local == shared else [local, shared]


def resolve_scene_directory(name: str) -> Optional[Path]:
    """
    The bundle directory a scene name resolves to, searched local-first, or None if no
    root has it.

    :param name: Name of the scene to look up.
    """
    for root in scene_roots():
        candidate = root / name
        if (candidate / "scene.json").is_file():
            return candidate
    return None


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

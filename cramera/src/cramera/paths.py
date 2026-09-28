"""
Filesystem locations for the packaged viewer and generated scene bundles.

Live recordings are saved under ``CRAMERA_DATA`` or the user's data directory.
``CRAMERA_SCENES`` selects an additional scene archive. A local ``cramera/scenes``
archive is also discovered when it contains an index.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from typing_extensions import List, Optional

WEB_ROOT = Path(__file__).resolve().parent / "web"
"""
The packaged frontend: index.html, panels, vendored libraries.
"""

SCENE_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
"""
Single-segment names accepted for saved recordings and internal captures.
"""

LIVE_SCENE_NAME = "__live__"
"""
Reserved scene name a live-attach snapshot is bundled under (see
:mod:`cramera.live.live_bundle`), rebuilt from the running demo's current world on every
attach. Written under :func:`local_scenes_directory`, like :data:`RECORDING_SCENE_NAME`:
a shared copy would be shadowed by a stale local one and would litter a git-tracked
checkout.

Excluded from the real scene index — never a bundle a user recorded.
"""

RECORDING_SCENE_NAME = "__recording__"
"""
Reserved scene name a captured live run is bundled under while unsaved (see
:mod:`cramera.live.recording_bundle`), analogous to :data:`LIVE_SCENE_NAME`. Always
written under :func:`local_scenes_directory`, never inside a shared scenes root, so
saving or discarding it never touches the shared scene archive.

Excluded from the real scene index — never a bundle a user recorded.
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
An optional local scene archive (``<member dir>/scenes``).
"""


def scenes_directory() -> Path:
    """
    Directory holding the recorded scene bundles (``<name>/scene.json``).

    Search order: the ``CRAMERA_SCENES`` environment variable, then the initialized
    local scene archive, then ``~/.cramera/scenes``. An empty archive directory is skipped (index.json is the marker).
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

    Deliberately ignores ``CRAMERA_SCENES`` and the local scene archive: a recording
    must never land inside a shared, git-tracked scenes root, even when one is checked
    out — saving a captured live run is a local action, not a contribution to it.
    """
    return data_directory() / "scenes"


def scene_roots() -> List[Path]:
    """
    Every directory a named scene may be bundled under, local root first.

    A local recording shadows a shared scene of the same name. Returns one entry when
    :func:`scenes_directory` already resolves to :func:`local_scenes_directory` (the
    common case, no shared scene archive configured), else both.
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
    if not SCENE_NAME_PATTERN.fullmatch(name):
        return None
    for root in scene_roots():
        candidate = root / name
        if (
            candidate.resolve().is_relative_to(root.resolve())
            and (candidate / "scene.json").is_file()
        ):
            return candidate
    return None


def repository_root() -> Path:
    """
    The CRAM repository this package is running from.

    Falls back to the conventional clone location when the package is installed outside
    a checkout, which then simply holds none of its files.
    """
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "coraplex").is_dir() and (parent / "krrood").is_dir():
            return parent
    return Path.home() / "cognitive_robot_abstract_machine"


def architecture_root() -> Path:
    """
    The CRAM repository whose packages/classes the knowledge graph shows.

    Defaults to the repository this package is running from, which is the common case
    inside the workspace; ``CRAMERA_ARCHITECTURE`` points the scan at another checkout.
    """
    return _configured_path("CRAMERA_ARCHITECTURE") or repository_root()

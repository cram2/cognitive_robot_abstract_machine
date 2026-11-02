"""
Which packages ``scripts/sync_version.py`` writes a version into, and where.

The script is a standalone file rather than part of an importable package, so it is
loaded from its path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SYNC_VERSION_SCRIPT = Path(__file__).parents[2] / "scripts" / "sync_version.py"
"""
The script under test.
"""


def load_sync_version() -> ModuleType:
    """
    Registered under its name before it runs, as an import would be, since a dataclass
    looks its own module up while it is defined.

    :return: The script, imported from its file.
    """
    specification = importlib.util.spec_from_file_location(
        SYNC_VERSION_SCRIPT.stem, SYNC_VERSION_SCRIPT
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


sync_version = load_sync_version()

repository = sync_version.Repository.of_this_script()


def test_every_version_module_in_the_repository_is_found():
    """
    A package is found from its own ``pyproject.toml``, so adding one needs no edit to
    the script - and one the script missed would keep an old version.
    """
    on_disk = set(repository.root.glob("*/_version.py")) | set(
        repository.root.glob("*/src/*/_version.py")
    )

    assert {
        package.version_file for package in repository.versioned_packages()
    } == on_disk

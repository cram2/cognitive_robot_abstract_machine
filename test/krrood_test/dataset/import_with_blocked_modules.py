"""
Imports modules, or runs a script, in an interpreter in which the blocked modules look
uninstalled.

Run as ``python import_with_blocked_modules.py --blocked MODULE... (--modules MODULE... |
--script PATH)``; the exit status is non-zero when one of them needs a blocked module.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import runpy
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

from krrood.patterns.called_positionally import called_positionally

# %% blocking


@dataclass(kw_only=True)
class ModuleBlocker(importlib.abc.MetaPathFinder):
    """
    Makes the blocked modules, and every module below them, fail to import as if they
    were not installed, while leaving them out of :data:`sys.modules`.
    """

    blocked: frozenset[str]
    """
    The dotted names of the blocked modules.
    """

    @called_positionally
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        """
        :raises ModuleNotFoundError: When the module is blocked or lies below a blocked one.
        """
        parts = fullname.split(".")
        prefixes = {".".join(parts[: length + 1]) for length in range(len(parts))}
        if prefixes & self.blocked:
            raise ModuleNotFoundError(fullname, name=fullname)
        return None

    def install(self) -> None:
        """
        Put the blocker ahead of every other finder, drop the search path entries a
        blocked distribution added, such as a directory that replaces a standard library
        module, and forget the krrood modules loaded so far, so they load again.
        """
        sys.meta_path.insert(0, self)
        for name in [
            name for name in sys.modules if name.partition(".")[0] == "krrood"
        ]:
            del sys.modules[name]
        sys.path[:] = [
            entry for entry in sys.path if Path(entry).name not in self.blocked
        ]


# %% running


def main() -> None:
    """
    Block the named modules, then import the modules or run the script given.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocked", nargs="*", default=[])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--modules", nargs="+")
    target.add_argument("--script", type=Path)
    arguments = parser.parse_args()
    ModuleBlocker(blocked=frozenset(arguments.blocked)).install()
    if arguments.script is not None:
        runpy.run_path(str(arguments.script), run_name="__main__")
        return
    for module in arguments.modules:
        importlib.import_module(module)


if __name__ == "__main__":
    main()

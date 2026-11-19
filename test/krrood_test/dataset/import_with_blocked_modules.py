"""
Imports modules, or runs a script, in an interpreter in which the blocked modules cannot
be imported.

Run as ``python import_with_blocked_modules.py --blocked MODULE... (--modules MODULE... |
--script PATH)``; the exit status is non-zero when one of them needs a blocked module.
"""

from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from pathlib import Path

# %% running


def main() -> None:
    """
    Block the named modules, then import the modules or run the script given.

    A module mapped to ``None`` in ``sys.modules`` raises ``ImportError`` when imported.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocked", nargs="*", default=[])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--modules", nargs="+")
    target.add_argument("--script", type=Path)
    arguments = parser.parse_args()
    sys.modules.update(dict.fromkeys(arguments.blocked))
    if arguments.script is not None:
        runpy.run_path(str(arguments.script), run_name="__main__")
        return
    for module in arguments.modules:
        importlib.import_module(module)


if __name__ == "__main__":
    main()

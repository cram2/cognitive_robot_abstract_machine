"""
Imports a module in an interpreter in which the named libraries cannot be imported.

Run as ``python import_with_blocked_libraries.py MODULE LIBRARY...``; the exit status is
non-zero when importing MODULE needs one of the libraries.
"""

from __future__ import annotations

import importlib
import sys

# %% running


def main() -> None:
    """
    Block the libraries named on the command line and import the module named first.

    A library mapped to ``None`` in ``sys.modules`` raises ``ImportError`` when
    imported.
    """
    module, *libraries = sys.argv[1:]
    sys.modules.update(dict.fromkeys(libraries))
    importlib.import_module(module)


if __name__ == "__main__":
    main()

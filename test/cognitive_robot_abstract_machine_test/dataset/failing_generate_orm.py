"""
Stand-in for an ORM interface generator that fails.

Writes a diagnostic the way a real generator's traceback would, and exits without having
written an interface.
"""

from __future__ import annotations

import sys

DIAGNOSTIC = "ModuleNotFoundError: no module named 'a_package_that_is_not_there'"
"""
What this generator writes before giving up.
"""


def main() -> None:
    """
    Write the diagnostic and exit as a failure.
    """
    print(DIAGNOSTIC, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()

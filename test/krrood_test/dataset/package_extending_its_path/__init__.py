"""
A package that adds a directory outside itself to its ``__path__`` when it is imported,
the way :func:`pkgutil.extend_path` does, so some of its submodules are found only once
the package has run.
"""

from pathlib import Path

__path__.append(str(Path(__file__).parent.parent / "modules_outside_their_package"))

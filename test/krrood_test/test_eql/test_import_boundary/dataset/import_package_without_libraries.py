"""
Imports every module of a package while the named top-level libraries cannot be
imported, exiting with an error at the first module that fails to import.

Run as ``python import_package_without_libraries.py <package> <library>...``.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec

from krrood.patterns.called_positionally import called_positionally

# %% blocking libraries


@dataclass(kw_only=True)
class LibraryBlocker(MetaPathFinder):
    """
    A meta path finder that makes the given top-level libraries fail to import, as if
    they were not installed.
    """

    blocked_libraries: frozenset[str]
    """
    The top-level names of the libraries that fail to import.
    """

    @called_positionally
    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> ModuleSpec | None:
        """
        :raises ModuleNotFoundError: If the module belongs to a blocked library.
        :return: ``None``, so the next finder looks for any other module.
        """
        if fullname.partition(".")[0] in self.blocked_libraries:
            raise ModuleNotFoundError(f"{fullname} is blocked", name=fullname)
        return None


# %% importing the package


def import_every_module(*, package_name: str) -> None:
    """
    :param package_name: The package whose modules, subpackages included, are imported.
    """
    package = importlib.import_module(package_name)
    for module in pkgutil.walk_packages(package.__path__, f"{package_name}."):
        importlib.import_module(module.name)


if __name__ == "__main__":
    sys.meta_path.insert(0, LibraryBlocker(blocked_libraries=frozenset(sys.argv[2:])))
    import_every_module(package_name=sys.argv[1])

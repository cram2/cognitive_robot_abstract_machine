"""
The modules that declare sections of ``AGENTS.md``.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from krrood.development_tools.agents.rule import Section


@dataclass(frozen=True)
class RuleModule:
    """
    A module, or a package with all of its modules, declaring sections of ``AGENTS.md``.
    """

    module: ModuleType
    """
    The module or package.
    """

    @classmethod
    def import_from(cls, directory: Path, module_name: str) -> RuleModule | None:
        """
        Import a module or package from a directory, once per process.

        :param directory: The directory holding the module or package.
        :param module_name: The name of the module or package.
        :return: The imported module; none if the directory holds neither.
        """
        specification = importlib.machinery.PathFinder.find_spec(
            module_name, [str(directory)]
        )
        if specification is None:
            return None
        if module_name in sys.modules:
            return cls(sys.modules[module_name])
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)
        return cls(module)

    @property
    def modules(self) -> list[ModuleType]:
        """
        The module itself and, for a package, every module in it, imported.
        """
        if self.module.__spec__.submodule_search_locations is None:
            return [self.module]
        return [self.module] + [
            importlib.import_module(submodule.name)
            for submodule in pkgutil.walk_packages(
                self.module.__path__, f"{self.module.__name__}."
            )
        ]

    @property
    def sections(self) -> list[type[Section]]:
        """
        The top-level sections defined in :attr:`modules`, in alphabetical order of
        their titles.

        Sections a module only imports are not among them.
        """
        sections = [
            value
            for module in self.modules
            for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, Section)
            and value.__module__ == module.__name__
            and value.parent is None
        ]
        return sorted(sections, key=lambda section: section.title)

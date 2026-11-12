"""
The modules that declare sections of ``AGENTS.md`` or skills.
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

from krrood.development_tools.agents.checker import CodeRuleChecker
from krrood.development_tools.agents.rule import Section
from krrood.development_tools.agents.skill import Skill


@dataclass(kw_only=True, frozen=True)
class RuleModule:
    """
    A module, or a package with all of its modules, declaring sections of ``AGENTS.md``
    or skills.
    """

    module: ModuleType
    """
    The module or package.
    """

    @classmethod
    def import_from(cls, *, directory: Path, module_name: str) -> RuleModule | None:
        """
        Import a module or package from a directory, once per process.

        The directory joins the import path, so the module can import the rest of its
        repository without the repository being installed.

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
            return cls(module=sys.modules[module_name])
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)
        return cls(module=module)

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

    @property
    def checkers(self) -> tuple[CodeRuleChecker, ...]:
        """
        The checker of every rule in :attr:`sections` and the sections nested in them.
        """
        return tuple(
            checker for section in self.sections for checker in section.checkers
        )

    @property
    def skills(self) -> list[type[Skill]]:
        """
        The skills defined in :attr:`modules`, in alphabetical order of their names.

        Skills a module only imports are not among them.
        """
        skills = [
            value
            for module in self.modules
            for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, Skill)
            and value.__module__ == module.__name__
        ]
        return sorted(skills, key=lambda skill: skill.name)

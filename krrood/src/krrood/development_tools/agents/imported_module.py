"""
A checked source file as the module Python imports, for the checks that need the classes
it defines and not only its syntax.
"""

from __future__ import annotations

import abc
import importlib
import sys
import sysconfig
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from types import ModuleType

from krrood.utils import get_import_path_from_path, make_path_importable

# %% the imported module


@dataclass(frozen=True, kw_only=True)
class ImportedModule:
    """
    A source file as the module Python imports it.
    """

    path: Path
    """
    The source file of the module.
    """

    @cached_property
    def module(self) -> ModuleType | None:
        """
        The module imported from :attr:`path`; none when it cannot be imported.
        """
        make_path_importable(self.path.parent)
        package = get_import_path_from_path(str(self.path.parent))
        name = self.path.stem if package is None else f"{package}.{self.path.stem}"
        try:
            module = importlib.import_module(name.removesuffix(".__init__"))
        except Exception:
            # Handle unimportable modules gracefully: whatever the import raises, only
            # the checks needing the module's classes are skipped.
            return None
        if module.__file__ is None or Path(module.__file__).resolve() != (
            self.path.resolve()
        ):
            return None
        return module

    def class_named(self, *, scope: tuple[str, ...] | None) -> type | None:
        """
        :param scope: The names of a class and the classes enclosing it; none for a
            class defined inside a function.
        :return: The class; none when it cannot be found.
        """
        if scope is None or self.module is None:
            return None
        found: ModuleType | type = self.module
        for name in scope:
            found = vars(found).get(name)
            if not isinstance(found, type):
                return None
        return found

    def implements_abstract_method(
        self, *, scope: tuple[str, ...] | None, method_name: str
    ) -> bool:
        """
        :param scope: The names of a class and the classes enclosing it.
        :param method_name: The name of a method of the class.
        :return: Whether the first base class defining *method_name* leaves it abstract.
        """
        for base in self.class_named(scope=scope).__mro__[1:]:
            if method_name in vars(base):
                return (
                    isinstance(base, abc.ABCMeta)
                    and method_name in base.__abstractmethods__
                )
        return False


# %% installed code


class InstallationDirectory(StrEnum):
    """
    The :func:`sysconfig.get_paths` keys of the directories installed code lies in.
    """

    STANDARD_LIBRARY = "stdlib"
    PLATFORM_STANDARD_LIBRARY = "platstdlib"
    PURE_PACKAGES = "purelib"
    PLATFORM_PACKAGES = "platlib"


@dataclass(frozen=True, kw_only=True)
class InstalledCode:
    """
    The standard library and the packages installed for the running interpreter, as
    opposed to the code of the repository being checked.
    """

    @cached_property
    def directories(self) -> tuple[Path, ...]:
        """
        The directories the standard library and the installed packages lie in.
        """
        paths = sysconfig.get_paths()
        return tuple(Path(paths[key]).resolve() for key in InstallationDirectory)

    def contains(self, *, cls: type) -> bool:
        """
        :return: Whether *cls* is built into the interpreter or defined in a module
            under :attr:`directories`.
        """
        module = sys.modules.get(cls.__module__)
        if module is None:
            return False
        file = vars(module).get("__file__")
        if file is None:
            return True
        path = Path(file).resolve()
        return any(path.is_relative_to(directory) for directory in self.directories)

    def defines_inherited_method(self, *, cls: type, method_name: str) -> bool:
        """
        :return: Whether a base class of *cls* that is installed code defines
            *method_name*.
        """
        return any(
            method_name in vars(base) and self.contains(cls=base)
            for base in cls.__mro__[1:]
        )

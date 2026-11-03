"""
Guards that keep krrood's core light: core is declared in krrood's ``pyproject.toml``, and
nothing outside the declared offenders may cross its boundary.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from pathlib import Path

import pytest
from importlinter.cli import EXIT_STATUS_SUCCESS, lint_imports
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from typing_extensions import Any

import krrood
from krrood.patterns.path_enumeration import PathEnumeration

# %% declaration of core


class DeclarationFile(PathEnumeration):
    """
    The files that declare krrood's core.
    """

    PYPROJECT = Path(krrood.__file__).parents[2] / "pyproject.toml"
    """
    krrood's ``pyproject.toml``.
    """


class ImportContractIdentifier(StrEnum):
    """
    The import-linter contracts krrood's ``pyproject.toml`` declares.
    """

    CORE_IMPORTS_NO_ALL_MODULE = "core-imports-no-all-module"
    """
    Core modules import no module of ``krrood[all]``.
    """


@dataclass
class CoreBoundary:
    """
    krrood's core and the distributions it may load, as krrood's ``pyproject.toml``
    declares them.
    """

    top_level_modules: list[str]
    """
    The modules and packages core consists of, the source modules of the contract
    :attr:`ImportContractIdentifier.CORE_IMPORTS_NO_ALL_MODULE`.
    """

    dependencies: list[str]
    """
    The distributions core may load, without what they require in turn.
    """

    modules_loading_other_distributions: list[str]
    """
    Today's core modules that load a distribution outside
    :attr:`allowed_distributions`; the list only ever shrinks.
    """

    @classmethod
    def from_json(cls, pyproject: dict[str, Any]) -> CoreBoundary:
        """
        :param pyproject: The parsed content of krrood's ``pyproject.toml``.
        :return: The core boundary it declares.
        """
        tool = pyproject["tool"]
        contract = next(
            contract
            for contract in tool["importlinter"]["contracts"]
            if contract["id"] == ImportContractIdentifier.CORE_IMPORTS_NO_ALL_MODULE
        )
        declaration = tool["krrood"]["core-boundary"]
        return cls(
            top_level_modules=contract["source_modules"],
            dependencies=declaration["dependencies"],
            modules_loading_other_distributions=declaration[
                "modules-loading-other-distributions"
            ],
        )

    @classmethod
    def from_pyproject(cls) -> CoreBoundary:
        """
        :return: The core boundary krrood's ``pyproject.toml`` declares.
        """
        return cls.from_json(tomllib.loads(DeclarationFile.PYPROJECT.read_text()))

    @cached_property
    def modules(self) -> list[str]:
        """
        Every module of core: each of :attr:`top_level_modules` and, for a package, every
        module below it.
        """
        modules = []
        for top_level_module in self.top_level_modules:
            modules.append(top_level_module)
            module = importlib.import_module(top_level_module)
            if hasattr(module, "__path__"):
                modules.extend(
                    submodule.name
                    for submodule in pkgutil.walk_packages(
                        module.__path__, f"{top_level_module}."
                    )
                )
        return modules

    @cached_property
    def modules_loading_only_allowed_distributions(self) -> list[str]:
        """
        The modules of core outside :attr:`modules_loading_other_distributions`.
        """
        return [
            module
            for module in self.modules
            if module not in self.modules_loading_other_distributions
        ]

    @cached_property
    def allowed_distributions(self) -> set[str]:
        """
        The canonical names of krrood, of :attr:`dependencies` and of every distribution
        they require, as installed.
        """
        allowed = {canonicalize_name(krrood.__name__)}
        pending = list(self.dependencies)
        while pending:
            distribution = canonicalize_name(pending.pop())
            if distribution in allowed:
                continue
            allowed.add(distribution)
            for requirement in map(
                Requirement, importlib.metadata.requires(distribution) or []
            ):
                if requirement.marker is None or requirement.marker.evaluate(
                    {"extra": ""}
                ):
                    pending.append(requirement.name)
        return allowed


# %% modules a fresh interpreter loads


@dataclass
class FreshInterpreter:
    """
    A fresh, isolated interpreter that reports, through ``python -X importtime``, the
    distributions an import loads.
    """

    distributions_by_top_level_module: dict[str, list[str]] = field(
        default_factory=importlib.metadata.packages_distributions
    )
    """
    The distributions that provide each installed top-level module.
    """

    def top_level_modules_loaded_by(self, source: str) -> set[str]:
        """
        :param source: The Python source the interpreter runs.
        :return: The top-level modules the interpreter loads while starting and running
            *source*.
        """
        import_report = subprocess.run(
            [sys.executable, "-I", "-X", "importtime", "-c", source],
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            line.rpartition("|")[2].strip().partition(".")[0]
            for line in import_report.stderr.splitlines()
            if line.startswith("import time:")
        }

    @cached_property
    def startup_modules(self) -> set[str]:
        """
        The top-level modules the interpreter loads before running anything.
        """
        return self.top_level_modules_loaded_by("")

    def distributions_loaded_by_importing(self, module: str) -> set[str]:
        """
        :param module: The module to import.
        :return: The canonical names of the distributions importing *module* loads,
            beyond what the interpreter loads at startup.
        """
        top_level_modules = (
            self.top_level_modules_loaded_by(f"import {module}") - self.startup_modules
        )
        return {
            canonicalize_name(distribution)
            for top_level_module in top_level_modules
            for distribution in self.distributions_by_top_level_module.get(
                top_level_module, []
            )
        }


@pytest.fixture(scope="module")
def fresh_interpreter() -> FreshInterpreter:
    """
    :return: One fresh interpreter for every import a test module checks.
    """
    return FreshInterpreter()


# %% runtime boundary


@pytest.mark.parametrize(
    "module", CoreBoundary.from_pyproject().modules_loading_only_allowed_distributions
)
def test_core_module_loads_only_core_distributions(
    module: str, fresh_interpreter: FreshInterpreter
) -> None:
    """
    Importing a core module loads nothing but the standard library, krrood and core's
    declared dependencies.
    """
    core_boundary = CoreBoundary.from_pyproject()

    loaded = fresh_interpreter.distributions_loaded_by_importing(module)

    assert loaded - core_boundary.allowed_distributions == set()


@pytest.mark.parametrize(
    "module", CoreBoundary.from_pyproject().modules_loading_other_distributions
)
def test_listed_offender_still_loads_other_distributions(
    module: str, fresh_interpreter: FreshInterpreter
) -> None:
    """
    A module listed as loading another distribution still does; once it no longer does,
    it is removed from the list, so the list only ever shrinks.
    """
    core_boundary = CoreBoundary.from_pyproject()

    loaded = fresh_interpreter.distributions_loaded_by_importing(module)

    assert loaded - core_boundary.allowed_distributions != set()


def test_listed_offenders_are_core_modules() -> None:
    """
    Every module listed as loading another distribution is a module of core.
    """
    core_boundary = CoreBoundary.from_pyproject()

    assert (
        set(core_boundary.modules_loading_other_distributions)
        - set(core_boundary.modules)
        == set()
    )


# %% static boundary


def test_core_imports_no_module_of_all() -> None:
    """
    No core module imports a module of ``krrood[all]``, apart from the imports the
    contract lists as today's offenders; an offender that no longer imports is an error.
    """
    exit_status = lint_imports(
        config_filename=str(DeclarationFile.PYPROJECT),
        limit_to_contracts=(ImportContractIdentifier.CORE_IMPORTS_NO_ALL_MODULE,),
        no_cache=True,
        no_logo=True,
    )

    assert exit_status == EXIT_STATUS_SUCCESS

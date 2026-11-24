"""
The krrood distributions of the checkout and the modules each may load: those of the
distributions it declares, with everything they require in turn.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import tomllib
from collections import deque
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import ModuleType

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# %% the distributions of the checkout


@dataclass(kw_only=True)
class CheckoutDistribution:
    """
    A krrood distribution whose project lives in the checkout.
    """

    project_directory: Path
    """
    The directory holding the distribution's ``pyproject.toml``.
    """

    @cached_property
    def project(self) -> dict:
        """
        The ``[project]`` table of the distribution's ``pyproject.toml``.
        """
        declaration = (self.project_directory / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        return tomllib.loads(declaration)["project"]

    @property
    def distribution_name(self) -> str:
        """
        The canonical name the distribution is installed under.
        """
        return canonicalize_name(self.project["name"])

    @property
    def requirements(self) -> list[Requirement]:
        """
        The distributions a plain install of this one installs alongside it.
        """
        return [
            Requirement(requirement) for requirement in self.project["dependencies"]
        ]

    @property
    def source_directory(self) -> Path:
        """
        The ``src`` directory holding the distribution's part of :mod:`krrood`.
        """
        return self.project_directory / "src"

    @property
    def subpackage_names(self) -> list[str]:
        """
        The dotted names of the subpackages and modules directly below :mod:`krrood`
        that this distribution ships.
        """
        return [
            f"krrood.{path.stem}"
            for path in sorted((self.source_directory / "krrood").iterdir())
            if not path.name.startswith("_")
            and (path.suffix == ".py" or (path / "__init__.py").exists())
        ]

    @property
    def module_names(self) -> list[str]:
        """
        The dotted name of every module the distribution ships.
        """
        return self.module_listing_script().module_names(
            source_directory=self.source_directory
        )

    @staticmethod
    def module_listing_script() -> ModuleType:
        """
        The script the install-alone workflow imports every module of a distribution
        with, loaded from the checkout.
        """
        path = Path(__file__).parents[3] / "scripts" / "import_distribution_modules.py"
        specification = importlib.util.spec_from_file_location(path.stem, path)
        script = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(script)
        return script


def checkout_distributions() -> list[CheckoutDistribution]:
    """
    :return: Every krrood-* distribution of the checkout.
    """
    krrood_directory = Path(__file__).parents[3] / "krrood"
    return [
        CheckoutDistribution(project_directory=directory)
        for directory in sorted(krrood_directory.glob("krrood-*"))
        if (directory / "pyproject.toml").exists()
    ]


# %% the modules a distribution may not load


@dataclass(kw_only=True)
class UndeclaredModules:
    """
    The installed modules a distribution may not load, because neither it nor anything
    it requires declares the distribution that provides them.
    """

    distribution: CheckoutDistribution
    """
    The distribution whose declared dependencies decide what it may load.
    """

    checkout: list[CheckoutDistribution]
    """
    Every krrood distribution of the checkout, whose subpackages share :mod:`krrood`.
    """

    @cached_property
    def declared_distribution_names(self) -> set[str]:
        """
        The canonical names of the distribution and of every installed distribution a
        plain install of it brings along.
        """
        checkout_by_name = {
            member.distribution_name: member for member in self.checkout
        }
        installed_names = {
            canonicalize_name(installed.metadata["Name"])
            for installed in importlib.metadata.distributions()
        }
        declared = {self.distribution.distribution_name}
        pending = deque(self.distribution.requirements)
        while pending:
            requirement = pending.popleft()
            name = canonicalize_name(requirement.name)
            if name in declared:
                continue
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            declared.add(name)
            if name in checkout_by_name:
                pending.extend(checkout_by_name[name].requirements)
            elif name in installed_names:
                pending.extend(
                    Requirement(requirement)
                    for requirement in importlib.metadata.requires(name) or []
                )
        return declared

    @property
    def module_names(self) -> list[str]:
        """
        The top-level modules of undeclared installed distributions, and the
        :mod:`krrood` subpackages of undeclared krrood distributions.
        """
        top_level_modules = [
            module
            for module, providers in importlib.metadata.packages_distributions().items()
            if module != "krrood"
            and module not in sys.stdlib_module_names
            and not {canonicalize_name(provider) for provider in providers}
            & self.declared_distribution_names
        ]
        krrood_subpackages = [
            subpackage
            for member in self.checkout
            if member.distribution_name not in self.declared_distribution_names
            for subpackage in member.subpackage_names
        ]
        return sorted(top_level_modules) + krrood_subpackages

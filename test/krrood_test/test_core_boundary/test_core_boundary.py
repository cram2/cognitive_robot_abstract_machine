"""
The runtime backstop of krrood's core boundary: loading all of core in a fresh
interpreter loads no distribution the static import graph does not predict.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from packaging.utils import canonicalize_name

import krrood
from krrood.development_tools.agents.core_boundary import (
    CoreBoundary,
    DeclarationFile,
)

# %% loading core


@dataclass(frozen=True)
class ImportedPackage:
    """
    A top-level package that an interpreter loaded, with the package that loaded it.
    """

    package: str
    """
    The top-level package that was loaded.
    """

    importer: str | None
    """
    The top-level package whose import statement first loaded it, or ``None`` when the
    interpreter's own source imported it.
    """


def packages_imported_while_running(source: str) -> list[ImportedPackage]:
    """
    :param source: The Python source a fresh, isolated interpreter runs.
    :return: The top-level packages the interpreter loads while starting and running it,
        as ``python -X importtime`` reports them. A module is reported before the
        module that imports it, one indentation level shallower.
    """
    import_report = subprocess.run(
        [sys.executable, "-I", "-X", "importtime", "-c", source],
        capture_output=True,
        text=True,
        check=True,
    )
    reported = [
        line.rpartition("|")[2].rstrip()
        for line in import_report.stderr.splitlines()
        if line.startswith("import time:")
    ]
    imported_packages = []
    for position, module in enumerate(reported):
        indentation = len(module) - len(module.lstrip())
        importer = next(
            (
                later.strip()
                for later in reported[position + 1 :]
                if len(later) - len(later.lstrip()) < indentation
            ),
            None,
        )
        imported_packages.append(
            ImportedPackage(
                module.strip().partition(".")[0],
                importer and importer.partition(".")[0],
            )
        )
    return imported_packages


def test_loading_core_loads_only_the_distributions_its_import_graph_predicts() -> None:
    """
    Catches what the static graph cannot see, such as a dynamic import in a core module.

    A package that a library loads on its own account, because it happens to be
    installed, is not core's to predict.
    """
    boundary = CoreBoundary.from_pyproject(
        Path(krrood.__file__).parents[2] / DeclarationFile.PYPROJECT
    )
    source = "\n".join(f"import {module}" for module in boundary.core_modules)
    startup_packages = {
        imported.package for imported in packages_imported_while_running("")
    }
    loaded_by_core = {
        imported.package
        for imported in packages_imported_while_running(source)
        if imported.importer in (None, krrood.__name__)
    } - startup_packages
    distributions_by_package = importlib.metadata.packages_distributions()
    loaded_distributions = {
        canonicalize_name(distribution)
        for package in loaded_by_core
        for distribution in distributions_by_package.get(package, [])
    }
    predicted = boundary.requirement_closure(
        set().union(
            *map(boundary.distributions_loaded_beyond_core, boundary.core_modules)
        )
    )

    assert loaded_distributions - boundary.allowed_distributions - predicted == set()


# %% attributing a loaded package to its importer


@pytest.fixture
def mimic_dataset_directory() -> Path:
    """
    :return: The directory of the mimic modules, which a fresh interpreter can import.
    """
    return Path(__file__).parent / "dataset"


def imports_of_mimic_modules(
    dataset_directory: Path, *modules: str
) -> set[ImportedPackage]:
    """
    :param dataset_directory: The directory the interpreter imports the modules from.
    :param modules: The mimic modules the interpreter imports.
    :return: The packages that loading the modules loads, each with its importer.
    """
    source = "\n".join(
        [f"import sys; sys.path.insert(0, {str(dataset_directory)!r})"]
        + [f"import {module}" for module in modules]
    )
    return set(packages_imported_while_running(source))


def test_a_package_loaded_by_a_library_is_attributed_to_the_library(
    mimic_dataset_directory: Path,
) -> None:
    """
    A package a library loads on its own account is attributed to the library, not to
    the core module that imported the library.
    """
    imports = imports_of_mimic_modules(
        mimic_dataset_directory, "core_with_library_import"
    )

    assert ImportedPackage("optional_extra", "library_with_optional_probe") in imports


def test_a_package_loaded_by_name_from_a_core_module_is_attributed_to_the_module(
    mimic_dataset_directory: Path,
) -> None:
    """
    A package a core module loads by name is attributed to that module, so the backstop
    can tell it from a library's optional import.
    """
    imports = imports_of_mimic_modules(
        mimic_dataset_directory, "core_with_dynamic_import"
    )

    assert (
        ImportedPackage("dynamically_imported", "core_with_dynamic_import") in imports
    )


def test_a_package_imported_by_the_interpreter_source_has_no_importer(
    mimic_dataset_directory: Path,
) -> None:
    """
    A package the interpreter's own source imports has no importer.
    """
    imports = imports_of_mimic_modules(
        mimic_dataset_directory, "core_with_library_import"
    )

    assert ImportedPackage("core_with_library_import", None) in imports

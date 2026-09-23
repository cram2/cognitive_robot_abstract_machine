"""
The package's dependency declaration, and which of it this environment is missing.

The declaration is read from ``pyproject.toml``, so these tests read it there too rather
than restating the four distributions - what they hold is the reading, not the list.
"""

from __future__ import annotations

import tomllib

import pytest

from .script_runner import PythonModuleRunner
from basstler import dependencies
from basstler.dependencies import Dependency
from basstler.package_layout import PACKAGE_DIRECTORY, REPOSITORY_ROOT

# %% the declaration


def declared_specifiers() -> list[str]:
    """
    :return: The requirement specifiers ``pyproject.toml`` states, read straight from it.
    """
    project = tomllib.loads((PACKAGE_DIRECTORY / "pyproject.toml").read_text())
    return project["project"]["dependencies"]


def test_the_declared_dependencies_are_the_ones_pyproject_states():
    """
    One declaration: what the package installs is what its metadata says it needs.
    """
    assert [
        dependency.specifier for dependency in dependencies.declared_dependencies()
    ] == declared_specifiers()


def test_the_declaration_is_static_so_a_reader_needs_no_build():
    """
    ``dependencies`` resolved by the build backend is invisible to anything that reads
    the file - which is what this module, the session start and the workflows all do.
    """
    project = tomllib.loads((PACKAGE_DIRECTORY / "pyproject.toml").read_text())

    assert "dependencies" not in project["project"].get("dynamic", [])


@pytest.mark.parametrize(
    ("specifier", "distribution_name"),
    [
        ("nh3>=0.2", "nh3"),
        ("pyyaml >= 6", "pyyaml"),
        ("markdown", "markdown"),
        ("jinja2[i18n]>=3", "jinja2"),
        ("nh3 ; python_version >= '3.11'", "nh3"),
    ],
)
def test_the_distribution_name_is_the_specifier_without_its_constraints(
    specifier: str, distribution_name: str
):
    """
    A specifier carries version bounds, extras and markers; only its name identifies the
    distribution to look up.
    """
    assert Dependency(specifier).distribution_name == distribution_name


# %% what is missing


def test_a_dependency_this_environment_has_is_not_reported_missing():
    """
    The common case the session start turns on: nothing missing means no installer runs.
    """
    installed = Dependency("pytest")

    assert not installed.is_missing


def test_a_distribution_nothing_has_installed_is_reported_missing():
    """
    A name no distribution can answer to, so the check cannot pass by accident.
    """
    absent = Dependency("basstler-no-such-distribution>=1")

    assert absent.is_missing


def test_missing_dependencies_are_a_subset_of_what_is_declared():
    """
    What the caller installs is drawn from the declaration and nothing else.
    """
    missing = dependencies.missing_dependencies()

    assert {dependency.specifier for dependency in missing} <= set(
        declared_specifiers()
    )


# %% the command line the shell calls


def run_module(*arguments: str):
    """
    Run this module as the shell does, from the repository root.

    :param arguments: Arguments to the module.
    :return: The completed process, with output captured as text.
    """
    return PythonModuleRunner(
        project_root=REPOSITORY_ROOT, module_name=dependencies.__name__
    ).run(*arguments)


def test_the_command_line_prints_one_specifier_per_missing_dependency():
    """
    The output is what a caller passes straight to ``pip install``, so it carries the
    version constraints rather than bare names.
    """
    result = run_module()

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        dependency.specifier for dependency in dependencies.missing_dependencies()
    ]


def test_the_command_line_answers_help():
    """
    Every entry point of this package answers ``--help``; this one is called by a hook
    that has to be able to tell a broken invocation from a satisfied one.
    """
    result = run_module("--help")

    assert result.returncode == 0, result.stderr


def test_an_unreadable_declaration_fails_rather_than_reporting_nothing_missing():
    """
    A missing or unparseable ``pyproject.toml`` must not look like a satisfied
    environment: the caller reports that it could not check instead of installing
    nothing.
    """
    result = PythonModuleRunner(
        project_root=REPOSITORY_ROOT, module_name=dependencies.__name__
    ).run("--declaration", "/nonexistent/pyproject.toml")

    assert result.returncode != 0
    assert result.stdout.strip() == ""

"""
Every package published to PyPI links the repository's changelog in its metadata.

A package is published when a ``<package>-publish-to-pypi.yml`` workflow exists for it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

# %% published packages

REPOSITORY_ROOT = Path(__file__).parents[2]
"""
The root of the monorepo.
"""

PUBLISHED_PACKAGES = sorted(
    workflow.name.removesuffix("-publish-to-pypi.yml").replace("-", "_")
    for workflow in (REPOSITORY_ROOT / ".github" / "workflows").glob(
        "*-publish-to-pypi.yml"
    )
)
"""
The directory names of the packages that have a publish workflow, which spells
them with hyphens where the directory uses underscores.
"""


def directory_of(package: str) -> Path:
    """
    :param package: The directory name of a package.
    :return: The directory of the workspace member with that name.
    """
    root = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    members = root["tool"]["uv"]["workspace"]["members"]
    return REPOSITORY_ROOT / next(
        member for member in members if Path(member).name == package
    )


def changelog_url_of(package: str) -> str | None:
    """
    :param package: The directory name of a package.
    :return: The ``Changelog`` entry of its ``[project.urls]``, or ``None`` without one.
    """
    pyproject = tomllib.loads(
        (directory_of(package) / "pyproject.toml").read_text()
    )
    return pyproject["project"].get("urls", {}).get("Changelog")


# %% changelog links


@pytest.mark.parametrize("package", PUBLISHED_PACKAGES)
def test_published_package_links_the_changelog_file(package: str) -> None:
    url = changelog_url_of(package)

    assert url is not None
    assert url.endswith("/CHANGELOG.md")


def test_published_packages_link_the_same_changelog() -> None:
    urls = {changelog_url_of(package) for package in PUBLISHED_PACKAGES}

    assert len(urls) == 1


def test_the_changelog_the_packages_link_exists() -> None:
    assert (REPOSITORY_ROOT / "CHANGELOG.md").is_file()

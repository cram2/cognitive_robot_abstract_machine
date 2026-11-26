"""
Tests that every krrood distribution carries the metadata its PyPI project shows: a
description page, its license file and a build backend that understands its license.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from .dataset.declared_dependencies import (
    CheckoutDistribution,
    checkout_distributions,
)


def released_distributions() -> list[CheckoutDistribution]:
    """
    :return: The ``krrood`` distribution and every krrood-* distribution it depends on.
    """
    krrood_directory = Path(__file__).parents[2] / "krrood"
    return [
        CheckoutDistribution(project_directory=krrood_directory),
        *checkout_distributions(),
    ]


# %% description page


@pytest.mark.parametrize(
    "distribution",
    released_distributions(),
    ids=lambda distribution: distribution.distribution_name,
)
def test_distribution_declares_an_existing_readme(
    distribution: CheckoutDistribution,
) -> None:
    """
    The distribution declares a readme that exists, so its PyPI project has a
    description page.
    """
    readme = distribution.project_directory / distribution.project["readme"]

    assert readme.is_file()


# %% license


@pytest.mark.parametrize(
    "distribution",
    released_distributions(),
    ids=lambda distribution: distribution.distribution_name,
)
def test_distribution_ships_its_license_file(
    distribution: CheckoutDistribution,
) -> None:
    """
    The distribution's directory holds the license file, which setuptools includes in
    the wheel and the source distribution by default.
    """
    assert (distribution.project_directory / "LICENSE").is_file()


@pytest.mark.parametrize(
    "distribution",
    released_distributions(),
    ids=lambda distribution: distribution.distribution_name,
)
def test_build_backend_reads_a_license_expression(
    distribution: CheckoutDistribution,
) -> None:
    """
    The distribution states its license as an SPDX expression, which setuptools reads
    only from version 77 on, so its build requirement admits no older setuptools.
    """
    declaration = tomllib.loads(
        (distribution.project_directory / "pyproject.toml").read_text(encoding="utf-8")
    )
    setuptools = next(
        Requirement(requirement)
        for requirement in declaration["build-system"]["requires"]
        if Requirement(requirement).name == "setuptools"
    )

    assert isinstance(distribution.project["license"], str)
    assert not setuptools.specifier.contains(Version("76.1.0"))

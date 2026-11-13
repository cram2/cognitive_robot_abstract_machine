"""
Reject unsupported dependency installation selections.
"""

from __future__ import annotations

# %% Imports

from typing_extensions import TYPE_CHECKING

import pytest

from scripts.cramployer.installation import Installation
from scripts.cramployer.exceptions import (
    InvalidDependencySource,
    UnknownWorkspacePackage,
)
from scripts.cramployer.vocabulary import OptionalPackage, DependencySource

if TYPE_CHECKING:
    from pathlib import Path


# %% Dependency selection
@pytest.mark.parametrize("selection", ["missing:feature", OptionalPackage.ROBOKUDO])
def test_workspace_extra_requires_a_workspace_package(
    repository: Path, selection: str
) -> None:
    """
    Reject unknown packages and selections without an extra separator.
    """
    with pytest.raises(UnknownWorkspacePackage):
        Installation(
            repository, DependencySource.PROCESSOR_ONLY, workspace_extras=(selection,)
        ).build_commands()


def test_unknown_dependency_source_is_rejected(repository: Path) -> None:
    """
    Reject dependency sources outside the supported modes.
    """
    with pytest.raises(InvalidDependencySource):
        Installation(repository, "unsupported").build_commands()

"""
Optional dependency selections parsed from workspace project metadata.
"""

from __future__ import annotations

# %% Imports
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from string import Template

from typing_extensions import TYPE_CHECKING

from scripts.cramployer.exceptions import UnknownWorkspaceExtra, UnknownWorkspacePackage
from scripts.cramployer.vocabulary import CommandArgument, FileName, PayloadKey, Program

if TYPE_CHECKING:
    from typing_extensions import Any, Self


# %% Workspace metadata
@dataclass
class Workspace:
    """
    Package directories declared by the workspace manager.
    """

    members: list[str]
    """Directories containing workspace members."""

    @classmethod
    def from_json(class_type, payload: dict[str, Any]) -> Self:
        """
        Parse workspace members from root project metadata.
        """
        return class_type(
            payload[PayloadKey.TOOL][PayloadKey.UV][PayloadKey.WORKSPACE][
                PayloadKey.MEMBERS
            ]
        )


@dataclass
class Package:
    """
    Optional dependencies declared by a workspace member.
    """

    extras: dict[str, list[str]]
    """Declared optional dependency groups."""

    @classmethod
    def from_json(class_type, payload: dict[str, Any]) -> Self:
        """
        Parse a member's optional dependencies from project metadata.
        """
        return class_type(
            payload[PayloadKey.PROJECT].get(PayloadKey.OPTIONAL_DEPENDENCIES, {})
        )


# %% Optional dependency selection
@dataclass
class WorkspaceFeature:
    """
    A validated optional dependency group of a workspace member.
    """

    package: str
    """Workspace member providing the dependency group."""
    extra: str
    """
    Declared optional dependency group.
    """

    @classmethod
    def from_selection(
        class_type, selection: str, repository: Path, workspace: Workspace
    ) -> Self:
        """
        Validate a package:extra selection against checkout metadata.
        """
        package, separator, extra = selection.partition(":")
        if not separator or package not in workspace.members:
            raise UnknownWorkspacePackage(selection)
        metadata = Package.from_json(
            tomllib.loads((repository / package / FileName.PROJECT).read_text())
        )
        if extra not in metadata.extras:
            raise UnknownWorkspaceExtra(selection)
        return class_type(package, extra)

    def build_commands(self) -> str:
        """
        Build the command that installs the extra while preserving installed packages.
        """
        command = shlex.join(
            [
                Program.UV,
                CommandArgument.SYNC,
                CommandArgument.ACTIVE,
                CommandArgument.INEXACT,
                CommandArgument.PACKAGE,
                self.package,
                CommandArgument.EXTRA,
                self.extra,
            ]
        )
        template = Template(
            (Path(__file__).parent / FileName.EXTRA_TEMPLATE).read_text()
        )
        return template.substitute(
            checkout=shlex.quote(FileName.CHECKOUT), command=command
        ).rstrip()

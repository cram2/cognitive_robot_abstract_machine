"""
The names of the ``krrood`` subcommands.
"""

from __future__ import annotations

from enum import StrEnum

from krrood.ormatic.utils import classproperty


class Subcommand(StrEnum):
    """
    The subcommands of ``krrood``.
    """

    FORMAT = "format"
    """
    Format Python files with ruff and docformatter.
    """

    AGENTS_MD = "agents-md"
    """
    Generate the repository's ``AGENTS.md``.
    """

    SKILLS = "skills"
    """
    Generate the ``SKILL.md`` of every skill the repository declares.
    """

    CHECK = "check"
    """
    Report the code that breaks a rule a checker enforces.
    """

    @classproperty
    def program_name(cls) -> str:
        """
        The name the ``krrood`` command is invoked by.
        """
        return "krrood"

    @property
    def invocation(self) -> str:
        """
        The command line running this subcommand.
        """
        return f"{Subcommand.program_name} {self}"

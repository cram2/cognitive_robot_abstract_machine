"""
Exceptions raised by the development tools.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass
class FormatterNotInstalled(DataclassException):
    """
    Raised when a formatting tool the formatter runs is not installed for the running
    interpreter.
    """

    tool_module: str
    """
    The module the missing tool runs under.
    """

    def error_message(self) -> str:
        return f"The formatting tool `{self.tool_module}` is not installed."

    def suggest_correction(self) -> str:
        return 'Install the formatting tools with `pip install "krrood[development-tools]"`.'


@dataclass
class GitCommandFailed(DataclassException):
    """
    Raised when a git command the development tools depend on exits with an error.
    """

    arguments: tuple[str, ...]
    """
    The git subcommand and its arguments, as invoked.
    """

    standard_error: str
    """
    What git printed to its error stream.
    """

    def error_message(self) -> str:
        return f"`git {' '.join(self.arguments)}` failed: {self.standard_error.strip()}"

    def suggest_correction(self) -> str:
        return "Run the command from inside a git repository."

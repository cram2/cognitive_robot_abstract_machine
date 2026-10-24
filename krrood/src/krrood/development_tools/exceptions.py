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
class SectionWithoutParent(DataclassException):
    """
    Raised when a section of the rules names no section it is nested under.
    """

    section: type
    """
    The section without a parent.
    """

    def error_message(self) -> str:
        return f"The section {self.section.__qualname__} names no section it is nested under."

    def suggest_correction(self) -> str:
        return (
            "Give the section a `parent` classproperty returning the section or rule set "
            "it is nested under."
        )

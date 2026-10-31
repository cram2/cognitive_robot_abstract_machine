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

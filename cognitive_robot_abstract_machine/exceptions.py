"""
Errors raised while working with the repository checkout itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException
from typing_extensions import Tuple


@dataclass
class MissingOrmGeneratorError(DataclassException, FileNotFoundError):
    """
    Raised when the script that generates a package's ORM interface is not there.
    """

    package_name: str
    """
    Name of the package whose generator is missing.
    """

    path: Path
    """
    Where the generator was looked for.
    """

    def error_message(self) -> str:
        return f"{self.package_name} has no ORM interface generator at {self.path}."

    def suggest_correction(self) -> str:
        return (
            "Check that this is a complete checkout of the repository and that the "
            "package still generates its ORM interface."
        )


@dataclass
class OrmGenerationFailedError(DataclassException, RuntimeError):
    """
    Raised when a package's ORM interface generator exits without having built it.
    """

    package_name: str
    """
    Name of the package whose generator failed.
    """

    output: str
    """
    What the generator wrote before it gave up, empty when it wrote straight to the
    terminal rather than into this report.
    """

    def error_message(self) -> str:
        report = f"Generating the ORM interface of {self.package_name} failed."
        if not self.output:
            return report
        return f"{report} It wrote:\n{self.output}"

    def suggest_correction(self) -> str:
        return (
            "Run the generation again with --debug to follow what the generator does."
        )


@dataclass
class MissingOrmBuildChoiceError(DataclassException, ValueError):
    """
    Raised when a test run names the option that says when to build the ORM interfaces
    without saying which choice it means.
    """

    option: str
    """
    The option that was left without a choice.
    """

    choices: Tuple[str, ...]
    """
    The choices it accepts.
    """

    def error_message(self) -> str:
        return f"{self.option} says nothing about when to build the ORM interfaces."

    def suggest_correction(self) -> str:
        return f"Follow it with one of {', '.join(self.choices)}."


@dataclass
class UnknownOrmBuildChoiceError(DataclassException, ValueError):
    """
    Raised when a test run says to build the ORM interfaces at a time that does not
    exist.
    """

    source: str
    """
    Where the choice was read, such as the option or the environment variable stating
    it.
    """

    choice: str
    """
    What it said, which names no time to build at.
    """

    choices: Tuple[str, ...]
    """
    The choices it accepts.
    """

    def error_message(self) -> str:
        return (
            f"{self.source} says to build the ORM interfaces '{self.choice}', which is "
            f"no time to build at."
        )

    def suggest_correction(self) -> str:
        return f"State one of {', '.join(self.choices)}."

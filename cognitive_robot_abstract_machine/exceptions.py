"""
Errors raised while working with the repository checkout itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException


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

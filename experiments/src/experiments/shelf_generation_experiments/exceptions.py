from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from krrood.exceptions import DataclassException


@dataclass
class PathError(DataclassException):
    """
    Raised when a given path is either None or invalid.
    """

    path: Optional[Path]

    def error_message(self) -> str:
        return f"Path {self.path} is invalid."

    def suggest_correction(self) -> str:
        return "Please provide a valid path."

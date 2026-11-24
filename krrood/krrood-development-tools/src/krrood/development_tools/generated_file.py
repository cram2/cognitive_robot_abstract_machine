"""
Files whose content a ``krrood`` subcommand generates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

# %% generated files


class GeneratedFile(ABC):
    """
    A file whose whole content is generated, which only its generator writes.
    """

    @property
    @abstractmethod
    def path(self) -> Path:
        """
        Where the file lies.
        """

    @property
    @abstractmethod
    def text(self) -> str:
        """
        The content the file must have.
        """

    def is_current(self) -> bool:
        """
        :return: Whether the file exists and equals :attr:`text`.
        """
        if not self.path.is_file():
            return False
        return self.path.read_text(encoding="utf-8") == self.text

    def write(self) -> bool:
        """
        Write the file, and the directories it lies in, replacing whatever it held.

        :return: Whether the file changed.
        """
        if self.is_current():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.text, encoding="utf-8")
        return True

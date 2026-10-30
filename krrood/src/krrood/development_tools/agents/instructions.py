"""
Generates a repository's ``AGENTS.md`` from the rules krrood ships and the rules the
repository declares.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.code_generation.generator import CodeGenerator
from krrood.development_tools.agents import shipped_rules
from krrood.development_tools.agents.repository_files import (
    RepositoryInstructionsFile,
)
from krrood.development_tools.agents.rule import Section
from krrood.development_tools.agents.rule_module import RuleModule
from krrood.patterns.path_enumeration import PathEnumeration

# %% file contents


class TemplateFile(PathEnumeration):
    """
    The templates the generated files are rendered from.
    """

    AGENTS = Path(__file__).parent / "templates" / "agents.md.jinja"
    """
    The content of ``AGENTS.md``: rules under the headings of their sections.
    """


# %% generation


@dataclass
class AgentsInstructions:
    """
    The ``AGENTS.md`` of one repository, listing the rules of the given sections.
    """

    repository_root: Path
    """
    The directory holding the repository's ``AGENTS.md``.
    """

    sections: list[type[Section]]
    """
    The top-level sections the file lists, each with the sections nested under it.
    """

    @classmethod
    def from_repository(cls, repository_root: Path) -> AgentsInstructions:
        """
        :param repository_root: The directory holding the repository's ``AGENTS.md``.
        :return: The file listing the sections krrood ships, then the repository's own.
        """
        sections = RuleModule(shipped_rules).sections
        project_rules = RuleModule.import_from(
            repository_root, RepositoryInstructionsFile.PROJECT_RULES
        )
        if project_rules is not None:
            sections += project_rules.sections
        return cls(repository_root, sections)

    @property
    def path(self) -> Path:
        """
        The path of the repository's ``AGENTS.md``.
        """
        return self.repository_root / RepositoryInstructionsFile.AGENTS

    @property
    def text(self) -> str:
        """
        The content ``AGENTS.md`` must have: the rules of :attr:`sections`, under their
        headings, in the order they are declared.
        """
        return CodeGenerator(str(TemplateFile.AGENTS.parent)).render(
            TemplateFile.AGENTS.name,
            sections=self.sections,
        )

    def is_current(self) -> bool:
        """
        :return: Whether ``AGENTS.md`` exists and equals :attr:`text`.
        """
        if not self.path.is_file():
            return False
        return self.path.read_text(encoding="utf-8") == self.text

    def write(self) -> bool:
        """
        Write ``AGENTS.md``, replacing whatever it held.

        :return: Whether the file changed.
        """
        if self.is_current():
            return False
        self.path.write_text(self.text, encoding="utf-8")
        return True

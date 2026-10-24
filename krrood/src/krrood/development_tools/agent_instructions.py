"""
Generates a repository's ``AGENTS.md`` from the rules krrood ships and the rules the
repository declares.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jinja2

from krrood.development_tools.agent_rule import ProjectRules, RuleSet
from krrood.development_tools.repository_files import RepositoryInstructionsFile

# imported to declare the sections of the shipped rule sets
from krrood.development_tools.rules import krrood_rules, python_rules  # noqa: F401
from krrood.patterns.path_enumeration import PathEnumeration

# %% file contents


class TemplateFile(PathEnumeration):
    """
    The templates the generated files are rendered from.
    """

    AGENTS = Path(__file__).parent / "templates" / "agents.md.jinja"
    """
    The content of ``AGENTS.md``: the rules of the chosen rule sets, under the headings
    of their sections.
    """


# %% generation


@dataclass
class AgentsInstructions:
    """
    The ``AGENTS.md`` of one repository, generated from the chosen rule sets.
    """

    repository_root: Path
    """
    The directory holding the repository's ``AGENTS.md``.
    """

    rule_sets: tuple[type[RuleSet], ...]
    """
    The rule sets the file lists, in this order.
    """

    def __post_init__(self) -> None:
        """
        Import the repository's own rules, so that :class:`ProjectRules` lists its
        sections.
        """
        ProjectRules.import_from(self.repository_root)

    @property
    def path(self) -> Path:
        """
        The path of the repository's ``AGENTS.md``.
        """
        return self.repository_root / RepositoryInstructionsFile.AGENTS

    @property
    def text(self) -> str:
        """
        The content ``AGENTS.md`` must have: the rules of the chosen rule sets, under the
        headings of the sections that list any.
        """
        environment = jinja2.Environment(
            loader=jinja2.FileSystemLoader(TemplateFile.AGENTS.parent),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        return environment.get_template(TemplateFile.AGENTS.name).render(
            rule_sets=self.rule_sets
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

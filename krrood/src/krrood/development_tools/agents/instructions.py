"""
Generates a repository's ``AGENTS.md`` from the rules krrood ships and the rules the
repository declares, and the ``SKILL.md`` of each skill the repository declares.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from krrood.code_generation.generator import CodeGenerator
from krrood.development_tools.agents import shipped_rules
from krrood.development_tools.agents.repository_files import (
    RepositoryInstructionsFile,
)
from krrood.development_tools.agents.checker import CodeRuleChecker
from krrood.development_tools.agents.rule import Section
from krrood.development_tools.agents.rule_module import RuleModule
from krrood.development_tools.agents.skill import Skill
from krrood.development_tools.subcommand import Subcommand
from krrood.patterns.path_enumeration import PathEnumeration

# %% file contents


class TemplateFile(PathEnumeration):
    """
    The templates the generated files are rendered from.
    """

    SECTIONS = Path(__file__).parent / "templates" / "sections.md.jinja"
    """
    The content of ``AGENTS.md``: rules under the headings of their sections.
    """

    SKILL = Path(__file__).parent / "templates" / "skill.md.jinja"
    """
    The content of a ``SKILL.md``: :attr:`SECTIONS` with the skill's front matter and
    introduction in place of the ``AGENTS.md`` notice, and every heading one level lower.
    """


# %% generated files


class GeneratedInstructions(ABC):
    """
    An instructions file whose whole content is generated.
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


# %% AGENTS.md


@dataclass(kw_only=True)
class AgentsInstructions(GeneratedInstructions):
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
    def from_repository(cls, *, repository_root: Path) -> AgentsInstructions:
        """
        :param repository_root: The directory holding the repository's ``AGENTS.md``.
        :return: The file listing the sections krrood ships, then the repository's own.
        """
        sections = RuleModule(module=shipped_rules).sections
        project_rules = RuleModule.import_from(
            directory=repository_root,
            module_name=RepositoryInstructionsFile.PROJECT_RULES,
        )
        if project_rules is not None:
            sections += project_rules.sections
        return cls(repository_root=repository_root, sections=sections)

    @property
    def checkers(self) -> tuple[CodeRuleChecker, ...]:
        """
        The checker of every rule in :attr:`sections` and the sections nested in them.
        """
        return tuple(
            checker for section in self.sections for checker in section.checkers
        )

    @property
    def path(self) -> Path:
        return self.repository_root / RepositoryInstructionsFile.AGENTS

    @property
    def text(self) -> str:
        return CodeGenerator(
            template_directory=str(TemplateFile.SECTIONS.parent)
        ).render(
            template_name=TemplateFile.SECTIONS.name,
            sections=self.sections,
            check_command=Subcommand.CHECK.invocation,
        )


# %% skills


@dataclass(kw_only=True)
class SkillInstructions(GeneratedInstructions):
    """
    The ``SKILL.md`` of one skill of a repository.
    """

    repository_root: Path
    """
    The directory the skill's :attr:`Skill.path` is relative to.
    """

    skill: type[Skill]
    """
    The skill the file instructs.
    """

    @property
    def path(self) -> Path:
        return self.repository_root / self.skill.path

    @property
    def text(self) -> str:
        return CodeGenerator(template_directory=str(TemplateFile.SKILL.parent)).render(
            template_name=TemplateFile.SKILL.name,
            skill=self.skill,
            sections=self.skill.sections,
            generate_command=Subcommand.SKILLS.invocation,
            check_command=Subcommand.CHECK.invocation,
        )


@dataclass(kw_only=True)
class RepositorySkills:
    """
    The skills a repository declares in its :attr:`RepositoryInstructionsFile.PROJECT_SKILLS`
    module.
    """

    repository_root: Path
    """
    The directory holding the module, which the skills' paths are relative to.
    """

    @property
    def instructions(self) -> list[SkillInstructions]:
        """
        The ``SKILL.md`` of every declared skill; none if the repository has no such
        module.
        """
        project_skills = RuleModule.import_from(
            directory=self.repository_root,
            module_name=RepositoryInstructionsFile.PROJECT_SKILLS,
        )
        if project_skills is None:
            return []
        return [
            SkillInstructions(repository_root=self.repository_root, skill=skill)
            for skill in project_skills.skills
        ]

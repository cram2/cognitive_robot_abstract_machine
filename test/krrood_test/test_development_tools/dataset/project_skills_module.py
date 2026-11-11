"""
A repository's own skills module, standing in for the one a repository declares.
"""

from __future__ import annotations

from pathlib import Path

from krrood.development_tools.agents.rule import Section
from krrood.development_tools.agents.skill import AgentTool, Skill
from krrood.ormatic.utils import classproperty

# %% sections


class StepsRules(Section):
    """
    A top-level section of the skill, holding a subsection.
    """

    TAKE_ONE_STEP = "A rule about steps", ("A first detail",)


class UndoingRules(Section):
    """
    The subsection of the skill's top-level section.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return StepsRules

    UNDO_INSTEAD_OF_REBUILDING = "A rule about undoing"


class SetupRules(Section):
    """
    A second top-level section, listed after the first although it sorts before it.
    """

    START_FIRST = "A rule about setting up"


# %% skills


class DeployingTheServiceSkill(Skill):
    """
    A skill with an introduction and two sections.
    """

    @classproperty
    def path(cls) -> Path:
        return Path(".claude/skills/deploying-the-service/SKILL.md")

    @classproperty
    def description(cls) -> str:
        return "Deploy the service. Use when the developer asks to deploy."

    @classproperty
    def allowed_tools(cls) -> list[AgentTool]:
        return [AgentTool.BASH, AgentTool.READ]

    @classproperty
    def introduction(cls) -> str:
        return "An introduction to deploying."

    @classproperty
    def sections(cls) -> list[type[Section]]:
        return [StepsRules, SetupRules]


class AuditingSkill(Skill):
    """
    A second skill, listed before the first.
    """

    @classproperty
    def path(cls) -> Path:
        return Path(".claude/skills/auditing/SKILL.md")

    @classproperty
    def description(cls) -> str:
        return "Audit the service."

    @classproperty
    def allowed_tools(cls) -> list[AgentTool]:
        return [AgentTool.READ]

    @classproperty
    def introduction(cls) -> str:
        return "An introduction to auditing."

    @classproperty
    def sections(cls) -> list[type[Section]]:
        return [SetupRules]

"""
Skills: instructions a coding agent loads for one kind of task, each written into a
``SKILL.md`` of its own.
"""

from __future__ import annotations

from abc import abstractmethod
from enum import StrEnum

from krrood.development_tools.agents.rule import ClassNamedDeclaration, Section
from krrood.ormatic.utils import classproperty

# %% tools


class AgentTool(StrEnum):
    """
    The tools of Claude Code a skill may allow, by the names its ``SKILL.md`` lists.
    """

    BASH = "Bash"
    """
    Running shell commands.
    """

    READ = "Read"
    """
    Reading files.
    """

    WRITE = "Write"
    """
    Writing whole files.
    """

    EDIT = "Edit"
    """
    Replacing text in files.
    """

    GREP = "Grep"
    """
    Searching file contents.
    """

    GLOB = "Glob"
    """
    Finding files by name.
    """


# %% skills


class Skill(ClassNamedDeclaration):
    """
    A skill, declared as a class named after it followed by ``Skill``, whose sections
    are written below its introduction in the order :attr:`sections` lists them.
    """

    @classproperty
    def name_suffix(cls) -> str:
        return "Skill"

    @classproperty
    def name(cls) -> str:
        """
        The name the agent invokes the skill by: the words of its topic in lower case,
        joined by hyphens.
        """
        return "-".join(word.lower() for word in cls.topic_words)

    @classproperty
    @abstractmethod
    def description(cls) -> str:
        """
        When to use the skill, which the agent reads to decide whether to load it.
        """

    @classproperty
    @abstractmethod
    def allowed_tools(cls) -> list[AgentTool]:
        """
        The tools the agent may use without asking while it follows the skill.
        """

    @classproperty
    @abstractmethod
    def introduction(cls) -> str:
        """
        The paragraphs between the skill's title and its first section.
        """

    @classproperty
    @abstractmethod
    def sections(cls) -> list[type[Section]]:
        """
        The top-level sections of the skill, each with the sections nested under it.
        """

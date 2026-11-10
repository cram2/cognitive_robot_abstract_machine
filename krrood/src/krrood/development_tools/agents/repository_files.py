"""
The files ``krrood agents-md`` reads and writes in a repository.
"""

from __future__ import annotations

from enum import StrEnum


class RepositoryInstructionsFile(StrEnum):
    """
    The agent instruction files of a repository and the modules they are generated from,
    by their paths relative to the repository's root, or to a skill's directory for
    :attr:`SKILL`.
    """

    AGENTS = "AGENTS.md"
    """
    The instructions agents read, generated from the rules krrood ships and the
    repository's own.
    """

    SKILLS = ".claude/skills"
    """
    The directory holding a directory per skill, named after it.
    """

    SKILL = "SKILL.md"
    """
    The instructions of one skill, in the skill's directory.
    """

    PROJECT_RULES = "agent_rules"
    """
    The module or package ``krrood agents-md`` imports to find the repository's own
    rules: the sections defined in it.
    """

    PROJECT_SKILLS = "agent_skills"
    """
    The module or package ``krrood skills`` imports to find the repository's skills: the
    skills defined in it.
    """

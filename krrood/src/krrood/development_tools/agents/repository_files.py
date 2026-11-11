"""
The files ``krrood agents-md`` reads and writes in a repository.
"""

from __future__ import annotations

from enum import StrEnum


class RepositoryInstructionsFile(StrEnum):
    """
    The agent instruction files of a repository and the modules they are generated from,
    by their paths relative to the repository's root.
    """

    AGENTS = "AGENTS.md"
    """
    The instructions agents read, generated from the rules krrood ships and the
    repository's own.
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

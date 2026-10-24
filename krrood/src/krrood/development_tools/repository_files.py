"""
The files the development tools read and write in a repository.
"""

from __future__ import annotations

from enum import StrEnum


class RepositoryInstructionsFile(StrEnum):
    """
    The agent instruction files at a repository's root, by name.
    """

    AGENTS = "AGENTS.md"
    """
    The instructions agents read, generated from the rule sets.
    """

    PROJECT_RULES = "agent_rules.py"
    """
    The module ``krrood agents-md`` imports to find the repository's own rules: the
    sections nested under :class:`~krrood.development_tools.agent_rule.ProjectRules` it
    declares or imports.
    """

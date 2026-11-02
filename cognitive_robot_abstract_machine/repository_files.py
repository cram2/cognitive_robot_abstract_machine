"""
The files of this repository that code refers to by path.
"""

from __future__ import annotations

from pathlib import Path

from krrood.patterns.path_enumeration import PathEnumeration

# %% files


class RepositoryFile(PathEnumeration):
    """
    The files of this repository, by their path relative to its root.
    """

    CODE_OWNERS = Path(".github") / "CODEOWNERS"
    """
    The code owners of each part of the repository.
    """

    LABELER_CONFIGURATION = Path(".github") / "labeler.yml"
    """
    The labels set automatically on a pull request from the files it changes.
    """

    API_BREAK_CHECK_WORKFLOW = Path(".github") / "workflows" / "pr_api_break_check.yml"
    """
    The workflow that finds the public signatures a pull request breaks.
    """

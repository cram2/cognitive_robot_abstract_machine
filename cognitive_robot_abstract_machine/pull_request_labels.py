"""
The labels of pull requests on this repository.
"""

from __future__ import annotations

from enum import StrEnum

# %% labels


class ChangeKindLabel(StrEnum):
    """
    A label naming a kind of change a pull request makes, whose value is the label's
    text.
    """

    API_BREAK = "api-break"
    """
    The change can break code that already uses it.
    """

    NEW_FEATURE = "new-feature"
    """
    The change adds public API or user-visible behaviour.
    """

    BUG_FIX = "bug-fix"
    """
    The change makes existing behaviour match its contract.
    """

    DOCUMENTATION = "documentation"
    """
    The change modifies documentation files.
    """

    DEV_TOOLS = "dev-tools"
    """
    The change modifies the tools contributors work with.
    """

    MAINTENANCE = "maintenance"
    """
    The change modifies code without changing its behaviour or API.
    """

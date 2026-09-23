"""
The parts of a plan's data model that more than one tool has to agree on.

Nothing here imports anything the tools reading it do not already have: the dashboard
build imports jinja2 and markdown of its own, and the bootstrap tool imports neither.
The dependency runs one way only.
"""

from __future__ import annotations

from enum import StrEnum


class ItemStatus(StrEnum):
    """
    The statuses ``plan.yaml``'s ``status`` field accepts.

    Deliberately thin: everything about a pull request's actual GitHub state - open,
    draft, merged, its checks, its reviews - is never stored in the manifest. It is
    live-fetched and represented separately.
    """

    NOT_STARTED = "not_started"
    """
    Nothing has begun.
    """

    IN_PROGRESS = "in_progress"
    """
    The work is underway - what bootstrapping an item sets.
    """

    BLOCKED = "blocked"
    """
    Something outside the item has to move first.
    """

    DEFERRED = "deferred"
    """
    Deliberately parked rather than stuck.
    """

    DONE = "done"
    """
    Landed.
    """

    @property
    def display_label(self) -> str:
        """
        Derived from the value rather than kept in a table beside it, so a status added
        later labels itself and there is no second list to hold in step.

        :return: How this status is written on a rendered page, e.g. ``"Not started"``.
        """
        return self.value.replace("_", " ").capitalize()

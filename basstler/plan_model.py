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

    NOT_STARTED = "not started"
    """
    Nothing has begun.
    """

    IN_PROGRESS = "in progress"
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

    @classmethod
    def underscore_spellings(cls) -> dict[str, ItemStatus]:
        """
        The spelling manifests were written in before the values took spaces, which
        ``plan.yaml`` files still carry until they are rewritten.

        :return: Each status by its value with underscores in place of spaces.
        """
        return {status.value.replace(" ", "_"): status for status in cls}

    @classmethod
    def accepted_spellings(cls) -> frozenset[str]:
        """
        :return: Every text a manifest's ``status`` field may hold: each value, and its
            underscore spelling.
        """
        return frozenset(
            {status.value for status in cls} | set(cls.underscore_spellings())
        )

    @classmethod
    def _missing_(cls, value: object) -> ItemStatus | None:
        """
        Read a status from its underscore spelling as well as its value.

        :param value: The text no value matched.
        :return: The status that text spells, or ``None`` when it spells none.
        """
        return cls.underscore_spellings().get(value)

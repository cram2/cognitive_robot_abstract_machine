"""
Steps that each take a tool and an item, so a choice made for one step can be required of
the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# %% what the steps choose between


@dataclass(kw_only=True, eq=False)
class Tool:
    """
    Something a step works with, equal only to itself.
    """

    label: str
    """
    Tells the tools apart in test output.
    """

    reachable_items: list[Item] = field(default_factory=list)
    """
    The items this tool can work on.
    """


@dataclass(kw_only=True, eq=False)
class Item:
    """
    Something a step works on, equal only to itself.
    """

    label: str
    """
    Tells the items apart in test output.
    """


# %% the steps


@dataclass(kw_only=True)
class Taking:
    """
    Takes :attr:`item` with :attr:`tool`.
    """

    tool: Tool
    """
    The tool taking the item.
    """

    item: Item
    """
    The item taken.
    """


@dataclass(kw_only=True)
class Putting:
    """
    Puts :attr:`item` down with :attr:`tool`, which has to be the tool that took it.
    """

    tool: Tool
    """
    The tool holding the item.
    """

    item: Item
    """
    The item put down.
    """


@dataclass(kw_only=True)
class TakingAndPutting:
    """
    Takes an item and then puts it down.
    """

    taking: Taking
    """
    The step taking the item.
    """

    putting: Putting
    """
    The step putting the item down.
    """

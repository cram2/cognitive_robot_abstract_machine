"""
The search that shows whether one object keeps another alive through its references.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..reference_chain import ReferenceSearch


# %% linked objects


@dataclass(kw_only=True)
class Link:
    """
    An object that may hold a reference to one other object.
    """

    next_link: Link | None = None
    """
    The object this one keeps alive, if any.
    """


# %% chains


def test_a_referenced_object_is_found_through_every_object_in_between() -> None:
    target = Link()
    middle = Link(next_link=target)
    start = Link(next_link=middle)

    chain = ReferenceSearch(start=start).find_reference_chain(target=target)

    assert chain[0] is start
    assert chain[-1] is target
    assert any(link is middle for link in chain)


def test_an_unreferenced_object_is_not_found() -> None:
    start = Link(next_link=Link())

    assert ReferenceSearch(start=start).find_reference_chain(target=Link()) == []


def test_an_object_beyond_the_maximum_depth_is_not_found() -> None:
    target = Link()
    start = Link(next_link=target)
    reaching_depth = (
        len(ReferenceSearch(start=start).find_reference_chain(target=target)) - 1
    )

    shallow_search = ReferenceSearch(start=start, maximum_depth=reaching_depth - 1)

    assert shallow_search.find_reference_chain(target=target) == []

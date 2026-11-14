"""
Domain classes for rule trees whose conclusions are about some of the variables their
conditions use, and for queries that pass variables into calls or quantify one variable
twice.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from krrood.symbol_graph.symbol_graph import Symbol


# %% households


@dataclass(kw_only=True, eq=False)
class Household(Symbol):
    """
    A household whose size the conditions of a rule tree test.
    """

    size: int
    """
    The number of people living in the household.
    """


@dataclass(kw_only=True, eq=False)
class Child(Symbol):
    """
    A child a rule tree concludes a label for.
    """

    name: str
    """
    The name the child is told apart by in assertions.
    """

    age: int
    """
    The age of the child in years.
    """


@dataclass(kw_only=True, eq=False)
class ChildLabel(Symbol):
    """
    The label a rule tree concludes for a child.
    """

    child: Child
    """
    The child the label is about.
    """

    text: str
    """
    The label itself.
    """


# %% votes


@dataclass(kw_only=True, eq=False)
class Voter(Symbol):
    """
    A person who casts a vote.
    """

    name: str
    """
    The name the voter is told apart by in assertions.
    """


@dataclass(kw_only=True, eq=False)
class Vote(Symbol):
    """
    The vote one voter cast.
    """

    voter: Voter
    """
    The voter who cast the vote.
    """

    approves: bool
    """
    Whether the vote approves.
    """


@dataclass(kw_only=True, eq=False)
class Verdict(Symbol):
    """
    What a rule tree concludes about a voter from the votes.
    """

    voter: Voter
    """
    The voter the verdict is about.
    """

    approves: bool
    """
    Whether the verdict approves.
    """


@dataclass(kw_only=True, eq=False)
class Birth(Symbol):
    """
    The record of a voter's birth.
    """

    voter: Voter
    """
    The voter who was born.
    """


# %% payments


@dataclass(kw_only=True, eq=False)
class Payment(Symbol):
    """
    A payment made in one year.
    """

    year: int
    """
    The year the payment was made in.
    """

    def is_paid_in(self, *, year: int) -> bool:
        """
        :param year: A year.
        :return: Whether the payment was made in that year.
        """
        return self.year == year


@dataclass(kw_only=True, eq=False)
class TaxYear(Symbol):
    """
    A year payments are assessed in.
    """

    number: int
    """
    The calendar year.
    """


@dataclass(kw_only=True, eq=False)
class Period(Symbol):
    """
    A period whose field is named like a method every variable has.
    """

    first: datetime.date
    """
    The first day of the period.
    """

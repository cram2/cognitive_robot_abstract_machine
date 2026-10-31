"""
A method whose arguments are values that cannot be hashed, so a query calling it has to
tell its calls apart without hashing those values.
"""

from __future__ import annotations

from dataclasses import dataclass

# %% a value that cannot be hashed


@dataclass
class UnhashableLabel:
    """
    A label that compares by its text and, being a mutable dataclass, cannot be hashed.
    """

    text: str
    """
    What the label reads.
    """


# %% a holder whose method takes such values


@dataclass
class LabelHolder:
    """
    Something that carries one label and can be asked whether it carries another.
    """

    label: UnhashableLabel
    """
    The label this holder carries.
    """

    def carries(
        self, label: UnhashableLabel, also: tuple[UnhashableLabel, ...] = ()
    ) -> bool:
        """
        :param label: The label to look for.
        :param also: Further labels any of which counts as well.
        :return: Whether this holder carries *label* or one of *also*.
        """
        return self.label == label or self.label in also

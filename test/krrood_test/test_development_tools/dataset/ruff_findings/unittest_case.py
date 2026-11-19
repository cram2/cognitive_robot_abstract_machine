"""
A module testing with :class:`unittest.TestCase` and its assertion methods.
"""

from __future__ import annotations

import unittest
from unittest import TestCase  # reported


class ArithmeticCase(unittest.TestCase):  # reported
    """
    Tests written on the unittest base class.
    """

    def test_sum(self) -> None:
        """
        An equality checked with an assertion method.
        """
        self.assertEqual(1 + 1, 2)  # reported

    def test_division(self) -> None:
        """
        An exception expected with an assertion method.
        """
        with self.assertRaises(ZeroDivisionError):  # reported
            print(1 / 0)


__all__ = ["TestCase"]

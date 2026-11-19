"""
A module nesting where guard clauses and joined conditions would not.
"""

from __future__ import annotations


def sign(number: int) -> int:
    """
    An ``else`` after a branch that returns.
    """
    if number < 0:
        return -1
    else:  # reported
        return 1


def checked(number: int) -> int:
    """
    An ``else`` after a branch that raises.
    """
    if number < 0:
        raise ValueError
    else:  # reported
        return number


def both(first: bool, second: bool) -> bool:
    """
    An ``if`` whose condition could join its parent's.
    """
    if first:  # reported
        if second:
            return True
    return False


def deepest(rows: list[list[int]], limit: int, path: str) -> int:
    """
    Blocks nested six deep.
    """
    if rows:  # reported
        for row in rows:
            while limit:
                with open(path) as source:
                    for cell in row:
                        if cell > len(source.read()):
                            return cell
                limit -= 1
    return 0

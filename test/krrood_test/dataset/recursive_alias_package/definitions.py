"""
A recursive alias, which refers to itself by name.
"""

from __future__ import annotations

from typing_extensions import Union

RecursiveAliasNode = Union[int, list["RecursiveAliasNode"]]
"""
A whole number, or a list of nodes.
"""

RecursiveAliasTree = RecursiveAliasNode
"""
The name the package exports the alias under.
"""

"""
Tests that the type hints of a class resolve when they name a recursive alias of another
package.
"""

from __future__ import annotations

from typing_extensions import get_args

from krrood.class_diagrams.utils import get_type_hints_of_object

from ..dataset.recursive_alias_owner import RecursiveAliasOwner

# %% recursive aliases of another package


def test_a_recursive_alias_of_another_package_resolves():
    """
    The name a recursive alias refers to itself by is found in the module of the package
    that defines it, though the module of the annotated class never binds it.
    """
    tree_type = get_type_hints_of_object(RecursiveAliasOwner)["tree"]

    assert get_args(tree_type)[0] is int

"""
Tests that the fields of a class resolve when they are annotated with a recursive alias
of another package.
"""

from __future__ import annotations

from typing_extensions import get_args

from krrood.class_diagrams.class_diagram import ClassDiagram

from ..dataset.recursive_alias_owner import RecursiveAliasOwner

# %% recursive aliases of another package


def test_a_field_annotated_with_a_recursive_alias_of_another_package_resolves():
    """
    The name a recursive alias refers to itself by is found in the module of the package
    that defines it, though the module of the annotated class never binds it.
    """
    wrapped_class = ClassDiagram(classes=[RecursiveAliasOwner]).get_wrapped_class(
        clazz=RecursiveAliasOwner
    )
    (tree,) = [
        wrapped_field
        for wrapped_field in wrapped_class.fields
        if wrapped_field.field.name == "tree"
    ]

    assert get_args(tree.resolved_type)[0] is int

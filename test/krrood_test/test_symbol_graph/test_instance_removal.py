"""
Tests that the symbol graph removes exactly the wrapped instance it is asked to remove.
"""

from __future__ import annotations

import gc

from krrood.symbol_graph.symbol_graph import SymbolGraph

from ..dataset.value_equal_symbols import Label

# %% removing one instance


def test_removing_an_instance_keeps_an_equal_one():
    """
    Removing one of two equal instances removes that instance, not the one added first.
    """
    first, second = Label(text="a"), Label(text="a")

    SymbolGraph().remove_node(wrapped_instance=SymbolGraph().get_wrapped_instance(instance=second))

    remaining = {id(label) for label in SymbolGraph().get_instances_of_type(type_=Label)}
    assert (id(first) in remaining, id(second) in remaining) == (True, False)


# %% removing dead instances


def test_dead_instances_are_removed_and_live_ones_kept():
    """
    Removing dead instances keeps every instance that is still alive.
    """
    kept = Label(text="kept")
    for text in ("first dead", "second dead"):
        Label(text=text)
    gc.collect()

    SymbolGraph().remove_dead_instances()

    labels = list(SymbolGraph().get_instances_of_type(type_=Label))
    assert [label for label in labels if label is not None] == [kept]

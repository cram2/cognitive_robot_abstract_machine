"""
:meth:`MappedVariable.apply_mapping_on_external_root` follows a chain from a value
outside query evaluation, which is how features are read off an instance.

These tests pin what it does when a step along the way maps one value to several.
"""

import pytest

from krrood.entity_query_language.exceptions import (
    MultipleValuesAlongAccessPath,
    ReadOnlyMapping,
)
from krrood.entity_query_language.factories import flat_variable, variable

from ...dataset.semantic_world_like_classes import Cabinet

# %% following a chain of one-to-one mappings


def test_chain_of_attributes_reaches_its_value(handles_and_containers_world):
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = cabinets[0]
    chain = variable(Cabinet, domain=cabinets).container.name

    assert chain.apply_mapping_on_external_root(cabinet) == cabinet.container.name


def test_chain_through_an_index_by_a_value_reaches_its_value(
    handles_and_containers_world,
):
    """
    Indexing by a plain value reaches the one element stored under it, so it is followed
    like any other single-valued step.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) > 1)
    chain = variable(Cabinet, domain=cabinets).drawers[0].handle.name

    assert (
        chain.apply_mapping_on_external_root(cabinet) == cabinet.drawers[0].handle.name
    )


# %% a step that maps one value to several


def test_chain_through_an_index_by_an_expression_has_no_single_value(
    handles_and_containers_world,
):
    """
    Indexing by an expression reaches one element per value that expression takes, so
    the chain has no one value to follow even though it is an index.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) > 1)
    position = variable(int, domain=[0, 1])
    chain = variable(Cabinet, domain=cabinets).drawers[position].handle.name

    with pytest.raises(MultipleValuesAlongAccessPath):
        chain.apply_mapping_on_external_root(cabinet)


def test_chain_through_a_flattened_attribute_has_no_single_value(
    handles_and_containers_world,
):
    """
    Flattening a collection leaves the rest of the chain with an element per item rather
    than one value, which the walk reports instead of silently following the first.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) > 1)
    chain = flat_variable(variable(Cabinet, domain=cabinets).drawers).handle.name

    with pytest.raises(MultipleValuesAlongAccessPath):
        chain.apply_mapping_on_external_root(cabinet)


def test_chain_through_a_flattened_attribute_is_rejected_whatever_the_collection_holds(
    handles_and_containers_world,
):
    """
    Whether a chain reaches one value is decided by the mappings it is built from, not
    by how many elements a particular instance happens to hold, so a flattening is
    rejected even where it would have reached exactly one.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) == 1)
    chain = flat_variable(variable(Cabinet, domain=cabinets).drawers).handle.name

    with pytest.raises(MultipleValuesAlongAccessPath):
        chain.apply_mapping_on_external_root(cabinet)


# %% writing through a chain


def test_setting_through_an_index_by_a_value_writes_that_element(
    handles_and_containers_world,
):
    """
    Indexing by a plain value names where the element is stored, so a chain ending in
    one can write it back.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) > 1)
    replacement = cabinet.drawers[1]
    chain = variable(Cabinet, domain=cabinets).drawers[0]

    chain._set_external_root_instance_value_(cabinet, replacement)

    assert cabinet.drawers[0] is replacement


def test_setting_through_an_index_by_an_expression_is_not_supported(
    handles_and_containers_world,
):
    """
    An expression names which elements the indexing reaches, not where one is stored, so
    a chain ending in one has nowhere to write back to.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) > 1)
    position = variable(int, domain=[0, 1])
    chain = variable(Cabinet, domain=cabinets).drawers[position]

    with pytest.raises(ReadOnlyMapping):
        chain._set_external_root_instance_value_(cabinet, cabinet.drawers[1])


def test_setting_through_a_flattened_attribute_has_no_single_value(
    handles_and_containers_world,
):
    """
    Writing follows the chain to the value it sets, so a step that reaches several
    values leaves it without one to follow, just as reading does.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) > 1)
    chain = flat_variable(variable(Cabinet, domain=cabinets).drawers).handle.name

    with pytest.raises(MultipleValuesAlongAccessPath):
        chain._set_external_root_instance_value_(cabinet, "Handle9")

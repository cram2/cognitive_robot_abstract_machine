"""
:meth:`MappedVariable.apply_mapping_on_external_root` follows a chain from a value
outside query evaluation, which is how features are read off an instance.

These tests pin what it does when a step along the way maps one value to several, when a
step maps it to none, and when it meets a pattern standing for an instance.
"""

import pytest

from krrood.entity_query_language.exceptions import (
    MultipleValuesAlongAccessPath,
    NoValueAlongAccessPath,
    ReadOnlyMapping,
)
from krrood.entity_query_language.factories import a, flat_variable, variable

from ...dataset.semantic_world_like_classes import Cabinet, Container, Drawer, Handle

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


# %% a step that maps one value to none


def test_chain_through_an_index_nothing_is_stored_under_reaches_no_value(
    handles_and_containers_world,
):
    """
    An index reaches the element stored under it, so a chain through one that holds
    nothing says so rather than ending in an exhausted walk.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) == 1)
    chain = variable(Cabinet, domain=cabinets).drawers[1].handle.name

    with pytest.raises(NoValueAlongAccessPath):
        chain.apply_mapping_on_external_root(cabinet)


def test_a_step_that_reaches_no_value_is_reported_the_same_from_inside_a_generator(
    handles_and_containers_world,
):
    """
    Reading a chain off an instance is done from generators -- the backends evaluating a
    query are ones -- where an exhausted walk would be read as that generator's own end
    and surface as an unrelated failure instead.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) == 1)
    chain = variable(Cabinet, domain=cabinets).drawers[1].handle.name

    def following_it():
        yield chain.apply_mapping_on_external_root(cabinet)

    with pytest.raises(NoValueAlongAccessPath):
        list(following_it())


# %% a pattern along the chain


def test_chain_through_a_pattern_reaches_the_value_it_states():
    """
    A pattern stands for the instance it describes, so a step on it reaches the value
    the pattern states for that attribute.
    """
    handle = Handle(name="Handle1")
    cabinet = Cabinet(
        container=Container(name="Container1"),
        drawers=[a(Drawer)(handle=handle, container=Container(name="Container2"))],
    )
    chain = variable(Cabinet, domain=[cabinet]).drawers[0].handle

    assert chain.apply_mapping_on_external_root(cabinet) is handle


def test_chain_through_a_pattern_stating_a_pattern_reaches_the_inner_stated_value():
    """
    A pattern stated inside a pattern stands for its instance too, so the chain keeps
    reading stated values until it reaches the end.
    """
    handle_name = "Handle1"
    cabinet = Cabinet(
        container=Container(name="Container1"),
        drawers=[a(Drawer)(handle=a(Handle)(name=handle_name))],
    )
    chain = variable(Cabinet, domain=[cabinet]).drawers[0].handle.name

    assert chain.apply_mapping_on_external_root(cabinet) is handle_name


def test_chain_written_on_a_pattern_reaches_the_value_it_states():
    """
    A pattern reads like the instance it describes, so a chain written on the pattern
    itself, applied back to it, reaches the values it states.
    """
    handle_name = "Handle1"
    drawer = a(Drawer)(handle=a(Handle)(name=handle_name))

    assert drawer.handle.name.apply_mapping_on_external_root(drawer) is handle_name


def test_chain_through_a_pattern_that_does_not_state_the_attribute_reaches_no_value():
    """
    A pattern that leaves an attribute unstated gives the chain no value to follow,
    which is reported like any other step that reaches nothing.
    """
    cabinet = Cabinet(
        container=Container(name="Container1"),
        drawers=[a(Drawer)(container=Container(name="Container2"))],
    )
    chain = variable(Cabinet, domain=[cabinet]).drawers[0].handle

    with pytest.raises(NoValueAlongAccessPath):
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


def test_setting_through_a_step_that_reaches_no_value_is_reported(
    handles_and_containers_world,
):
    """
    Writing follows the chain to the value it sets, so a step that reaches nothing
    leaves it with nothing to write to, just as reading has nothing to read.
    """
    cabinets = [
        view for view in handles_and_containers_world.views if isinstance(view, Cabinet)
    ]
    cabinet = next(cabinet for cabinet in cabinets if len(cabinet.drawers) == 1)
    chain = variable(Cabinet, domain=cabinets).drawers[1].handle

    with pytest.raises(NoValueAlongAccessPath):
        chain._set_external_root_instance_value_(cabinet, cabinet.drawers[0].handle)

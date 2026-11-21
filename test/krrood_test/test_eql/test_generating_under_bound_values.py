"""
Generating the instances a match describes while some of its variables are bound
already, and reporting the values its variables took for each instance.
"""

from itertools import product

import pytest

from krrood.entity_query_language.backends import EntityQueryLanguageGenerativeBackend
from krrood.entity_query_language.factories import a, variable
from ..dataset.choices_made_in_one_step_and_used_in_another import (
    Item,
    TakingAndPutting,
    Putting,
    Taking,
    Tool,
)

# %% fixtures


@pytest.fixture
def tools() -> list[Tool]:
    """
    :return: Two tools, the first reaching both items of :func:`items`, the second only
        the last one.
    """
    first_item, second_item = Item(label="first item"), Item(label="second item")
    return [
        Tool(label="first tool", reachable_items=[first_item, second_item]),
        Tool(label="second tool", reachable_items=[second_item]),
    ]


@pytest.fixture
def items(tools: list[Tool]) -> list[Item]:
    """
    :return: The items the first of :func:`tools` reaches.
    """
    return tools[0].reachable_items


# %% reporting the values of the variables


def test_each_instance_comes_with_the_values_its_variables_took(
    tools: list[Tool], items: list[Item]
) -> None:
    """
    Every generated instance reports the value each variable it was given took for it.
    """
    tool = variable(Tool, domain=tools)
    item = variable(Item, domain=items)

    generated = list(
        EntityQueryLanguageGenerativeBackend().generate(
            expression=a(Taking)(tool=tool, item=item), bound_values={}
        )
    )

    assert [each.variable_values for each in generated] == [
        {tool._id_: each_tool, item._id_: each_item}
        for each_tool, each_item in product(tools, items)
    ]
    assert [(each.instance.tool, each.instance.item) for each in generated] == list(
        product(tools, items)
    )


def test_a_value_the_match_states_is_not_a_variable_value(
    tools: list[Tool], items: list[Item]
) -> None:
    """
    A plain value the match states is no variable, so it is not reported.
    """
    tool = variable(Tool, domain=tools)

    generated = list(
        EntityQueryLanguageGenerativeBackend().generate(
            expression=a(Taking)(tool=tool, item=items[0]), bound_values={}
        )
    )

    assert [each.variable_values for each in generated] == [
        {tool._id_: each_tool} for each_tool in tools
    ]


# %% generating under bound values


def test_a_bound_variable_takes_its_bound_value(
    tools: list[Tool], items: list[Item]
) -> None:
    """
    A variable that has a value already is not enumerated over its domain.
    """
    tool = variable(Tool, domain=tools)
    item = variable(Item, domain=items)

    generated = EntityQueryLanguageGenerativeBackend().generate(
        expression=a(Putting)(tool=tool, item=item),
        bound_values={tool._id_: tools[1]},
    )

    assert [(each.instance.tool, each.instance.item) for each in generated] == [
        (tools[1], each_item) for each_item in items
    ]


def test_a_domain_read_from_a_bound_variable_is_read_from_its_bound_value(
    tools: list[Tool],
) -> None:
    """
    A variable whose domain is read from a bound variable ranges only over what the
    bound value offers.
    """
    tool = variable(Tool, domain=tools)
    item = variable(Item, domain=tool.reachable_items)

    generated = EntityQueryLanguageGenerativeBackend().generate(
        expression=a(Putting)(tool=tool, item=item),
        bound_values={tool._id_: tools[1]},
    )

    assert [(each.instance.tool, each.instance.item) for each in generated] == [
        (tools[1], each_item) for each_item in tools[1].reachable_items
    ]


def test_a_value_bound_for_a_variable_the_match_does_not_use_changes_nothing(
    tools: list[Tool], items: list[Item]
) -> None:
    """
    A value bound for a variable the match never mentions leaves the instances as they
    are.
    """
    tool = variable(Tool, domain=tools)
    unused = variable(Tool, domain=tools)
    taking = a(Taking)(tool=tool, item=items[0])

    generated = EntityQueryLanguageGenerativeBackend().generate(
        expression=taking, bound_values={unused._id_: tools[1]}
    )

    assert [each.instance.tool for each in generated] == tools


def test_a_variable_used_twice_in_one_match_takes_one_value(
    tools: list[Tool], items: list[Item]
) -> None:
    """
    A variable given to two nested matches takes the same value in both.
    """
    tool = variable(Tool, domain=tools)
    item = variable(Item, domain=items)
    taking_and_putting = a(TakingAndPutting)(
        taking=a(Taking)(tool=tool, item=item),
        putting=a(Putting)(tool=tool, item=item),
    )

    generated = EntityQueryLanguageGenerativeBackend().generate(
        expression=taking_and_putting, bound_values={}
    )

    assert [
        (
            each.instance.taking.tool,
            each.instance.taking.item,
            each.instance.putting.tool,
            each.instance.putting.item,
        )
        for each in generated
    ] == [
        (each_tool, each_item, each_tool, each_item)
        for each_tool, each_item in product(tools, items)
    ]

"""
Calling a method in a query with arguments that cannot be hashed.
"""

from krrood.entity_query_language.factories import a, variable

from ...dataset.unhashable_call_arguments import LabelHolder, UnhashableLabel

# %% evaluating the call


def test_call_with_an_unhashable_argument_is_evaluated():
    red = UnhashableLabel("red")
    holder = LabelHolder(red)
    holders = variable(LabelHolder, [holder, LabelHolder(UnhashableLabel("blue"))])

    assert a(holders).where(holders.carries(red)).tolist() == [holder]


def test_call_with_unhashable_values_inside_a_tuple_is_evaluated():
    red = UnhashableLabel("red")
    holder = LabelHolder(red)
    holders = variable(LabelHolder, [holder])

    query = a(holders).where(holders.carries(UnhashableLabel("blue"), also=(red,)))

    assert query.tolist() == [holder]


# %% the node a call is written as


def test_the_same_call_written_twice_is_one_node():
    red = UnhashableLabel("red")
    holders = variable(LabelHolder, [])

    assert holders.carries(red, also=(red,)) is holders.carries(red, also=(red,))


def test_equal_but_distinct_unhashable_arguments_are_different_nodes():
    holders = variable(LabelHolder, [])

    assert holders.carries(UnhashableLabel("red")) is not holders.carries(
        UnhashableLabel("red")
    )

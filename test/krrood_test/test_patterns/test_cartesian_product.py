from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Iterator

from krrood.patterns.cartesian_product import CartesianProductByRankSum

# %% a factor recording how far it was read


@dataclass(kw_only=True)
class RecordedValues:
    """
    A one-off source of values that records how many of them were read.
    """

    values: list[str]
    """
    The values, in the order they are read.
    """

    read: int = field(default=0, init=False)
    """
    How many values were read so far.
    """

    def __iter__(self) -> Iterator[str]:
        for value in self.values:
            self.read += 1
            yield value


# %% the order of the tuples


def test_the_values_ranked_first_are_combined_before_later_ones() -> None:
    """
    Every tuple comes up once, and none takes a later value of a factor before the
    tuples of the values ranked before it.
    """
    product = CartesianProductByRankSum(factors=[["a1", "a2"], ["b1", "b2", "b3"]])

    assert list(product) == [
        ("a1", "b1"),
        ("a1", "b2"),
        ("a2", "b1"),
        ("a1", "b3"),
        ("a2", "b2"),
        ("a2", "b3"),
    ]


def test_a_factor_without_values_leaves_no_tuple() -> None:
    product = CartesianProductByRankSum(factors=[["a1"], []])

    assert list(product) == []


def test_a_value_equal_to_none_is_a_value_like_any_other() -> None:
    product = CartesianProductByRankSum(factors=[[None, "a2"], ["b1"]])

    assert list(product) == [(None, "b1"), ("a2", "b1")]


# %% reading the factors


def test_a_factor_is_read_only_as_far_as_a_tuple_reaches_into_it() -> None:
    first = RecordedValues(values=["a1", "a2", "a3"])
    second = RecordedValues(values=["b1", "b2", "b3"])
    tuples = iter(CartesianProductByRankSum(factors=[iter(first), iter(second)]))

    next(tuples)

    assert (first.read, second.read) == (1, 1)


def test_the_values_read_from_each_factor_are_kept_in_the_order_it_yielded_them() -> (
    None
):
    product = CartesianProductByRankSum(
        factors=[iter(["a1", "a2"]), iter(["b1", "b2", "b3"])]
    )
    tuples = iter(product)

    [next(tuples) for _ in range(3)]

    assert product.read == [["a1", "a2"], ["b1", "b2"]]

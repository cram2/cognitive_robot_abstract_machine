"""
A set of examples as the pipelines see it: split into training and held-out parts, its
parts reordered, and the rate of the effect the questions ask about summarised.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from typing_extensions import Any, Callable, Dict, List, Self, Tuple, TypeVar

from experiments.causal_reasoning.comparison.domain import RelationalDomain

T = TypeVar("T")


@dataclass(frozen=True)
class EffectRate:
    """
    How often a group of examples shows the effect.
    """

    example_count: int
    """
    How many examples the group holds.
    """

    effect_count: int
    """
    How many of them show the effect.
    """

    @property
    def rate(self) -> float:
        """
        The share that do.
        """
        return self.effect_count / self.example_count


@dataclass
class ExampleDataset:
    """
    A set of examples, as read and as the pipelines see them.
    """

    domain: RelationalDomain
    """
    The example and its parts.
    """

    examples: List[Any] = field(default_factory=list)
    """
    The examples.
    """

    def shows_effect(self, example: Any) -> bool:
        """
        :param example: An example.
        :return: Whether it shows the effect the questions ask about.
        """
        return bool(vars(example)[self.domain.effect_field])

    @property
    def effect_rate(self) -> float:
        """
        Share of examples that show the effect.
        """
        return sum(self.shows_effect(example) for example in self.examples) / len(
            self.examples
        )

    def effect_rate_by(self, key: Callable[[Any], T]) -> Dict[T, EffectRate]:
        """
        How often the examples sharing a value show the effect.

        :param key: What to group the examples by.
        :return: Each value's rate, by value.
        """
        by_value: Dict[T, List[Any]] = {}
        for example in self.examples:
            by_value.setdefault(key(example), []).append(example)
        return {
            value: EffectRate(
                example_count=len(examples),
                effect_count=sum(self.shows_effect(example) for example in examples),
            )
            for value, examples in sorted(by_value.items())
        }

    def with_shuffled_parts(self, random_state: np.random.Generator) -> Self:
        """
        The same examples with every kind of part in a random order each.

        :param random_state: Source of randomness for the orders.
        :return: The dataset with reordered parts.
        """
        return replace(
            self,
            examples=[
                replace(
                    example,
                    **{
                        part_field: [
                            parts[index]
                            for index in random_state.permutation(len(parts))
                        ]
                        for part_field in self.domain.part_fields
                        for parts in [self.domain.parts_of(example, part_field)]
                    },
                )
                for example in self.examples
            ],
        )

    def split(
        self, train_fraction: float, random_state: np.random.Generator
    ) -> Tuple[Self, Self]:
        """
        Shuffle the examples and split them in two.

        :param train_fraction: Share of examples that go into the first part.
        :param random_state: Source of randomness for the shuffle.
        :return: The first and second part.
        """
        order = random_state.permutation(len(self.examples))
        split_index = int(train_fraction * len(self.examples))
        first = [self.examples[index] for index in order[:split_index]]
        second = [self.examples[index] for index in order[split_index:]]
        return replace(self, examples=first), replace(self, examples=second)

"""
Dataclasses whose fields hold ordered, nested and keyed collections, for checking that
they read back from the database as they were stored.
"""

from __future__ import annotations

from dataclasses import dataclass

from .example_classes import Entity

# %% values


@dataclass(kw_only=True, frozen=True)
class RecipeStep:
    """
    A mapped value that collections hold, equal and hashed by its fields.
    """

    number: int
    """
    The position of the step in the recipe it was written for.
    """


# %% collections


@dataclass(kw_only=True)
class OrderedRecipe:
    """
    A class with a field of every kind of collection ORMatic stores.
    """

    steps: list[RecipeStep]
    """
    A list of mapped objects.
    """

    step_tuple: tuple[RecipeStep, ...]
    """
    A tuple of mapped objects.
    """

    step_set: set[RecipeStep]
    """
    A set of mapped objects.
    """

    step_lists: list[list[RecipeStep]]
    """
    A nested list of mapped objects.
    """

    steps_by_name: dict[str, RecipeStep]
    """
    A dictionary whose values are mapped objects.
    """

    names_by_step: dict[RecipeStep, str]
    """
    A dictionary whose keys are mapped objects.
    """

    step_lists_by_name: dict[str, list[RecipeStep]]
    """
    A dictionary whose values are lists of mapped objects.
    """

    step_sets_by_step: dict[RecipeStep, set[RecipeStep]]
    """
    A dictionary keyed by mapped objects whose values are sets of mapped objects.
    """

    amounts_by_step: dict[RecipeStep, int]
    """
    A dictionary keyed by mapped objects whose values are builtins.
    """

    counts_by_name: dict[str, int]
    """
    A dictionary of builtins.
    """

    measurements_by_number: dict[int, list[float]]
    """
    A dictionary of builtins whose keys are not strings.
    """

    number_sets: list[set[int]]
    """
    A nested collection of builtins.
    """


@dataclass(kw_only=True)
class NestedAlternativelyMappedEntities:
    """
    A nested collection of objects whose class is stored through an alternative mapping.
    """

    entity_lists: list[list[Entity]]
    """
    The nested collection.
    """

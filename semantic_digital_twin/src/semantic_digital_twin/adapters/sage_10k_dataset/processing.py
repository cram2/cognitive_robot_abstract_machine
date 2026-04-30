from krrood.entity_query_language.factories import *
from semantic_digital_twin.adapters.sage_10k_dataset.semantic_annotations import (
    NaturalLanguageDescriptionWithTypeDescription,
)
from semantic_digital_twin.pipeline.pipeline import BodyFilter, Pipeline
from semantic_digital_twin.reasoning.predicates import is_supported_by
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from semantic_digital_twin.world import World


def remove_clutter(
    cluttered_world: World,
    non_clutter: List[NaturalLanguageDescriptionWithTypeDescription],
):
    """
    Remove every body not in the non_clutter list and not where there is no structural annotation pointing to
    the body.

    Modified the world in-place.

    :param cluttered_world: The world with clutter bodies in it
    :param non_clutter: The list of bodies that should not be removed
    :return: The world with clutter removed
    """
    structural_annotation = variable(HasRootBody, cluttered_world.semantic_annotations)
    structural_annotations = (
        an(entity(structural_annotation))
        .where(
            not_(
                HasType(
                    structural_annotation, NaturalLanguageDescriptionWithTypeDescription
                )
            )
        )
        .tolist()
    )
    kept_bodies = [
        annotation.root for annotation in structural_annotations + non_clutter
    ] + [cluttered_world.root]

    bottles_and_benches_filter = BodyFilter(lambda x: x in kept_bodies)
    preprocessing_pipeline = Pipeline(
        [
            bottles_and_benches_filter,
        ]
    )

    for semantic_annotation in cluttered_world.get_semantic_annotations_by_type(
        NaturalLanguageDescription
    ):
        if semantic_annotation.root._world is None:
            cluttered_world.remove_semantic_annotation(semantic_annotation)

    return preprocessing_pipeline.apply(cluttered_world)


def bottles_on_benches(
    world,
) -> Tuple[
    List[NaturalLanguageDescriptionWithTypeDescription],
    List[NaturalLanguageDescriptionWithTypeDescription],
]:

    bench = variable(
        NaturalLanguageDescriptionWithTypeDescription, world.semantic_annotations
    )
    benches = an(entity(bench).where(contains(bench.type_description, "bench")))

    bottle = variable(
        NaturalLanguageDescriptionWithTypeDescription, world.semantic_annotations
    )
    bottles = an(
        entity(bottle).where(
            contains(bottle.type_description, "bottle"),
            is_supported_by(bottle.root, benches.root),
        )
    )
    return bottles.tolist(), benches.tolist()


def open_door(robot: AbstractRobot, door: Door): ...

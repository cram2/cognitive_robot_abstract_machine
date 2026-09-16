"""Load real robocasa objects as data for the confidence-aware EQL pipeline.

Bridges :class:`~semantic_digital_twin.adapters.robocasa_dataset.loader.RoboCasaDatasetLoader`
to the confidence pipeline: every downloaded instance of every object category the
adapter's own resolver knows how to classify is loaded as its real semantic-annotation
instance, with no category singled out - the confidence model should be built on
whatever robocasa actually provides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.adapters.robocasa_dataset.loader import RoboCasaDatasetLoader
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaObjectCategory,
    RoboCasaObjectResolver,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import Dict, List, Type


@dataclass
class RoboCasaInstanceLoadFailure:
    """One downloaded robocasa object instance that failed to parse into a world."""

    category: RoboCasaObjectCategory
    """The object category the failed instance belongs to."""

    instance_index: int
    """Which of the category's downloaded instances failed."""

    error: Exception
    """The exception the world parser raised for this instance."""


@dataclass
class RoboCasaObjectLoadResult:
    """Every robocasa object instance that could be loaded, and every one that could not.

    Loading a real, third-party asset catalog is expected to hit occasional instances
    the world parser cannot handle; those are recorded here rather than aborting the
    whole load, so a caller can inspect exactly what was skipped and why.
    """

    instances_by_class: Dict[Type[SemanticAnnotation], List[SemanticAnnotation]] = field(
        default_factory=dict
    )
    """Mapping from each resolved semantic-annotation class to its loaded instances."""

    failures: List[RoboCasaInstanceLoadFailure] = field(default_factory=list)
    """Every downloaded instance that failed to load, with its cause."""


def downloaded_instance_count(
    loader: RoboCasaDatasetLoader, category: RoboCasaObjectCategory
) -> int:
    """Count the downloaded model instances available for one object category.

    :param loader: Loader configured with the directory the robocasa kitchen assets
        were downloaded into.
    :param category: The object category to count downloaded instances of.
    :return: The number of instances found on disk, ``0`` if none were downloaded.
    """
    objects_directory = loader.directory / "objects"
    return sum(
        1
        for group in loader.self_contained_object_groups
        for _ in (objects_directory / group).glob(f"{category.value}/**/model.xml")
    )


def load_all_robocasa_objects(
    loader: RoboCasaDatasetLoader, max_instances_per_category: int = 10
) -> RoboCasaObjectLoadResult:
    """Load downloaded instances of every object category robocasa provides.

    Categories with no downloaded instances contribute an empty list rather than
    being skipped, so the result always reflects the resolver's full category set. An
    instance the world parser cannot load is recorded as a failure rather than
    aborting the load of every other instance.

    Loading one instance leaves memory it does not release (traced as far as the
    MuJoCo bindings loading its mesh, outside this project's control), so no more than
    ``max_instances_per_category`` instances are loaded per category to keep the total
    load within a single process's memory budget; this caps every category alike
    rather than singling any one out. The default was measured empirically: loading
    every category's full downloaded catalog exhausted a 7.8GB container's memory
    before finishing, while capping at 10 completed cleanly at roughly 2.1GB peak.

    :param loader: Loader configured with the directory the robocasa kitchen assets
        were downloaded into.
    :param max_instances_per_category: The most instances to load for any one
        category, applied uniformly.
    :return: Every instance that loaded successfully, and every one that did not.
    """
    resolver = RoboCasaObjectResolver()
    result = RoboCasaObjectLoadResult()
    for category, annotation_class in resolver.category_to_annotation_class.items():
        instance_count = min(
            downloaded_instance_count(loader, category), max_instances_per_category
        )
        instances = []
        for instance_index in range(instance_count):
            try:
                world = loader.load_object(category, instance_index=instance_index)
            except Exception as error:  # noqa: BLE001 - third-party asset parsing, recorded not raised
                result.failures.append(
                    RoboCasaInstanceLoadFailure(category, instance_index, error)
                )
                continue
            [annotation] = [
                annotation
                for annotation in world.semantic_annotations
                if isinstance(annotation, annotation_class)
            ]
            instances.append(annotation)
        result.instances_by_class[annotation_class] = instances
    return result

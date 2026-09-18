"""Load real robocasa objects as data for the confidence-aware EQL pipeline.

Bridges :class:`~semantic_digital_twin.adapters.robocasa_dataset.loader.RoboCasaDatasetLoader`
to the confidence pipeline: every downloaded instance of every object category the
adapter's own resolver knows how to classify is loaded as its real semantic-annotation
instance, with no category singled out - the confidence model should be built on
whatever robocasa actually provides.
"""

from __future__ import annotations

from semantic_digital_twin.adapters.robocasa_dataset.loader import RoboCasaDatasetLoader
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaObjectCategory,
    RoboCasaObjectResolver,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import Dict, List, Type


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
) -> Dict[Type[SemanticAnnotation], List[SemanticAnnotation]]:
    """Load downloaded instances of every object category robocasa provides.

    Categories with no downloaded instances contribute an empty list rather than
    being skipped, so the result always reflects the resolver's full category set.

    Loading one instance leaves memory it does not release (traced as far as the
    MuJoCo bindings loading its mesh, outside this project's control), so no more than
    ``max_instances_per_category`` instances are loaded per category to keep the total
    load within a single process's memory budget; this caps every category alike
    rather than singling any one out. The default was measured empirically: loading
    every category's full downloaded catalog exhausted a 7.8GB container's memory
    before finishing, while capping at 10 completed cleanly at roughly 2.1GB peak.

    ..note::
        An asset the world parser cannot read raises, rather than being collected and
        reported as a partial result: a downloaded asset that does not parse is a
        defect to fix in the parser, not an outcome for callers to sift through.

    :param loader: Loader configured with the directory the robocasa kitchen assets
        were downloaded into.
    :param max_instances_per_category: The most instances to load for any one
        category, applied uniformly.
    :return: Mapping from each resolved semantic-annotation class to its loaded instances.
    """
    resolver = RoboCasaObjectResolver()
    instances_by_class: Dict[Type[SemanticAnnotation], List[SemanticAnnotation]] = {}
    for category, annotation_class in resolver.category_to_annotation_class.items():
        instance_count = min(
            downloaded_instance_count(loader, category), max_instances_per_category
        )
        instances = []
        for instance_index in range(instance_count):
            world = loader.load_object(category, instance_index=instance_index)
            [annotation] = [
                annotation
                for annotation in world.semantic_annotations
                if isinstance(annotation, annotation_class)
            ]
            instances.append(annotation)
        instances_by_class[annotation_class] = instances
    return instances_by_class

import os
from importlib.resources import files
from pathlib import Path

from krrood.entity_query_language.factories import *
from pycram.robot_plans.actions.base import ActionDescription
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from semantic_digital_twin.adapters.sage_10k_dataset.semantic_annotations import (
    NaturalLanguageDescriptionWithTypeDescription,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.pipeline.pipeline import BodyFilter, Pipeline
from semantic_digital_twin.reasoning.predicates import is_supported_by
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    GraphOfConvexSets,
    navigation_map_at_target,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body


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


def create_pr2_in_world(world: World):
    pr2_urdf = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"

    pr2_parser = URDFParser.from_file(file_path=pr2_urdf)
    world_with_pr2 = pr2_parser.parse()
    with world_with_pr2.modify_world():
        pr2_root = world_with_pr2.root
        localization_body = Body(name=PrefixedName("odom_combined"))
        world_with_pr2.add_kinematic_structure_entity(localization_body)
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=pr2_root, world=world_with_pr2
        )
        world_with_pr2.add_connection(c_root_bf)

    world.merge_world(world_with_pr2)

    pr2 = PR2.from_world(world)
    return pr2


def create_hsrb_in_world(world: World):

    urdf_dir = os.path.join(
        Path(files("semantic_digital_twin")).parent.parent.parent,
        "pycram",
        "resources",
        "robots",
    )
    hsr = os.path.join(urdf_dir, "hsrb.urdf")

    hsrb_parser = URDFParser.from_file(file_path=hsr)
    world_with_hsrb = hsrb_parser.parse()
    with world_with_hsrb.modify_world():
        hsrb_root = world_with_hsrb.root
        localization_body = Body(name=PrefixedName("odom_combined"))
        world_with_hsrb.add_kinematic_structure_entity(localization_body)
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=hsrb_root, world=world_with_hsrb
        )
        world_with_hsrb.add_connection(c_root_bf)

    world.merge_world(world_with_hsrb)

    hsrb = HSRB.from_world(world)
    return hsrb

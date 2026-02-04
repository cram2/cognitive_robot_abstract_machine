from __future__ import annotations

import os
import re
import time
from typing import List

import tqdm
from sqlalchemy.orm import Session

from krrood.entity_query_language.symbol_graph import SymbolGraph
from krrood.ormatic.dao import to_dao
from krrood.ormatic.utils import drop_database, create_engine
from krrood.utils import recursive_subclasses
from semantic_digital_twin.world import World

sg = SymbolGraph()

from semantic_digital_twin.adapters.fbx import FBXParser
from semantic_digital_twin.adapters.procthor.procthor_pipelines import (
    dresser_from_body_in_world,
)
from semantic_digital_twin.orm.ormatic_interface import *
from semantic_digital_twin.adapters.procthor.procthor_resolver import (
    ProcthorResolver,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
)
from semantic_digital_twin.pipeline.pipeline import (
    Pipeline,
    BodyFilter,
    BodyFactoryReplace,
    CenterLocalGeometryAndPreserveWorldPose,
)


def remove_root_and_move_children_into_new_worlds(world: World) -> List[World]:
    """
    Remove the root of the given world and move its children into new worlds.
    Each child that has a parent with "grp" in its name and does not have "grp" in its own name
    will be moved to a new world, (sometimes groups are nested).
    The new world's name will be set to the child's name.

    :param world: The World object to process. This world will be unusable after this operation.
    :return: List of new World objects created from the root's children.
    """
    root_children = [
        entity
        for entity in world.kinematic_structure_entities
        if entity.parent_kinematic_structure_entity
        and "grp" in entity.parent_kinematic_structure_entity.name.name
        and "grp" not in entity.name.name
    ]

    with world.modify_world():

        worlds = [world.move_branch_to_new_world(child) for child in root_children]
        for world in worlds:
            if world.root is None:
                ...
            world.name = world.root.name.name

    return worlds


def replace_dresser_meshes_with_factories(
    worlds: List[World], dresser_pattern: re.Pattern[str]
) -> List[World]:
    """
    Replace dresser meshes in the given worlds with dresser factories.
    A dresser is identified by its name matching the given dresser_pattern regex.

    :param worlds: List of World objects to process.
    :param dresser_pattern: A compiled regex pattern to identify dresser bodies.
    :return: List of World objects with dresser meshes replaced by factories.
    """
    procthor_replace_pipeline = Pipeline(
        [
            BodyFactoryReplace(
                body_condition=lambda b: bool(dresser_pattern.fullmatch(b.name.name))
                and not (
                    "drawer" in b.name.name.lower() or "door" in b.name.name.lower()
                ),
                annotation_creator=dresser_from_body_in_world,
            )
        ]
    )
    worlds = [procthor_replace_pipeline.apply(w) for w in worlds]
    return worlds


def parse_fbx_file_to_world_mapping_daos(fbx_file_path: str) -> List[WorldMappingDAO]:
    """
    Parse a Procthor FBX file path and return a list of WorldMappingDAO objects.
    """
    dresser_pattern = re.compile(r"^.*dresser_(?!drawer\b).*$", re.IGNORECASE)

    pipeline = Pipeline(
        [
            CenterLocalGeometryAndPreserveWorldPose(),
            BodyFilter(lambda x: not x.name.name.startswith("PS_")),
            BodyFilter(lambda x: not x.name.name.endswith("slice")),
        ]
    )

    parser = FBXParser(fbx_file_path)
    world = parser.parse()
    world = pipeline.apply(world)

    worlds = remove_root_and_move_children_into_new_worlds(world)

    worlds = replace_dresser_meshes_with_factories(worlds, dresser_pattern)
    resolver = ProcthorResolver(*[recursive_subclasses(HasRootBody)])
    for world in worlds:
        resolved = resolver.resolve(world.name)
        if resolved:
            with world.modify_world():
                world.add_semantic_annotation(resolved(body=world.root))

    return [to_dao(world) for world in worlds]


def parse_procthor_files_and_save_to_database(
    fbx_file_pattern: re.Pattern[str] = re.compile(r".*_grp\.fbx$", re.IGNORECASE),
    drop_existing_database: bool = True,
):
    """
    Parse all Procthor FBX files and store the resulting WorldMappingDAO objects in a database.
    Currently, only grp files are parsed, and some files and names are excluded.
    TODO: Ensure all relevant files, even those not inside a grp, are parsed.
    """
    semantic_digital_twin_database_uri = os.environ.get(
        "SEMANTIC_DIGITAL_TWIN_DATABASE_URI"
    )
    assert (
        semantic_digital_twin_database_uri is not None
    ), "Please set the SEMANTIC_DIGITAL_TWIN_DATABASE_URI environment variable."

    procthor_root = os.path.join(os.path.expanduser("~"), "ai2thor")
    procthor_root = os.path.join(os.path.expanduser("~"), "work", "ai2thor")

    files = []
    for root, dirs, filenames in os.walk(procthor_root):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    assert (
        len(files) > 0
    ), f"No files found in {procthor_root}, please set the correct path to the ai2thor directory."

    excluded_words = [
        "FirstPersonCharacter",
        "SourceFiles_Procedural",
        "RobotArmTest",
        "_shards_",
    ]

    fbx_files = [
        f
        for f in files
        if not any([e in f for e in excluded_words]) and fbx_file_pattern.fullmatch(f)
    ]
    # Create database engine and session
    engine = create_engine(semantic_digital_twin_database_uri)
    session = Session(engine)

    if drop_existing_database:
        drop_database(engine)
        Base.metadata.create_all(engine)

    start_time = time.time_ns()

    dao_names = []
    daos = []

    for i, fbx_file in tqdm.tqdm(enumerate(fbx_files)):
        for dao in parse_fbx_file_to_world_mapping_daos(fbx_file):
            # Some item names (for example "bowl_19") were used for multiple items. For now the solution is to just
            # skip duplicate names.
            if dao.name not in dao_names:
                dao_names.append(dao.name)
                daos.append(dao)

    session.add_all(daos)
    session.commit()
    print(
        f"Parsing {len(fbx_files)} files took {(time.time_ns() - start_time) / 1e9} seconds"
    )


if __name__ == "__main__":
    parse_procthor_files_and_save_to_database()

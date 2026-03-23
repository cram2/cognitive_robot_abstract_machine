
from __future__ import annotations

import os
import pathlib
from typing_extensions import Optional, Tuple, Type

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.package_resolver import PathResolver
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk, Cereal
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    DifferentialDrive,
    Connection6DoF,
)
from semantic_digital_twin.world_description.world_entity import Body

from pycram.datastructures.dataclasses import Context

# ── Path resolution ────────────────────────────────────────────────────────────
# This file lives at:  <repo>/cognitive_robot_abstract_machine/llmr/src/llmr/world_setup.py
# parents[2]  →  <repo>/cognitive_robot_abstract_machine/llmr/
# parents[3]  →  <repo>/cognitive_robot_abstract_machine/
_HERE = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]  # llmr/
_REPO_ROOT = _HERE.parents[3]  # cognitive_robot_abstract_machine/

_PYCRAM_RESOURCES = _REPO_ROOT / "pycram" / "resources"
_APARTMENT_URDF = str(_PYCRAM_RESOURCES / "worlds" / "apartment.urdf")
_MILK_STL = str(_PYCRAM_RESOURCES / "objects" / "milk.stl")
_CEREAL_STL = str(_PYCRAM_RESOURCES / "objects" / "breakfast_cereal.stl")
_PR2_URDF = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"


# ── Internal builders ──────────────────────────────────────────────────────────


def _build_apartment_world() -> World:
    """Parse the apartment URDF and place milk + cereal objects inside it."""
    apartment_world = URDFParser.from_file(_APARTMENT_URDF).parse()

    milk_world = STLParser(_MILK_STL).parse()
    cereal_world = STLParser(_CEREAL_STL).parse()

    apartment_world.merge_world_at_pose(
        milk_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.05, reference_frame=apartment_world.root
        ),
    )
    apartment_world.merge_world_at_pose(
        cereal_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 1.8, 1.05, reference_frame=apartment_world.root
        ),
    )

    milk_view = Milk(root=apartment_world.get_body_by_name("milk.stl"), _world=apartment_world)

    cereal_view = Cereal(
        root=apartment_world.get_body_by_name("breakfast_cereal.stl"), _world=apartment_world
    )

    with apartment_world.modify_world():
        apartment_world.add_semantic_annotation(milk_view)
        apartment_world.add_semantic_annotation(cereal_view)

    return apartment_world


def _build_urdf_world(
    urdf_path: str,
    robot_annotation: Optional[Type[AbstractRobot]],
    drive_type: Type,
    starting_pose: Optional[HomogeneousTransformationMatrix] = None,
    path_resolver: Optional[PathResolver] = None,
) -> World:
    """Parse a URDF and wire it into the map → odom_combined → robot tree."""
    world = URDFParser.from_file(file_path=urdf_path, path_resolver=path_resolver).parse()

    if robot_annotation is not None:
        robot_annotation.from_world(world)

    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        odom_body = Body(name=PrefixedName("odom_combined"))
        world.add_connection(Connection6DoF.create_with_dofs(world, map_body, odom_body))
        drive_conn = drive_type.create_with_dofs(parent=odom_body, child=world.root, world=world)
        world.add_connection(drive_conn)
        drive_conn.has_hardware_interface = True
        if starting_pose is not None:
            drive_conn.origin = starting_pose

    return world


def _build_pr2_world() -> World:
    return _build_urdf_world(_PR2_URDF, PR2, OmniDrive)


def _build_pr2_apartment_world() -> World:
    """Merge the PR2 robot world with the apartment + objects."""
    pr2_world = _build_pr2_world()
    apartment_world = _build_apartment_world()

    pr2_world.merge_world(apartment_world)
    pr2_world.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(1.3, 2, 0)
    )
    return pr2_world


# ── Public API ─────────────────────────────────────────────────────────────────


def load_pr2_apartment_world() -> Tuple[World, PR2, Context]:
    """Load a mutable PR2 apartment world ready for simulation.

    Builds the world from scratch on every call so that semantic annotations
    are preserved (no deepcopy).

    Usage::

        world, pr2, context = load_pr2_apartment_world()

    :return: ``(world, pr2, context)`` — a mutable simulation-ready tuple.
    """
    world = _build_pr2_apartment_world()
    pr2 = world.get_semantic_annotations_by_type(PR2)[0]
    return world, pr2, Context(world, pr2)

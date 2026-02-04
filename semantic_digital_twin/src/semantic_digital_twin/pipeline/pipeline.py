import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable

import numpy as np

from ..semantic_annotations.mixins import HasRootKinematicStructureEntity
from ..spatial_types import Point3
from ..spatial_types.spatial_types import HomogeneousTransformationMatrix
from ..world import World
from ..world_description.geometry import TriangleMesh, FileMesh
from ..world_description.world_entity import Body


@dataclass
class Step(ABC):
    """
    A Step is a transformation that takes a World as input and produces a modified World as output.
    Steps are intended to be used in a Pipeline, where the output World of one Step is passed as the input World to the next Step.
    """

    @abstractmethod
    def _apply(self, world: World) -> World: ...

    def apply(self, world: World) -> World:
        with world.modify_world():
            return self._apply(world)


@dataclass
class Pipeline:
    """
    A Pipeline is a sequence of Steps that are applied to a World in order.
    Each Step takes the World as input and produces a modified World as output.
    The output World of one Step is passed as the input World to the next Step.
    """

    steps: List[Step]
    """
    The list of Steps to be applied in the Pipeline.
    """

    def apply(self, world: World) -> World:
        for step in self.steps:
            world = step.apply(world)
        return world


@dataclass
class BodyFilter(Step):
    """
    Filters bodies in the world based on a given condition.
    """

    condition: Callable[[Body], bool]

    def _apply(self, world: World) -> World:
        for body in world.bodies:
            if not self.condition(body):
                world.remove_kinematic_structure_entity(body)
        return world


@dataclass
class CenterLocalGeometryAndPreserveWorldPose(Step):
    """
    Adjusts the vertices of the collision meshes of each body in the world so that the origin is at the center of the
    mesh, and then updates the parent connection of the body to preserve the original world pose.
    An example where this is useful is when parsing FBX files where all bodies in the resulting world have an origin
    at (0, 0, 0), even through the collision meshes are not centered around that point.
    """

    def _apply(self, world: World) -> World:
        for body in world.bodies_topologically_sorted:

            vertices = []

            for coll in body.collision:
                if isinstance(coll, (FileMesh, TriangleMesh)):
                    mesh = coll.mesh
                    if mesh.vertices.shape[0] > 0:
                        vertices.append(mesh.vertices.copy())

            if len(vertices) == 0:
                logging.warning(
                    f"Body {body.name.name} has no vertices in visual or collision shapes, skipping."
                )
                continue

            # Compute the axis-aligned bounding box center of all vertices
            all_vertices = np.vstack(vertices)
            mins = all_vertices.min(axis=0)
            maxs = all_vertices.max(axis=0)
            center = (mins + maxs) / 2.0

            for coll in body.collision:
                if isinstance(coll, (FileMesh, TriangleMesh)):
                    m = coll.mesh
                    if m.vertices.shape[0] > 0:
                        m.vertices -= center

            old_origin_T_new_origin = (
                HomogeneousTransformationMatrix.from_point_rotation_matrix(
                    Point3(*center)
                )
            )

            if body.parent_connection:
                parent_T_old_origin = (
                    body.parent_connection.parent_T_connection_expression
                )

                body.parent_connection.parent_T_connection_expression = (
                    parent_T_old_origin @ old_origin_T_new_origin
                )

            for child in world.compute_child_kinematic_structure_entities(body):
                old_origin_T_child_origin = (
                    child.parent_connection.parent_T_connection_expression
                )
                child.parent_connection.parent_T_connection_expression = (
                    old_origin_T_new_origin.inverse() @ old_origin_T_child_origin
                )
        return world


@dataclass
class BodyFactoryReplace(Step):
    """
    Replace bodies in the world that match a given condition with new structures created by a factory.
    """

    annotation_creator: Callable[[Body, World], HasRootKinematicStructureEntity]
    """
    A callable that takes a Body, and creates a new semantic annotation (including a new body) for it, and adds them to the world.
    """

    body_condition: Callable[[Body], bool] = lambda x: bool(
        re.compile(r"^dresser_\d+.*$").fullmatch(x.name.name)
    )
    """
    Condition to filter bodies that should be replaced. Defaults to matching bodies containing "dresser_" followed by digits in their name.
    """

    def _apply(self, world: World) -> World:
        filtered_bodies = [body for body in world.bodies if self.body_condition(body)]

        for body in filtered_bodies:
            semantic_annotation = self.annotation_creator(body, world)
            new_world = world.move_branch_to_new_world(semantic_annotation.root)
            parent_connection = body.parent_connection
            if parent_connection is None:
                return new_world

            for entity in world.get_kinematic_structure_entities_of_branch(body):
                world.remove_kinematic_structure_entity(entity)

            world.remove_kinematic_structure_entity(body)

            parent_connection.child = new_world.root
            world.merge_world(new_world, parent_connection)

        return world

from __future__ import absolute_import, annotations

from collections import OrderedDict
from functools import lru_cache
from typing import Dict, Tuple, TYPE_CHECKING
from uuid import UUID

import numpy as np
import rustworkx.visit

from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    Matrix,
    VariableParameters,
)
from ..datastructures.types import NpMatrix4x4
from ..spatial_types import HomogeneousTransformationMatrix
from ..spatial_types.math import inverse_frame
from ..utils import copy_lru_cache
from ..world_description.world_entity import Connection, KinematicStructureEntity

if TYPE_CHECKING:
    from ..world import World


class ForwardKinematicsManager(rustworkx.visit.DFSVisitor):
    """
    Visitor class for collection various forward kinematics expressions in a world model.

    This class is designed to traverse a world, compute the forward kinematics transformations in batches for different
    use cases.
    1. Efficient computation of forward kinematics between any bodies in the world.
    2. Efficient computation of forward kinematics for all bodies with collisions for updating collision checkers.
    3. Efficient computation of forward kinematics as position and quaternion, useful for ROS tf.
    """

    compiled_collision_fks: CompiledFunction
    compiled_all_fks: CompiledFunction

    forward_kinematics_for_all_bodies: np.ndarray
    """
    A 2D array containing the stacked forward kinematics expressions for all bodies in the world.
    Dimensions are ((number of bodies) * 4) x 4.
    They are computed in batch for efficiency.
    """
    body_id_to_forward_kinematics_idx: Dict[UUID, int]
    """
    Given a body id, returns the index of the first row in `forward_kinematics_for_all_bodies` that corresponds to that body.
    """

    def __init__(self, world: World):
        self.world = world
        self.child_body_to_fk_expr: Dict[UUID, HomogeneousTransformationMatrix] = {
            self.world.root.id: HomogeneousTransformationMatrix()
        }
        self.tf: Dict[Tuple[UUID, UUID], Matrix] = OrderedDict()

    def recompile(self):
        self.child_body_to_fk_expr: Dict[UUID, HomogeneousTransformationMatrix] = {
            self.world.root.id: HomogeneousTransformationMatrix()
        }
        self.tf: Dict[Tuple[UUID, UUID], Matrix] = OrderedDict()
        self.world._travel_branch(self.world.root, self)
        self.compile()

    def connection_call(self, edge: Tuple[int, int, Connection]):
        """
        Gathers forward kinematics expressions for a connection.
        """
        connection = edge[2]
        map_T_parent = self.child_body_to_fk_expr[connection.parent.id]
        self.child_body_to_fk_expr[connection.child.id] = map_T_parent.dot(
            connection.origin_expression
        )
        self.tf[(connection.parent.id, connection.child.id)] = (
            connection.origin_as_position_quaternion()
        )

    tree_edge = connection_call

    def compile(self) -> None:
        """
        Compiles forward kinematics expressions for fast evaluation.
        """
        all_fks = Matrix.vstack(
            [
                self.child_body_to_fk_expr[body.id]
                for body in self.world.kinematic_structure_entities
            ]
        )
        tf = Matrix.vstack([pose for pose in self.tf.values()])
        collision_fks = []
        for body in sorted(
            self.world.bodies_with_enabled_collision, key=lambda b: b.id
        ):
            if body == self.world.root:
                continue
            collision_fks.append(self.child_body_to_fk_expr[body.id])
        collision_fks = Matrix.vstack(collision_fks)
        params = [v.variables.position for v in self.world.degrees_of_freedom]
        self.compiled_all_fks = all_fks.compile(
            parameters=VariableParameters.from_lists(params)
        )
        self.compiled_collision_fks = collision_fks.compile(
            parameters=VariableParameters.from_lists(params)
        )
        self.compiled_tf = tf.compile(parameters=VariableParameters.from_lists(params))
        self.idx_start = {
            body.id: i * 4
            for i, body in enumerate(self.world.kinematic_structure_entities)
        }

    def recompute(self) -> None:
        """
        Clears cache and recomputes all forward kinematics. Should be called after a state update.
        """
        self.compute_np.cache_clear()
        self.subs = self.world.state.positions
        self.forward_kinematics_for_all_bodies = self.compiled_all_fks(self.subs)
        self.collision_fks = self.compiled_collision_fks(self.subs)

    def compute_tf(self) -> np.ndarray:
        """
        Computes a (number of bodies) x 7 matrix of forward kinematics in position/quaternion format.
        The rows are ordered by body id.
        The first 3 entries are position values, the last 4 entires are quaternion values in x, y, z, w order.

        This is not updated in 'recompute', because this functionality is only used with ROS.
        :return: A large matrix with all forward kinematics.
        """
        return self.compiled_tf(self.subs)

    @copy_lru_cache()
    def compose_expression(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> HomogeneousTransformationMatrix:
        """
        :param root: The root KinematicStructureEntity in the kinematic chain.
            It determines the starting point of the forward kinematics calculation.
        :param tip: The tip KinematicStructureEntity in the kinematic chain.
            It determines the endpoint of the forward kinematics calculation.
        :return: An expression representing the computed forward kinematics of the tip KinematicStructureEntity relative to the root KinematicStructureEntity.
        """

        fk = HomogeneousTransformationMatrix()
        root_chain, tip_chain = self.world.compute_split_chain_of_connections(root, tip)
        connection: Connection
        for connection in root_chain:
            tip_T_root = connection.origin_expression.inverse()
            fk = fk.dot(tip_T_root)
        for connection in tip_chain:
            fk = fk.dot(connection.origin_expression)
        fk.reference_frame = root
        fk.child_frame = tip
        return fk

    def compute(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> HomogeneousTransformationMatrix:
        """
        Compute the forward kinematics from the root KinematicStructureEntity to the tip KinematicStructureEntity.

        Calculate the transformation matrix representing the pose of the
        tip KinematicStructureEntity relative to the root KinematicStructureEntity.

        :param root: Root KinematicStructureEntity for which the kinematics are computed.
        :param tip: Tip KinematicStructureEntity to which the kinematics are computed.
        :return: Transformation matrix representing the relative pose of the tip KinematicStructureEntity with respect to the root KinematicStructureEntity.
        """
        return HomogeneousTransformationMatrix(
            data=self.compute_np(root, tip), reference_frame=root
        )

    @lru_cache(maxsize=None)
    def compute_np(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> NpMatrix4x4:
        """
        Computes the forward kinematics from the root body to the tip body, root_T_tip.

        This method computes the transformation matrix representing the pose of the
        tip body relative to the root body, expressed as a numpy ndarray.

        :param root: Root body for which the kinematics are computed.
        :param tip: Tip body to which the kinematics are computed.
        :return: Transformation matrix representing the relative pose of the tip body with respect to the root body.
        """
        root_hash = root.id
        tip = tip.id
        root_is_world = root_hash == self.world.root.id
        tip_is_world = tip == self.world.root.id

        if not tip_is_world:
            i = self.idx_start[tip]
            map_T_tip = self.forward_kinematics_for_all_bodies[i : i + 4]
            if root_is_world:
                return map_T_tip

        if not root_is_world:
            i = self.idx_start[root_hash]
            map_T_root = self.forward_kinematics_for_all_bodies[i : i + 4]
            root_T_map = inverse_frame(map_T_root)
            if tip_is_world:
                return root_T_map

        if tip_is_world and root_is_world:
            return np.eye(4)

        return root_T_map @ map_T_tip

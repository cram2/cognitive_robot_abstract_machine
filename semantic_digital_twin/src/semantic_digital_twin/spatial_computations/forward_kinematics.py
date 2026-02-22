from __future__ import absolute_import, annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, TYPE_CHECKING
from uuid import UUID

import numpy as np
import rustworkx.visit
from typing_extensions import List

from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    Matrix,
    VariableParameters,
    FloatVariable,
)
from ..callbacks.callback import ModelChangeCallback
from ..datastructures.types import NpMatrix4x4
from ..spatial_types import HomogeneousTransformationMatrix
from ..spatial_types.math import inverse_frame
from ..utils import copy_lru_cache
from ..world_description.world_entity import Connection, KinematicStructureEntity

if TYPE_CHECKING:
    from ..world import World


@dataclass
class ForwardKinematicsManager(ModelChangeCallback):
    """
    Visitor class for collection various forward kinematics expressions in a world model.

    This class is designed to traverse a world, compute the forward kinematics transformations in batches for different
    use cases.
    1. Efficient computation of forward kinematics between any bodies in the world.
    2. Efficient computation of forward kinematics for all bodies with collisions for updating collision checkers.
    3. Efficient computation of forward kinematics as position and quaternion, useful for ROS tf.
    """

    compiled_all_fks: CompiledFunction = field(init=False, repr=False)

    forward_kinematics_for_all_bodies: np.ndarray = field(init=False, repr=False)
    """
    A 2D array containing the stacked forward kinematics expressions for all bodies in the world.
    Dimensions are ((number of bodies) * 4) x 4.
    They are computed in batch for efficiency.
    """
    body_id_to_forward_kinematics_idx: Dict[UUID, int] = field(init=False, repr=False)
    """
    Given a body id, returns the index of the first row in `forward_kinematics_for_all_bodies` that corresponds to that body.
    """

    root_T_kse_expression_cache: Dict[UUID, HomogeneousTransformationMatrix] = field(
        init=False, repr=False
    )

    body_id_to_all_fk_index: Dict[UUID, int] = field(init=False, repr=False)

    def _notify(self, **kwargs):
        if len(self._world.kinematic_structure_entities) == 0:
            return
        self.update_root_T_kse_expression_cache()
        self.compose_expression.cache_clear()
        self.compile()
        self.recompute()  # we need to recompute because other model updaters might need fk.

    def __hash__(self):
        return hash(id(self))

    def update_root_T_kse_expression_cache(self):
        self.root_T_kse_expression_cache = {
            self._world.root.id: HomogeneousTransformationMatrix()
        }
        for parent, childs in rustworkx.bfs_successors(
            self._world.kinematic_structure, self._world.root.index
        ):
            root_T_parent = self.root_T_kse_expression_cache[parent.id]
            for child in childs:
                parent_C_child = self._world.get_connection(parent, child)
                self.root_T_kse_expression_cache[child.id] = (
                    root_T_parent @ parent_C_child.origin_expression
                )

    def compile(self) -> None:
        """
        Compiles forward kinematics expressions for fast evaluation.
        """
        all_fks = Matrix.vstack(
            [
                self.root_T_kse_expression_cache[body.id]
                for body in self._world.kinematic_structure_entities
            ]
        )

        self.compiled_all_fks = all_fks.compile(
            parameters=VariableParameters.from_lists(
                self._world.state.position_float_variables
            )
        )
        self.compiled_all_fks.bind_args_to_memory_view(0, self._world.state.positions)
        self.body_id_to_all_fk_index = {
            body.id: i * 4
            for i, body in enumerate(self._world.kinematic_structure_entities)
        }

    def recompute(self) -> None:
        """
        Clears cache and recomputes all forward kinematics. Should be called after a state update.
        """
        self.compute_np.cache_clear()
        self.forward_kinematics_for_all_bodies = self.compiled_all_fks.evaluate()

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
        if root == self._world.root:
            return self.root_T_kse_expression_cache[tip.id]
        fk = HomogeneousTransformationMatrix()
        root_chain, tip_chain = self._world.compute_split_chain_of_connections(
            root, tip
        )
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
        root = root.id
        tip = tip.id
        root_is_world = root == self._world.root.id
        tip_is_world = tip == self._world.root.id

        if not tip_is_world:
            i = self.body_id_to_all_fk_index[tip]
            map_T_tip = self.forward_kinematics_for_all_bodies[i : i + 4]
            if root_is_world:
                return map_T_tip

        if not root_is_world:
            i = self.body_id_to_all_fk_index[root]
            map_T_root = self.forward_kinematics_for_all_bodies[i : i + 4]
            root_T_map = inverse_frame(map_T_root)
            if tip_is_world:
                return root_T_map

        if tip_is_world and root_is_world:
            return np.eye(4)

        return root_T_map @ map_T_tip

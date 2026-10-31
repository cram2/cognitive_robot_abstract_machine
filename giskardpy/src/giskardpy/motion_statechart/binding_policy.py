from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from krrood.symbolic_math.float_variable_data import FloatVariableData
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


class GoalBindingPolicy(Enum):
    """
    This policy should be used together with ForwardKinematicsBinding.
    """

    Bind_at_build = 1
    """Forward kinematics is only computed once at build time."""
    Bind_on_start = 2
    """Forward kinematics is computed during on_start of the MotionStatechartNode."""


@dataclass
class ForwardKinematicsBinding:
    """
    Fixes the forward kinematics of the chain between root and tip to the state they
    had when :meth:`bind` was last called, as :attr:`root_T_tip`.

    Typically used together with :class:`GoalBindingPolicy`, which decides when
    :meth:`bind` is called.

    .. warning:: Create it in :meth:`~cramph.node.StatechartNode.set_up`, and keep a
        reference to it in the node.
    """

    float_variable_data: FloatVariableData
    """
    The data holding the values of the variables :attr:`root_T_tip` is expressed over.
    """
    name: PrefixedName
    """
    The name the variables of :attr:`root_T_tip` are created under.

    .. warning:: Make it unique, for example by deriving it from the name of the node
        holding the binding.
    """
    root: KinematicStructureEntity
    """Root of the kinematic chain."""
    tip: KinematicStructureEntity
    """Tip of the kinematic chain."""

    root_T_tip: HomogeneousTransformationMatrix | None = field(
        default=None, kw_only=True
    )
    """
    The forward kinematics from :attr:`root` to :attr:`tip`, expressed over the
    variables :meth:`bind` sets; created over new variables named after :attr:`name`
    if left unset.
    """

    _bound_root_T_tip: NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    """
    The forward kinematics :attr:`root_T_tip` was last set to, None until then.
    """

    def __post_init__(self) -> None:
        if self.root_T_tip is None:
            self.root_T_tip = HomogeneousTransformationMatrix.create_with_variables(
                str(self.name)
            )
        self.root_T_tip.reference_frame = self.root
        self.root_T_tip.child_frame = self.tip
        self.float_variable_data.register_expression(self.root_T_tip)

    def change_root(self, root: KinematicStructureEntity, world: World) -> None:
        """
        Measure the chain from `root`, such as after the branch holding :attr:`tip` was
        moved under another parent, keeping the pose of :attr:`tip` that was bound last.

        :param root: The new root of the chain.
        :param world: The world used for computing where `root` is now.
        """
        new_root_T_root = world.compute_forward_kinematics_np(root, self.root)
        self.root = root
        self.root_T_tip.reference_frame = root
        self._set_root_T_tip(new_root_T_root @ self._bound_root_T_tip)

    def bind(self, world: World) -> None:
        """
        Set :attr:`root_T_tip` to the current forward kinematics of the chain.

        :param world: The world used for computing the forward kinematics.
        """
        self._set_root_T_tip(world.compute_forward_kinematics_np(self.root, self.tip))

    def _set_root_T_tip(self, root_T_tip: NDArray[np.float64]) -> None:
        """
        Set :attr:`root_T_tip` to `root_T_tip`.

        :param root_T_tip: A homogeneous transformation matrix.
        """
        self._bound_root_T_tip = root_T_tip
        self.float_variable_data.set_value(
            self.root_T_tip, root_T_tip[:3, :4].T.flatten()
        )

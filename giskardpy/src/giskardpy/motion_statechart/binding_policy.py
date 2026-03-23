from dataclasses import dataclass, field, InitVar
from enum import Enum

from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import SymbolicMathType
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from giskardpy.motion_statechart.context import MotionStatechartContext


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
    Binds the forward kinematics of the chain between root and tip to its current state.
    This class is useful if you need to update a TransformationMatrix representing forward kinematics during execution
    of a MotionStatechartNode.
    It creates the TransformationMatrix root_T_tip, which is fixed to the current state.
    Call bind() to update its state to its current state.
    Typically used together with GoalBindingPolicy, where bind() is called depending on the chosen policy.
    ..warning:: Must be created during build() of a MotionStatechartNode.
    ..warning:: Ensure to keep a reference to this instance in the MotionStatechartNode.
    """

    float_variable_data: FloatVariableData
    """
    Reference to the FloatVariableData used for storing variables for the forward kinematics.
    """
    name: PrefixedName
    """
    Name of the Binding. It is used for naming the auxiliary variables.
    ..warning:: ensure to generate a unique name, e.g., using the name of the MotionStatechartNode.
    """
    root: KinematicStructureEntity
    """Root of the kinematic chain."""
    tip: KinematicStructureEntity
    """Tip of the kinematic chain."""

    _root_T_tip_expr: HomogeneousTransformationMatrix | None = field(
        default=None, init=False
    )
    """The TransformationMatrix root_T_tip, represented using auxiliary variables."""

    def __post_init__(self):
        self._root_T_tip_expr = HomogeneousTransformationMatrix.create_with_variables(
            str(self.name)
        )
        self._root_T_tip_expr.reference_frame = self.root
        self._root_T_tip_expr.child_frame = self.tip
        self._matrix_index = self.float_variable_data.register_expression(
            self._root_T_tip_expr
        )

    @property
    def root_T_tip(self):
        return self._root_T_tip_expr

    def bind(self, world: World):
        """
        Will update root_T_tip to the current state of the kinematic chain.
        Call during on_start() etc. of a MotionStatechartNode.
        :param world: The world used for computing the forward kinematics.
        """
        root_T_tip = world.compute_forward_kinematics_np(self.root, self.tip)
        root_T_tip_flat = root_T_tip[:3, :4].T.flatten()
        self.float_variable_data.set_value(self._root_T_tip_expr, root_T_tip_flat)

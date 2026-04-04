from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
from typing_extensions import (
    Self,
    TYPE_CHECKING,
    Generic,
    TypeVar,
    get_args,
    assert_never,
)

from giskardpy.qp.qp_data import QPData, QPDataExplicit, QPDataTwoSidedInequality
from krrood.symbolic_math.symbolic_math import (
    CompiledFunctionWithViews,
    VariableParameters,
    Matrix,
    hstack,
    CompiledFunction,
    FloatVariable,
    vstack,
)

if TYPE_CHECKING:
    from giskardpy.qp.adapters.qp_adapter import QPDataSymbolic


T = TypeVar("T", bound=QPData)


@dataclass
class QPDataFactory(Generic[T], ABC):
    """
    A factory that creates a specific QPData format objects from QPDataSymbolic objects.
    """

    qp_data: QPDataSymbolic
    """
    Symbolic representation of the QP problem.
    """

    @classmethod
    @property
    def qp_data_type(cls) -> type[T]:
        """
        The semDT type for which this converter handles conversion.
        """
        return get_args(cls.__orig_bases__[0])[0]

    @classmethod
    def get_factory_from_qp_data_type(
        cls, qp_data_type: type[QPData]
    ) -> type[QPDataFactory]:
        """
        Returns the factory that handles conversion for the given QPData type.
        """
        for subclass in cls.__subclasses__():
            if subclass.qp_data_type == qp_data_type:
                return subclass
        assert_never(qp_data_type)

    @abstractmethod
    def compile(
        self,
        world_state_symbols: list[FloatVariable],
        life_cycle_symbols: list[FloatVariable],
        float_variables: list[FloatVariable],
    ):
        """
        Transforms the symbolic representation of the QP problem into functions that compute the parts of QPData.
        :param world_state_symbols: list of variables representing the world state.
        :param life_cycle_symbols: list of variables representing the life cycle states of a motion statechart.
        :param float_variables: list of variables representing the additional float variables not covered by the world state and life cycle states.
        """

    @abstractmethod
    def evaluate(
        self,
        world_state: np.ndarray,
        life_cycle_state: np.ndarray,
        float_variables: np.ndarray,
    ) -> QPData:
        """
        Evaluates the QP problem represented by this factory using the given world state, life cycle state, and float variables.
        .. note:: it is assumed that the input data corresponds to the input giving to `compile`.
        :param world_state: Current state of the world.
        :param life_cycle_state: Current state of the motion statechart.
        :param float_variables: Additional float variables not covered by the world state and life cycle states.
        :return: Explicit representation of the QP problem with numerical values.
        """

    def __hash__(self):
        return hash(id(self))


@dataclass
class QPDataExplicitFactory(QPDataFactory[QPDataExplicit]):
    """
    A factory that creates QPDataExplicit objects from QPDataSymbolic objects.
    """

    equality_matrix_compiled: CompiledFunction = field(init=False)
    """
    Compiled casadi function computing the equality constraint matrix.
    """
    inequality_matrix_compiled: CompiledFunction = field(init=False)
    """
    Compiled casadi function computing the inequality constraint matrix.
    """
    combined_vector_f: CompiledFunctionWithViews = field(init=False)
    """
    Compiled casadi function computing all vector elements of the QP problem.
    """

    def compile(
        self,
        world_state_symbols: list[FloatVariable],
        life_cycle_symbols: list[FloatVariable],
        float_variables: list[FloatVariable],
    ):
        eq_matrix = hstack(
            [
                self.qp_data.eq_matrix_dofs,
                self.qp_data.eq_matrix_slack,
                Matrix.zeros(
                    self.qp_data.eq_matrix_slack.shape[0],
                    self.qp_data.num_neq_slack_variables,
                ),
            ]
        )
        neq_matrix = hstack(
            [
                self.qp_data.neq_matrix_dofs,
                Matrix.zeros(
                    self.qp_data.neq_matrix_slack.shape[0],
                    self.qp_data.num_eq_slack_variables,
                ),
                self.qp_data.neq_matrix_slack,
            ]
        )
        free_symbols = [
            world_state_symbols,
            life_cycle_symbols,
            float_variables,
        ]

        self.equality_matrix_compiled = eq_matrix.compile(
            parameters=VariableParameters.from_lists(*free_symbols),
            sparse=True,
        )
        self.inequality_matrix_compiled = neq_matrix.compile(
            parameters=VariableParameters.from_lists(*free_symbols),
            sparse=True,
        )

        self.combined_vector_f = CompiledFunctionWithViews(
            expressions=[
                self.qp_data.quadratic_weights,
                self.qp_data.linear_weights,
                self.qp_data.box_lower_constraints,
                self.qp_data.box_upper_constraints,
                self.qp_data.eq_bounds,
                self.qp_data.neq_lower_bounds,
                self.qp_data.neq_upper_bounds,
            ],
            parameters=VariableParameters.from_lists(*free_symbols),
        )

    def evaluate(
        self,
        world_state: np.ndarray,
        life_cycle_state: np.ndarray,
        float_variables: np.ndarray,
    ) -> QPDataExplicit:
        args = [
            world_state,
            life_cycle_state,
            float_variables,
        ]
        eq_matrix_np_raw = self.equality_matrix_compiled(*args)
        neq_matrix_np_raw = self.inequality_matrix_compiled(*args)
        (
            quadratic_weights_np_raw,
            linear_weights_np_raw,
            box_lower_constraints_np_raw,
            box_upper_constraints_np_raw,
            eq_bounds_np_raw,
            neq_lower_bounds_np_raw,
            neq_upper_bounds_np_raw,
        ) = self.combined_vector_f(*args)

        return QPDataExplicit(
            quadratic_weights=quadratic_weights_np_raw,
            linear_weights=linear_weights_np_raw,
            box_lower_constraints=box_lower_constraints_np_raw,
            box_upper_constraints=box_upper_constraints_np_raw,
            equality_matrix=eq_matrix_np_raw,
            equality_bounds=eq_bounds_np_raw,
            inequality_matrix=neq_matrix_np_raw,
            inequality_lower_bounds=neq_lower_bounds_np_raw,
            inequality_upper_bounds=neq_upper_bounds_np_raw,
            num_equality_slack_variables=self.qp_data.num_eq_slack_variables,
            num_inequality_slack_variables=self.qp_data.num_neq_slack_variables,
        )


@dataclass
class QPDataTwoSidedInequalityFactory(QPDataFactory[QPDataTwoSidedInequality]):

    equality_matrix_compiled: CompiledFunction = field(init=False)
    inequality_matrix_compiled: CompiledFunction = field(init=False)
    combined_vector_f: CompiledFunctionWithViews = field(init=False)

    def compile(
        self,
        world_state_symbols: list[FloatVariable],
        life_cycle_symbols: list[FloatVariable],
        float_variables: list[FloatVariable],
    ):
        if len(self.qp_data.neq_matrix_dofs) == 0:
            constraint_matrix = hstack(
                [self.qp_data.eq_matrix_dofs, self.qp_data.eq_matrix_slack]
            )
        else:
            eq_matrix = hstack(
                [
                    self.qp_data.eq_matrix_dofs,
                    self.qp_data.eq_matrix_slack,
                    Matrix.zeros(
                        self.qp_data.eq_matrix_dofs.shape[0],
                        self.qp_data.neq_matrix_slack.shape[1],
                    ),
                ]
            )
            neq_matrix = hstack(
                [
                    self.qp_data.neq_matrix_dofs,
                    Matrix.zeros(
                        self.qp_data.neq_matrix_dofs.shape[0],
                        self.qp_data.eq_matrix_slack.shape[1],
                    ),
                    self.qp_data.neq_matrix_slack,
                ]
            )
            constraint_matrix = vstack([eq_matrix, neq_matrix])

        free_symbols = [
            world_state_symbols,
            life_cycle_symbols,
            float_variables,
        ]

        len_lb_be_lba_end = (
            self.qp_data.quadratic_weights.shape[0]
            + self.qp_data.box_lower_constraints.shape[0]
            + self.qp_data.eq_bounds.shape[0]
            + self.qp_data.neq_lower_bounds.shape[0]
        )
        len_ub_be_uba_end = (
            len_lb_be_lba_end
            + self.qp_data.box_upper_constraints.shape[0]
            + self.qp_data.eq_bounds.shape[0]
            + self.qp_data.neq_upper_bounds.shape[0]
        )

        self.combined_vector_f = CompiledFunctionWithViews(
            expressions=[
                self.qp_data.quadratic_weights,
                self.qp_data.box_lower_constraints,
                self.qp_data.eq_bounds,
                self.qp_data.neq_lower_bounds,
                self.qp_data.box_upper_constraints,
                self.qp_data.eq_bounds,
                self.qp_data.neq_upper_bounds,
                self.qp_data.linear_weights,
            ],
            parameters=VariableParameters.from_lists(*free_symbols),
            additional_views=[
                slice(self.qp_data.quadratic_weights.shape[0], len_lb_be_lba_end),
                slice(len_lb_be_lba_end, len_ub_be_uba_end),
            ],
        )

        self.inequality_matrix_compiled = constraint_matrix.compile(
            parameters=VariableParameters.from_lists(*free_symbols),
            sparse=True,
        )

    def evaluate(
        self,
        world_state: np.ndarray,
        life_cycle_state: np.ndarray,
        float_variables: np.ndarray,
    ) -> QPDataTwoSidedInequality:
        args = [
            world_state,
            life_cycle_state,
            float_variables,
        ]
        neq_matrix = self.inequality_matrix_compiled(*args)
        (
            quadratic_weights_np_raw,
            box_lower_constraints_np_raw,
            _,
            _,
            box_upper_constraints_np_raw,
            _,
            _,
            linear_weights_np_raw,
            box_eq_neq_lower_bounds_np_raw,
            box_eq_neq_upper_bounds_np_raw,
        ) = self.combined_vector_f(*args)
        self.qp_data_raw = QPDataTwoSidedInequality(
            quadratic_weights=quadratic_weights_np_raw,
            linear_weights=linear_weights_np_raw,
            inequality_matrix=neq_matrix,
            inequality_lower_bounds=box_eq_neq_lower_bounds_np_raw,
            inequality_upper_bounds=box_eq_neq_upper_bounds_np_raw,
            num_equality_slack_variables=self.qp_data.num_eq_slack_variables,
            num_inequality_slack_variables=self.qp_data.num_neq_slack_variables,
        )

        return self.qp_data_raw

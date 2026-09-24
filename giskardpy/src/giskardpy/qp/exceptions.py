"""
Exceptions raised while building and solving the quadratic program.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Type

from giskardpy.data_types.exceptions import (
    GiskardException,
    DontPrintStackTrace,
    SetupException,
)

if TYPE_CHECKING:
    from giskardpy.qp.constraint import GiskardConstraint
    from giskardpy.qp.qp_data import QPData


@dataclass
class QPSolverException(GiskardException):
    """
    Base class for errors raised by the QP solvers.
    """


@dataclass
class SolverReturnedFailureError(QPSolverException):
    """
    Raised when a QP solver returns a non-optimal status.
    """

    solver_status: str
    """
    The solver-specific status describing the failure.
    """

    def error_message(self) -> str:
        return f"QP solver failed with status: {self.solver_status}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InfeasibleException(QPSolverException):
    """
    Raised when the QP has no feasible solution.
    """

    solver_status: str = ""
    """
    The solver-specific status describing the infeasibility.
    """

    def error_message(self) -> str:
        return f"QP is infeasible. Solver status: {self.solver_status}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class QuadraticObjectiveUnsupportedError(QPSolverException):
    """
    Raised when a linear program solver is given a problem with a quadratic objective.
    """

    solver_name: str
    """
    The name of the solver that only accepts linear objectives.
    """

    def error_message(self) -> str:
        return f"{self.solver_name} only solves linear programs, but the problem has non-zero quadratic weights."

    def suggest_correction(self) -> str:
        return "Use a quadratic program solver for this problem."


@dataclass
class BrakingExceedsHorizonError(SetupException):
    """
    Raised when a braking from the velocity limit to rest does not fit into the
    prediction horizon, which every plan must end at rest within.
    """

    prediction_horizon: int
    """
    The prediction horizon of the QP controller.
    """

    minimum_prediction_horizon: int
    """
    The shortest prediction horizon the braking fits into.
    """

    def suggest_correction(self) -> str:
        return (
            f"Raise prediction_horizon to at least {self.minimum_prediction_horizon}, "
            f"or leave it unset so it is derived from braking_time."
        )


@dataclass
class BrakingTimeExceedsHorizonError(BrakingExceedsHorizonError):
    """
    Raised when the configured braking time needs more steps than the explicitly set
    prediction horizon provides.
    """

    braking_time: float
    """
    The configured braking time, in seconds.
    """

    time_step: float
    """
    The duration of one step of the prediction horizon.
    """

    def error_message(self) -> str:
        return (
            f"A braking time of {self.braking_time} s at a time step of {self.time_step} s "
            f"needs a prediction horizon of at least {self.minimum_prediction_horizon}, "
            f"but it is {self.prediction_horizon}."
        )


@dataclass
class DegreeOfFreedomBrakingExceedsHorizonError(BrakingExceedsHorizonError):
    """
    Raised when the jerk limit of a degree of freedom is too low to brake from its
    velocity limit to rest within the prediction horizon.
    """

    degree_of_freedom_name: str
    """
    The name of the degree of freedom whose braking does not fit.
    """

    velocity_limit: float
    """
    The velocity limit the degree of freedom brakes from.
    """

    jerk_limit: float
    """
    The jerk limit the degree of freedom brakes with.
    """

    def error_message(self) -> str:
        return (
            f'Degree of freedom "{self.degree_of_freedom_name}" cannot brake from its '
            f"velocity limit {self.velocity_limit} with jerk limit {self.jerk_limit} "
            f"within a prediction horizon of {self.prediction_horizon}; it needs at "
            f"least {self.minimum_prediction_horizon}."
        )

    def suggest_correction(self) -> str:
        return (
            f"Set prediction_horizon to at least {self.minimum_prediction_horizon}, or "
            f"raise the jerk or acceleration limit of the degree of freedom."
        )


@dataclass
class EmptyProblemException(InfeasibleException, DontPrintStackTrace):
    """
    Raised when the QP problem has no free variables.
    """

    def error_message(self) -> str:
        return "Empty QP problem."


@dataclass
class MismatchedLimitLengthsError(GiskardException):
    """
    Raised when the bounds, weights, and names of a DirectLimits do not all share the
    same length.
    """

    field_lengths: dict[str, int]
    """
    The length of each DirectLimits field, keyed by field name.
    """

    def error_message(self) -> str:
        return f"All DirectLimits fields must have the same length, got {self.field_lengths}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ConstraintTypeMismatchError(QPSolverException):
    """
    Raised when an enforcement strategy receives a constraint of the wrong type for the
    requested bounds.
    """

    strategy_name: str
    """
    The name of the enforcement strategy that received the constraint.
    """

    expected_type: Type[GiskardConstraint]
    """
    The constraint type the strategy expected.
    """

    actual_type: Type[GiskardConstraint]
    """
    The constraint type that was actually received.
    """

    constraint_name: str
    """
    The name of the offending constraint.
    """

    def error_message(self) -> str:
        return (
            f"{self.strategy_name} expected constraints of type {self.expected_type.__name__}, "
            f"but got {self.actual_type.__name__} for constraint {self.constraint_name!r}."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NoFactoryForQPDataTypeError(QPSolverException):
    """
    Raised when no registered factory handles the requested QPData type.
    """

    qp_data_type: Type[QPData]
    """
    The QPData type for which no factory is registered.
    """

    def error_message(self) -> str:
        return (
            f"No QPDataFactory registered for QPData type {self.qp_data_type.__name__}."
        )

    def suggest_correction(self) -> str:
        return ""

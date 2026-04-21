from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from krrood.utils import DataclassException
from random_events.variable import Variable


@dataclass
class SupportDeterminismViolation(DataclassException):
    """
    Base class for all violations produced by verify_support_determinism().

    Inherits from DataclassException so each violation is also a raiseable
    exception. Subclasses set self.message in __post_init__ before calling
    super().__post_init__(). String representation is handled by
    DataclassException via Exception.__str__, which returns self.message
    through the args tuple set in __post_init__.
    """


@dataclass
class MissingQueryVariableViolation(SupportDeterminismViolation):
    """
    Violation raised when a Variable declared in a query_set is absent from the circuit.

    Produced by Check 1 of verify_support_determinism().
    """

    missing_variables: List[Variable]
    """Variables present in the query_set but not in the circuit."""

    available_variables: List[Variable]
    """All Variables currently registered in the circuit."""

    def __post_init__(self) -> None:
        missing = [v.name for v in self.missing_variables]
        available = [v.name for v in self.available_variables]
        self.message = (
            f"Query-set Variables {missing} not found in circuit. "
            f"Available: {available}"
        )
        super().__post_init__()


@dataclass
class UnnormalizedSumUnitViolation(SupportDeterminismViolation):
    """
    Violation raised when a SumUnit's log-weights do not sum to log(1).

    Produced by Check 2 of verify_support_determinism().
    Unnormalized SumUnits produce incorrect backdoor adjustment probabilities.
    """

    sum_unit_index: int
    """Graph index of the offending SumUnit."""

    actual_log_weight_sum: float
    """The actual sum of log-weights, which should be 0.0."""

    def __post_init__(self) -> None:
        self.message = (
            f"SumUnit (index={self.sum_unit_index}) log-weights sum to "
            f"{self.actual_log_weight_sum:.6f}, expected 0.0. "
            f"Unnormalized circuits produce incorrect backdoor probabilities."
        )
        super().__post_init__()


@dataclass
class OverlappingChildSupportsViolation(SupportDeterminismViolation):
    """
    Violation raised when a SumUnit's children have overlapping marginal support
    on a declared query Variable.

    Produced by Check 3 of verify_support_determinism().
    Overlapping supports violate the support-determinism property required for
    tractable backdoor adjustment.
    """

    sum_unit_index: int
    """Graph index of the offending SumUnit."""

    query_variable: Variable
    """The declared query Variable on which the overlap was detected."""

    def __post_init__(self) -> None:
        self.message = (
            f"SumUnit (index={self.sum_unit_index}) has overlapping children supports "
            f"on declared query Variable '{self.query_variable.name}': children are not "
            f"support-deterministic for this Variable."
        )
        super().__post_init__()


@dataclass
class SupportDeterminismVerificationResult(DataclassException):
    """
    Result of verifying support determinism of a circuit against its
    Marginal Determinism Variable Tree.

    Support determinism requires that for each declared cause Variable,
    SumUnit children have disjoint support regions — i.e. each child
    exclusively owns a non-overlapping partition of that Variable's domain.
    This structural property is required for tractable backdoor adjustment.

    Based on the Q-determinism condition from:
        Broadrick et al. (2023), Tractable Probabilistic Circuits
        https://arxiv.org/abs/2304.07438
    """

    passed: bool
    """True if all three checks passed with no violations."""

    violations: List[SupportDeterminismViolation]
    """Typed violations found, in check order. Empty when passed is True."""

    checked_query_sets: List[Set[Variable]]
    """All non-empty query_sets from the Marginal Determinism Variable Tree."""

    circuit_variables: List[Variable]
    """All Variables present in the circuit at verification time."""

    def __post_init__(self) -> None:
        self.message = str(self)
        super().__post_init__()

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        checked_names = [{v.name for v in qs} for qs in self.checked_query_sets]
        circuit_names = [v.name for v in self.circuit_variables]
        lines = [
            f"Support determinism verification: {status}",
            f"  Checked query_sets: {checked_names}",
            f"  Circuit variables:  {circuit_names}",
        ]
        if self.violations:
            lines.append("  Violations:")
            for violation in self.violations:
                lines.append(f"    - {str(violation)}")
        return "\n".join(lines)


@dataclass
class UnregisteredVariableError(DataclassException):
    """
    Raised when a Variable passed to backdoor_adjustment or diagnose_failure
    is not registered as a cause or effect Variable in the CausalCircuit.
    """

    variable_name: str
    """Name of the unregistered Variable."""

    registered_names: List[str]
    """Names of all Variables registered in the relevant role."""

    role: str
    """Either 'cause' or 'effect', describing which role the Variable was expected in."""

    def __post_init__(self) -> None:
        self.message = (
            f"'{self.variable_name}' is not a registered {self.role} Variable. "
            f"Registered: {self.registered_names}."
        )
        super().__post_init__()


@dataclass
class EmptyInterventionalCircuitError(DataclassException):
    """
    Raised when backdoor_adjustment produces no ProductUnit components,
    meaning the interventional circuit is empty. This indicates the cause
    Variable's observed value lies entirely outside the training distribution,
    or the adjustment variables produce no valid strata.
    """

    cause_variable_name: str
    """Name of the cause Variable for which the circuit is empty."""

    adjustment_variable_names: List[str]
    """Names of adjustment Variables used, empty list if no adjustment was applied."""

    def __post_init__(self) -> None:
        if self.adjustment_variable_names:
            self.message = (
                f"Interventional circuit with adjustment is empty. "
                f"cause='{self.cause_variable_name}', "
                f"adjustment={self.adjustment_variable_names}."
            )
        else:
            self.message = (
                f"Interventional circuit is empty for cause '{self.cause_variable_name}'. "
                f"Ensure the circuit was trained on data covering this Variable's domain."
            )
        super().__post_init__()


@dataclass
class NoCauseVariablesError(DataclassException):
    """
    Raised when diagnose_failure finds no numeric cause Variables in
    observed_values that match any registered cause Variable.
    """

    registered_cause_names: List[str]
    """Names of all cause Variables registered in the CausalCircuit."""

    def __post_init__(self) -> None:
        self.message = (
            f"No cause Variables found in observed_values. "
            f"Expected at least one of: {self.registered_cause_names}."
        )
        super().__post_init__()
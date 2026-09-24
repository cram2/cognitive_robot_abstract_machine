"""
Per-degree-of-freedom decision-variable limits, weights, and the MPC-based velocity
profiles that keep each degree of freedom within its position limits across the
prediction horizon.
"""

from __future__ import annotations

import enum
from copy import copy
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.exceptions import (
    DegreeOfFreedomBrakingExceedsHorizonError,
    MismatchedLimitLengthsError,
)
from giskardpy.qp.jerk_limited_braking import JerkLimitedBraking
from giskardpy.qp.qp_controller_config import (
    NUMBER_OF_RESTING_STEPS,
    QPControllerConfig,
)
from giskardpy.qp.pos_in_vel_limits import BrakingProfile, SlowdownProfile
from krrood.symbolic_math.symbolic_math import Scalar, FloatVariable
from semantic_digital_twin.spatial_types.derivatives import Derivatives, DerivativeMap
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)

NUMBER_OF_JERK_RELAXED_STEPS = 3
"""
Number of initial horizon steps whose jerk limit may be relaxed to keep the position
goal reachable.
"""


@dataclass
class VelocityBoundProfiles:
    """
    Per-step velocity bounds of a degree of freedom across the prediction horizon,
    together with the goal velocity profile and the flag marking when the first step is
    already at rest against a position limit.
    """

    lower_bound: sm.Vector
    """
    Lower velocity bound at each step of the prediction horizon.
    """

    upper_bound: sm.Vector
    """
    Upper velocity bound at each step of the prediction horizon.
    """

    goal_profile: sm.Vector
    """
    Goal velocity profile derived from the lower and upper bounds.
    """

    skip_first: sm.Scalar
    """
    Flag marking that the first step is already at rest against a position limit.
    """

    @classmethod
    def unconstrained(
        cls, velocity_limit: float, prediction_horizon: int
    ) -> VelocityBoundProfiles:
        """
        Creates flat bounds at the velocity limit for a degree of freedom without
        position limits.

        :param velocity_limit: Velocity limit applied at every horizon step.
        :param prediction_horizon: Number of steps in the prediction horizon.
        """
        upper_bound = sm.Vector.ones(prediction_horizon) * velocity_limit
        return cls(
            lower_bound=-upper_bound,
            upper_bound=upper_bound,
            goal_profile=sm.Vector.zeros(prediction_horizon),
            skip_first=sm.Scalar.const_false(),
        )

    def is_violated_by(self, velocity_profile: sm.Vector, epsilon: float) -> sm.Scalar:
        """
        Returns whether a velocity profile leaves these bounds or fails to come to rest
        by the end of the horizon.

        :param velocity_profile: Velocity at each step of the prediction horizon.
        :param epsilon: Tolerance below which a violation is ignored.
        """
        leaves_bounds = sm.logic_or(
            sm.logic_any(velocity_profile < self.lower_bound - epsilon),
            sm.logic_any(velocity_profile > self.upper_bound + epsilon),
        )
        return sm.logic_or(leaves_bounds, sm.abs(velocity_profile[-1]) >= epsilon)


class BoundDirection(enum.Enum):
    """
    Selects whether a velocity bound brakes the degree of freedom against its lower or
    upper position limit.
    """

    LOWER = -1
    """
    Bound that brakes against the lower position limit.
    """

    UPPER = 1
    """
    Bound that brakes against the upper position limit.
    """

    @property
    def sign(self) -> int:
        """
        Returns the sign that mirrors the upper-bound computation onto the lower bound.
        """
        return self.value


@dataclass
class DirectLimits:
    """
    Represents weights and limits of decision variables in a QP.

    All fields must have the same length.
    """

    lower_bounds: sm.Vector
    """
    Lower box limit of each decision variable.
    """

    upper_bounds: sm.Vector
    """
    Upper box limit of each decision variable.
    """

    quadratic_weights: sm.Vector
    """
    Quadratic objective weight of each decision variable.
    """

    linear_weights: sm.Vector
    """
    Linear objective weight of each decision variable.
    """

    names: list[str]
    """
    Human readable name of each decision variable, used for debugging.
    """

    def __post_init__(self):
        """
        Ensures all bound, weight, and name fields describe the same number of decision
        variables.
        """
        lengths = {
            "lower_bounds": self.lower_bounds.shape[0],
            "upper_bounds": self.upper_bounds.shape[0],
            "quadratic_weights": self.quadratic_weights.shape[0],
            "linear_weights": self.linear_weights.shape[0],
            "names": len(self.names),
        }
        if len(set(lengths.values())) > 1:
            raise MismatchedLimitLengthsError(field_lengths=lengths)

    @classmethod
    def empty(cls) -> DirectLimits:
        """
        Creates a DirectLimits without any decision variables.
        """
        return cls(
            lower_bounds=sm.Vector([]),
            upper_bounds=sm.Vector([]),
            quadratic_weights=sm.Vector([]),
            linear_weights=sm.Vector([]),
            names=[],
        )


@dataclass
class DegreeOfFreedomLimitProfiler:
    """
    Computes the per-degree-of-freedom velocity, acceleration, and jerk bounds across
    the prediction horizon, including the MPC-based position-aware slowdown profiles.
    """

    qp_controller_config: QPControllerConfig
    """
    Controller configuration providing horizon length, time step, and solver.
    """

    @property
    def time_step(self) -> float:
        """
        Duration of a single step of the prediction horizon.
        """
        return self.qp_controller_config.model_predictive_control_time_step

    @property
    def prediction_horizon(self) -> int:
        """
        Number of steps in the prediction horizon.
        """
        return self.qp_controller_config.prediction_horizon

    def _compute_position_constrained_velocity_bounds(
        self,
        degree_of_freedom_symbols: DerivativeMap[FloatVariable],
        limits: DegreeOfFreedomLimits[float],
    ) -> VelocityBoundProfiles:
        """
        Computes per-step velocity bounds that keep the degree of freedom within its
        position limits, slowing it down early enough to stop before a limit is reached.

        :param degree_of_freedom_symbols: Symbolic current state of the degree of
            freedom.
        :param limits: Resolved limits of the degree of freedom.
        """
        time_step = self.time_step
        lower_limits, upper_limits = limits.lower, limits.upper
        velocity_limit = upper_limits.velocity
        if lower_limits.position is None:
            return VelocityBoundProfiles.unconstrained(
                velocity_limit, self.prediction_horizon
            )

        jerk_limit = upper_limits.jerk
        position_range = upper_limits.position - lower_limits.position
        velocity_limit = min(velocity_limit * time_step, position_range / 2) / time_step
        braking_profile = BrakingProfile.fastest(
            initial_velocity=velocity_limit,
            acceleration_limit=upper_limits.acceleration,
            jerk_limit=jerk_limit,
            time_step=time_step,
            prediction_horizon=self.prediction_horizon,
        )
        velocity_lower_bound = self._directional_velocity_bound(
            braking_profile=braking_profile,
            position_error=lower_limits.position - degree_of_freedom_symbols.position,
            jerk_limit=jerk_limit,
            velocity_limit=velocity_limit,
            direction=BoundDirection.LOWER,
        )
        velocity_upper_bound = self._directional_velocity_bound(
            braking_profile=braking_profile,
            position_error=upper_limits.position - degree_of_freedom_symbols.position,
            jerk_limit=jerk_limit,
            velocity_limit=velocity_limit,
            direction=BoundDirection.UPPER,
        )
        goal_profile = sm.max(velocity_lower_bound, 0) + sm.min(velocity_upper_bound, 0)
        skip_first = sm.logic_or(
            velocity_lower_bound[0] >= 0, velocity_upper_bound[0] <= 0
        )
        return VelocityBoundProfiles(
            lower_bound=velocity_lower_bound,
            upper_bound=velocity_upper_bound,
            goal_profile=goal_profile,
            skip_first=skip_first,
        )

    def _directional_velocity_bound(
        self,
        braking_profile: BrakingProfile,
        position_error: sm.Scalar,
        jerk_limit: float,
        velocity_limit: float,
        direction: BoundDirection,
    ) -> sm.Vector:
        """
        Computes the velocity bound that brakes the degree of freedom against one
        position limit, shifting the nominal braking profile by the remaining distance
        to that limit and capping the first step to a single jerk-limited change.

        :param braking_profile: Fastest braking from the velocity limit to rest.
        :param position_error: Remaining distance to the position limit being braked
            against.
        :param jerk_limit: Jerk limit used to cap the first step change.
        :param velocity_limit: Velocity limit clamping the first step change.
        :param direction: Whether the bound brakes against the lower or upper position
            limit.
        """
        sign = direction.sign
        time_step = self.time_step
        velocity_bound = braking_profile.shifted_by(sign * position_error) * sign
        one_step_change = jerk_limit * time_step**2
        one_step_change_bound = sm.limit(
            position_error / time_step,
            min(0.0, -sign * one_step_change),
            max(0.0, -sign * one_step_change),
        )
        one_step_change_bound = sm.limit(
            one_step_change_bound, -velocity_limit, velocity_limit
        )
        velocity_bound[0] = sm.if_less(
            sign * position_error,
            0,
            one_step_change_bound,
            copy(velocity_bound[0]),
        )
        return velocity_bound

    def compute_horizon_bounds(
        self,
        degree_of_freedom_symbols: DerivativeMap[FloatVariable],
        limits: DegreeOfFreedomLimits[float],
        epsilon: float = 0.00001,
    ) -> DegreeOfFreedomLimits[sm.Vector]:
        """
        Computes the velocity, acceleration, and jerk bounds for one degree of freedom
        across the whole prediction horizon, relaxing the jerk limit on the first steps
        when the position goal would otherwise be unreachable.

        :param degree_of_freedom_symbols: Symbolic current state of the degree of
            freedom.
        :param limits: Resolved limits of the degree of freedom.
        :param epsilon: Tolerance below which a velocity bound violation is ignored.
        """
        jerk_limit = limits.upper.jerk
        acceleration_limit = limits.upper.acceleration

        velocity_bounds = self._compute_position_constrained_velocity_bounds(
            degree_of_freedom_symbols=degree_of_freedom_symbols, limits=limits
        )
        velocity_lower_bound = velocity_bounds.lower_bound
        velocity_upper_bound = velocity_bounds.upper_bound

        acceleration_profile = (
            sm.Vector.ones(velocity_upper_bound.shape[0]) * acceleration_limit
        )
        jerk_profile = sm.Vector.ones(velocity_upper_bound.shape[0]) * jerk_limit

        projected_velocity_profile, projected_jerk_profile_violated = (
            self._project_velocity_profiles(
                degree_of_freedom_symbols=degree_of_freedom_symbols,
                goal_profile=velocity_bounds.goal_profile,
                jerk_limit=jerk_limit,
                skip_first=velocity_bounds.skip_first,
            )
        )
        needs_relaxed_jerk_limits = velocity_bounds.is_violated_by(
            projected_velocity_profile, epsilon
        )
        self._relax_jerk_on_initial_steps(
            jerk_profile=jerk_profile,
            projected_jerk_profile_violated=projected_jerk_profile_violated,
            needs_relaxed_jerk_limits=needs_relaxed_jerk_limits,
            jerk_limit=jerk_limit,
        )
        return self._assemble_degree_of_freedom_limits(
            velocity_lower_bound=velocity_lower_bound,
            velocity_upper_bound=velocity_upper_bound,
            acceleration_profile=acceleration_profile,
            jerk_profile=jerk_profile,
        )

    def _project_velocity_profiles(
        self,
        degree_of_freedom_symbols: DerivativeMap[FloatVariable],
        goal_profile: sm.Vector,
        jerk_limit: float,
        skip_first: sm.Scalar,
    ) -> tuple[sm.Vector, sm.Vector]:
        """
        Projects the slow-down-as-fast-as-possible velocity profile under the real jerk
        limit and the jerk profile that would be required without a jerk limit.

        :param degree_of_freedom_symbols: Symbolic current state of the degree of
            freedom.
        :param goal_profile: Goal velocity profile the projection drives towards.
        :param jerk_limit: Jerk limit applied while projecting the velocity profile.
        :param skip_first: Flag marking that the first step is already at rest against a
            limit.
        """
        projected_velocity_profile = SlowdownProfile.immediate(
            current_velocity=degree_of_freedom_symbols.velocity,
            current_acceleration=degree_of_freedom_symbols.acceleration,
            target_velocity_profile=goal_profile,
            jerk_limit=Scalar(jerk_limit),
            time_step=Scalar(self.time_step),
            prediction_horizon=self.prediction_horizon,
            skip_first=skip_first,
        ).velocity
        projected_jerk_profile_violated = SlowdownProfile.immediate(
            current_velocity=degree_of_freedom_symbols.velocity,
            current_acceleration=degree_of_freedom_symbols.acceleration,
            target_velocity_profile=goal_profile,
            jerk_limit=Scalar(np.inf),
            time_step=Scalar(self.time_step),
            prediction_horizon=self.prediction_horizon,
            skip_first=skip_first,
        ).jerk
        return projected_velocity_profile, projected_jerk_profile_violated

    def _relax_jerk_on_initial_steps(
        self,
        jerk_profile: sm.Vector,
        projected_jerk_profile_violated: sm.Vector,
        needs_relaxed_jerk_limits: sm.Scalar,
        jerk_limit: float,
    ) -> None:
        """
        Raises the jerk limit on the first horizon steps to the magnitude required to
        reach the position goal, but only when normal braking would otherwise be
        insufficient.

        :param jerk_profile: Per-step jerk profile that is modified in place.
        :param projected_jerk_profile_violated: Jerk magnitudes required without a jerk
            limit.
        :param needs_relaxed_jerk_limits: Flag marking that the jerk limit must be
            relaxed.
        :param jerk_limit: Nominal jerk limit used when no relaxation is needed.
        """
        for step in range(NUMBER_OF_JERK_RELAXED_STEPS):
            jerk_profile[step] = sm.if_else(
                needs_relaxed_jerk_limits,
                sm.max(
                    Scalar(jerk_limit), sm.abs(projected_jerk_profile_violated[step])
                ),
                sm.Scalar(jerk_limit),
            )

    def _assemble_degree_of_freedom_limits(
        self,
        velocity_lower_bound: sm.Vector,
        velocity_upper_bound: sm.Vector,
        acceleration_profile: sm.Vector,
        jerk_profile: sm.Vector,
    ) -> DegreeOfFreedomLimits[sm.Vector]:
        """
        Combines the velocity, acceleration, and jerk profiles into the horizon limits,
        ensuring the lower bound never exceeds the upper bound.

        :param velocity_lower_bound: Per-step lower velocity bound.
        :param velocity_upper_bound: Per-step upper velocity bound.
        :param acceleration_profile: Per-step acceleration magnitude.
        :param jerk_profile: Per-step jerk magnitude.
        """
        time_step = self.time_step
        velocity_lower_bound = sm.min(velocity_lower_bound, velocity_upper_bound)
        velocity_upper_bound = sm.max(velocity_lower_bound, velocity_upper_bound)
        acceleration_lower_bounds = -acceleration_profile
        acceleration_upper_bounds = acceleration_profile
        jerk_lower_bounds = sm.min(jerk_profile, -jerk_profile) * time_step**2
        jerk_upper_bounds = sm.max(jerk_profile, -jerk_profile) * time_step**2
        return DegreeOfFreedomLimits[sm.Vector](
            lower=DerivativeMap(
                velocity=velocity_lower_bound,
                acceleration=acceleration_lower_bounds,
                jerk=jerk_lower_bounds,
            ),
            upper=DerivativeMap(
                velocity=velocity_upper_bound,
                acceleration=acceleration_upper_bounds,
                jerk=jerk_upper_bounds,
            ),
        )

    def resolve_limits(
        self, degree_of_freedom: DegreeOfFreedom
    ) -> DegreeOfFreedomLimits[float]:
        """
        Builds the lower and upper limit maps for a degree of freedom, filling in
        unbounded acceleration limits and deriving a jerk limit from the configured
        braking time when none is given.

        :param degree_of_freedom: Degree of freedom whose limits are resolved.
        """
        lower_limits = DerivativeMap()
        upper_limits = DerivativeMap()

        if degree_of_freedom.has_position_limits():
            lower_limits.position = degree_of_freedom.limits.lower.position
            upper_limits.position = degree_of_freedom.limits.upper.position
        else:
            lower_limits.position = upper_limits.position = None

        lower_limits.velocity = degree_of_freedom.limits.lower.velocity
        upper_limits.velocity = degree_of_freedom.limits.upper.velocity

        if degree_of_freedom.limits.lower.acceleration is None:
            lower_limits.acceleration = -np.inf
        else:
            lower_limits.acceleration = degree_of_freedom.limits.lower.acceleration
        if degree_of_freedom.limits.upper.acceleration is None:
            upper_limits.acceleration = np.inf
        else:
            upper_limits.acceleration = degree_of_freedom.limits.upper.acceleration

        braking = self._braking(degree_of_freedom)
        upper_limits.jerk = braking.jerk_limit
        if degree_of_freedom.limits.lower.jerk is None:
            lower_limits.jerk = -braking.jerk_limit
        else:
            lower_limits.jerk = max(
                degree_of_freedom.limits.lower.jerk, -braking.jerk_limit
            )

        return DegreeOfFreedomLimits(lower=lower_limits, upper=upper_limits)

    def _braking(self, degree_of_freedom: DegreeOfFreedom) -> JerkLimitedBraking:
        """
        Returns the braking of a degree of freedom from its velocity limit, with its
        declared jerk limit or, without one, the jerk limit of the configured braking
        time, lowered to keep a declared acceleration limit.

        .. warning:: Relaxing the jerk limit near a position limit can exceed the
            acceleration limit as well.

        :param degree_of_freedom: Degree of freedom that brakes.
        """
        velocity_limit = degree_of_freedom.limits.upper.velocity
        time_step = self.time_step
        if degree_of_freedom.limits.upper.jerk is None:
            braking = JerkLimitedBraking.from_braking_time(
                velocity_limit=velocity_limit,
                braking_time=self.qp_controller_config.braking_time,
                time_step=time_step,
            )
        else:
            braking = JerkLimitedBraking(
                velocity_limit=velocity_limit,
                jerk_limit=degree_of_freedom.limits.upper.jerk,
                time_step=time_step,
            )
        acceleration_limit = degree_of_freedom.limits.upper.acceleration
        if acceleration_limit is None:
            return braking
        return braking.limited_to_acceleration(acceleration_limit)

    def compute(
        self,
        degree_of_freedom: DegreeOfFreedom,
    ) -> DegreeOfFreedomLimits[sm.Vector]:
        """
        Computes the horizon bounds for a single degree of freedom, filling in missing
        acceleration and jerk limits.

        :param degree_of_freedom: Degree of freedom whose horizon bounds are computed.
        :raises DegreeOfFreedomBrakingExceedsHorizonError: If the degree of freedom
            cannot brake from its velocity limit to rest within the prediction horizon.
        """
        self._raise_if_braking_exceeds_horizon(degree_of_freedom)
        return self.compute_horizon_bounds(
            degree_of_freedom_symbols=degree_of_freedom.variables,
            limits=self.resolve_limits(degree_of_freedom),
        )

    def _raise_if_braking_exceeds_horizon(
        self, degree_of_freedom: DegreeOfFreedom
    ) -> None:
        """
        Raises when the degree of freedom needs more steps to brake from its velocity
        limit to rest than the prediction horizon leaves before its resting steps.

        :param degree_of_freedom: Degree of freedom whose braking is checked.
        """
        qp_controller_config = self.qp_controller_config
        braking = self._braking(degree_of_freedom)
        if braking.number_of_steps <= qp_controller_config.control_horizon:
            return
        raise DegreeOfFreedomBrakingExceedsHorizonError(
            prediction_horizon=qp_controller_config.prediction_horizon,
            minimum_prediction_horizon=braking.number_of_steps
            + NUMBER_OF_RESTING_STEPS,
            degree_of_freedom_name=str(degree_of_freedom.name),
            velocity_limit=braking.velocity_limit,
            jerk_limit=braking.jerk_limit,
        )


@dataclass
class DecisionVariableSlot:
    """
    One decision variable of the QP: a derivative of a degree of freedom at one step of
    the prediction horizon.
    """

    derivative: Derivatives
    """
    Derivative the decision variable holds.
    """

    step: int
    """
    Step of the prediction horizon the decision variable belongs to.
    """

    degree_of_freedom: DegreeOfFreedom
    """
    Degree of freedom the decision variable belongs to.
    """

    @property
    def debug_name(self) -> str:
        """
        Human readable name of the decision variable, used for debugging.
        """
        short_label = {Derivatives.velocity: "vel", Derivatives.jerk: "jerk"}
        return f"{self.degree_of_freedom.name}_{short_label[self.derivative]}_k_{self.step}"


@dataclass
class DegreeOfFreedomDecisionVariables:
    """
    The velocity and jerk decision variables of the robot's degrees of freedom across the
    prediction horizon, with their bounds and objective weights.
    """

    degrees_of_freedom: list[DegreeOfFreedom]
    """
    Degrees of freedom contributing decision variables.
    """

    qp_controller_config: QPControllerConfig
    """
    Controller configuration providing horizon, derivatives, and weights.
    """

    profiler: DegreeOfFreedomLimitProfiler = field(init=False)
    """
    Profiler resolving the limits and horizon bounds of each degree of freedom.
    """

    def __post_init__(self):
        self.profiler = DegreeOfFreedomLimitProfiler(self.qp_controller_config)

    def direct_limits(self) -> DirectLimits:
        """
        Returns the bounds, weights, and names of the decision variables.
        """
        lower_bounds, upper_bounds = self.free_variable_bounds()
        quadratic_weights, linear_weights = self.init_weights()
        return DirectLimits(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            quadratic_weights=quadratic_weights,
            linear_weights=linear_weights,
            names=[slot.debug_name for slot in self.slots],
        )

    def number_of_steps(self, derivative: Derivatives) -> int:
        """
        Returns the number of prediction horizon steps that have a decision variable
        for ``derivative``.

        :param derivative: Derivative whose decision variables are counted.
        """
        return self.qp_controller_config.prediction_horizon - (
            self.qp_controller_config.max_derivative - derivative
        )

    @property
    def slots(self) -> list[DecisionVariableSlot]:
        """
        Every decision variable, in the order shared by bounds, weights, and names.
        """
        return [
            DecisionVariableSlot(
                derivative=derivative, step=step, degree_of_freedom=degree_of_freedom
            )
            for derivative in (Derivatives.velocity, Derivatives.jerk)
            for step in range(self.number_of_steps(derivative))
            for degree_of_freedom in self.degrees_of_freedom
        ]

    def free_variable_bounds(self) -> tuple[sm.Vector, sm.Vector]:
        """
        Computes the lower and upper box limits of every decision variable.
        """
        horizon_bounds: dict[UUID, DegreeOfFreedomLimits[sm.Vector]] = {
            degree_of_freedom.id: self.profiler.compute(degree_of_freedom)
            for degree_of_freedom in self.degrees_of_freedom
        }
        lower_bounds = []
        upper_bounds = []
        for slot in self.slots:
            bounds = horizon_bounds[slot.degree_of_freedom.id]
            lower_bounds.append(bounds.lower[slot.derivative][slot.step])
            upper_bounds.append(bounds.upper[slot.derivative][slot.step])
        return sm.Vector(lower_bounds), sm.Vector(upper_bounds)

    def init_weights(self) -> tuple[sm.Vector, sm.Vector]:
        """
        Computes the quadratic and linear objective weights of every decision variable.

        The weights ramp up to their full value at the last step with a velocity
        decision variable.
        """
        qp_controller_config = self.qp_controller_config
        decision_variable_limits = {
            degree_of_freedom.id: self._decision_variable_limits(
                upper_limits=self.profiler.resolve_limits(degree_of_freedom).upper,
                time_step=qp_controller_config.model_predictive_control_time_step,
            )
            for degree_of_freedom in self.degrees_of_freedom
        }
        last_velocity_step = self.number_of_steps(Derivatives.velocity) - 1
        quadratic_weights = []
        for slot in self.slots:
            normalized_weight = self.normalize_degree_of_freedom_weight(
                variable_limit=decision_variable_limits[slot.degree_of_freedom.id][
                    slot.derivative
                ],
                base_weight=qp_controller_config.get_degree_of_freedom_weight(
                    slot.degree_of_freedom.name, slot.derivative
                ),
                horizon_index=slot.step,
                total_horizon_length=last_velocity_step,
                growth_factor=qp_controller_config.horizon_weight_gain_scalar,
            )
            quadratic_weights.append(normalized_weight)
        return sm.Vector(quadratic_weights), sm.Vector.zeros(len(quadratic_weights))

    def _decision_variable_limits(
        self, upper_limits: DerivativeMap[float], time_step: float
    ) -> DerivativeMap[float]:
        """
        Returns the bound of each kind of decision variable of a degree of freedom.

        Jerk decision variables hold jerk times the squared time step, so their bound is
        the jerk limit scaled the same way.

        :param upper_limits: Resolved upper limits of the degree of freedom.
        :param time_step: Duration of a single horizon step.
        """
        return DerivativeMap(
            velocity=upper_limits.velocity,
            jerk=upper_limits.jerk * time_step**2,
        )

    def normalize_degree_of_freedom_weight(
        self,
        variable_limit: float | None,
        base_weight: float | None,
        horizon_index: int,
        total_horizon_length: int,
        growth_factor: float,
    ) -> sm.Scalar:
        """
        Scales a free variable weight by its limit so derivatives become comparable, and
        ramps it over the horizon so later time steps are penalized more.

        :param variable_limit: Limit of the free variable used to normalize the weight.
        :param base_weight: Base objective weight before normalization and ramping,
            ``None`` if the free variable is not weighted.
        :param horizon_index: Index of the horizon step the weight applies to.
        :param total_horizon_length: Horizon length over which the weight is ramped.
        :param growth_factor: Factor scaling the weight at the start of the horizon.
        """

        def linear(
            horizon_index: float,
            weight: float,
            total_horizon_length: int,
            growth_factor: float,
        ) -> float:
            start = weight * growth_factor
            slope = (weight - start) / total_horizon_length
            return slope * horizon_index + start

        if variable_limit is None or base_weight is None:
            return 0.0
        weight = linear(horizon_index, base_weight, total_horizon_length, growth_factor)

        return weight * (1 / variable_limit) ** 2

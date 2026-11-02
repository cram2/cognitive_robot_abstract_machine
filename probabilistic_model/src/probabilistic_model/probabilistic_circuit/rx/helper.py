import math
from collections.abc import Sequence

import numpy.typing as npt

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import *
from probabilistic_model.distributions.distributions import (
    SymbolicDistribution,
    IntegerDistribution,
    DiracDeltaDistribution,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.exceptions import (
    EventMixesPointsAndIntervalsError,
    UnboundedEventError,
)
from probabilistic_model.utils import (
    MissingDict,
    event_compatible_for_truncation_with_singletons,
)


def uniform_measure_of_event(event: Event) -> ProbabilisticCircuit:
    """
    Create a uniform measure for the given event.

    :param event: The event
    :return: The circuit describing the uniform measure
    :raises UnboundedEventError: If the event does not bound a continuous variable from
        below and above.
    :raises EventMixesPointsAndIntervalsError: If a continuous variable takes points and
        intervals in the box bounding the event.
    """
    # calculate the bounding box of the event
    bounding_box = event.bounding_box()

    # create a uniform measure for the bounding box
    uniform_model = uniform_measure_of_simple_event(bounding_box)
    # condition the uniform measure on the event
    return uniform_model.truncated(event).distribution


def uniform_measure_of_simple_event(simple_event: SimpleEvent) -> ProbabilisticCircuit:
    """
    Create a uniform measure for the given simple event.

    :param simple_event: The simple event
    :return: The circuit describing the uniform measure over the simple event
    :raises UnboundedEventError: If the simple event does not bound a continuous
        variable from below and above.
    :raises EventMixesPointsAndIntervalsError: If a continuous variable takes points and
        intervals.
    """
    if not event_compatible_for_truncation_with_singletons(
        simple_event.as_composite_set()
    ):
        raise EventMixesPointsAndIntervalsError(simple_event)

    # initialize the root of the circuit
    result = ProbabilisticCircuit()
    uniform_model = ProductUnit(probabilistic_circuit=result)
    for variable, assignment in simple_event.items():

        # handle different variables
        if isinstance(variable, Continuous):
            if not (
                math.isfinite(assignment.simple_sets[0].lower)
                and math.isfinite(assignment.simple_sets[-1].upper)
            ):
                raise UnboundedEventError(variable=variable)

            distribution = SumUnit(probabilistic_circuit=result)
            if all(assignment_.is_singleton() for assignment_ in assignment):
                for assignment_ in assignment:
                    dirac = DiracDeltaDistribution(
                        variable=variable, location=assignment_.lower, density_cap=1.0
                    )
                    distribution.add_subcircuit(
                        UnivariateContinuousLeaf(dirac, probabilistic_circuit=result),
                        0.0,
                    )
            else:
                for assignment_ in assignment:
                    uniform = UniformDistribution(
                        variable=variable, interval=assignment_
                    )
                    distribution.add_subcircuit(
                        UnivariateContinuousLeaf(uniform, probabilistic_circuit=result),
                        -uniform.log_probability_density_function_value(),
                    )
            distribution.normalize()

        # create uniform distribution for symbolic variables
        elif isinstance(variable, Symbolic):
            distribution = SymbolicDistribution(
                variable=variable,
                probabilities=MissingDict(
                    float,
                    {
                        hash(value): 1 / len(assignment.simple_sets)
                        for value in assignment
                    },
                ),
            )
            distribution = UnivariateDiscreteLeaf(
                distribution, probabilistic_circuit=result
            )

        # create uniform distribution for integer variables
        elif isinstance(variable, Integer):
            distribution = IntegerDistribution(
                variable=variable,
                probabilities=MissingDict(
                    float,
                    {
                        value.lower: 1 / len(assignment.simple_sets)
                        for value in assignment
                    },
                ),
            )
            distribution = UnivariateDiscreteLeaf(
                distribution, probabilistic_circuit=result
            )

        else:
            raise NotImplementedError

        # mount the distribution on the root
        uniform_model.add_subcircuit(distribution)

    return result


def fully_factorized(
    variables: Iterable[Variable],
    means: Optional[Dict[Continuous, float]] = None,
    variances: Optional[Dict[Continuous, float]] = None,
) -> ProbabilisticCircuit:
    """
    Create a fully factorized distribution over a set of variables.

    For symbolic variables, the distribution is uniform. For continuous variables, the
    distribution is normal.

    :param variables: The variables.
    :param means: The means of the normal distributions. Defaults to 0 for every not
        specified variable.
    :param variances: The variances of the normal distributions. Defaults to 1 for every
        not specified variable.
    :return: The circuit describing the fully factorized normal distribution
    """
    pc = ProbabilisticCircuit()
    if means is None:
        means = {}

    if variances is None:
        variances = {}

    # initialize the root of the circuit
    root = ProductUnit(probabilistic_circuit=pc)
    for variable in variables:

        # create a normal distribution for every continuous or integer variable
        if isinstance(variable, (Continuous, Integer)):
            distribution = GaussianDistribution(
                variable=variable,
                location=means.get(variable, 0.0),
                scale=variances.get(variable, 1.0),
            )
            distribution = leaf(distribution, pc)

        # create uniform distribution for symbolic variables
        elif isinstance(variable, Symbolic):
            domain_elements = list(variable.domain.simple_sets)
            distribution = SymbolicDistribution(
                variable=variable,
                probabilities=MissingDict(
                    float, {hash(v): 1 / len(domain_elements) for v in domain_elements}
                ),
            )
            distribution = leaf(distribution, pc)
        else:
            raise NotImplementedError(f"Variable type not supported: {variable}")
        root.add_subcircuit(distribution)

    return root.probabilistic_circuit


def factorized_gaussian_mixture(
    variables: Sequence[Continuous], means: npt.NDArray, scales: npt.NDArray
) -> ProbabilisticCircuit:
    """
    Create an equally weighted mixture of fully factorized Gaussian distributions.

    :param variables: The variables, in the order of the columns of ``means`` and
        ``scales``.
    :param means: The mean of every variable, one row per mixture component.
    :param scales: The standard deviation of every variable, one row per mixture
        component.
    :return: The circuit describing the mixture.
    """
    result = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=result)
    for component_means, component_scales in zip(means, scales):
        component = ProductUnit(probabilistic_circuit=result)
        for variable, mean, scale in zip(variables, component_means, component_scales):
            component.add_subcircuit(
                leaf(
                    GaussianDistribution(
                        variable=variable, location=float(mean), scale=float(scale)
                    ),
                    result,
                )
            )
        root.add_subcircuit(component, 0.0)
    root.normalize()
    return result


def multiply_distributions(
    distribution: ProbabilisticCircuit, expansion: ProbabilisticCircuit
):
    """
    Expand the `distribution` by the `expansion` using factorization.

    `distribution` is expanded in-place.

    The resulting distribution is the product of the expansion and the distribution.

    :param distribution: The probabilistic circuit to expand
    :param expansion: The probabilistic circuit to use for expansion
    :return: The expanded probabilistic circuit
    """
    old_distribution_root = distribution.root
    expansion_root_index = expansion.root.index
    remap = distribution.mount(expansion.root)
    ff_root = remap[expansion_root_index]

    new_root = ProductUnit(probabilistic_circuit=distribution)
    new_root.add_subcircuit(ff_root)
    new_root.add_subcircuit(old_distribution_root)
    return distribution

from .probabilistic_circuit import *
from ...distributions import UniformDistribution, SymbolicDistribution, IntegerDistribution, GaussianDistribution
from ...utils import MissingDict


def uniform_measure_of_event(event: Event) -> ProbabilisticCircuit:
    """
    Create a uniform measure for the given event.

    :param event: The event
    :return: The circuit describing the uniform measure
    """

    # calculate the bounding box of the event
    bounding_box = event.bounding_box()

    # create a uniform measure for the bounding box
    uniform_model = uniform_measure_of_simple_event(bounding_box)
    # condition the uniform measure on the event
    uniform_model, _ = uniform_model.truncated(event)

    return uniform_model


def uniform_measure_of_simple_event(simple_event: SimpleEvent) -> ProbabilisticCircuit:
    """
    Create a uniform measure for the given simple event.
    :param simple_event: The simple event
    :return: The circuit describing the uniform measure over the simple event
    """

    # initialize the root of the circuit
    result = ProbabilisticCircuit()
    uniform_model = ProductUnit(probabilistic_circuit=result)
    for variable, assignment in simple_event.items():

        # handle different variables
        if isinstance(variable, Continuous):

            # create a uniform distribution for every interval in a continuous variables description
            distribution = SumUnit(probabilistic_circuit=result)
            for assignment_ in assignment:
                u = UniformDistribution(variable, assignment_)
                distribution.add_subcircuit(UnivariateContinuousLeaf(u, probabilistic_circuit=result), 1 / u.pdf_value())
            distribution.normalize()

        # create uniform distribution for symbolic variables
        elif isinstance(variable, Symbolic):
            distribution = SymbolicDistribution(variable,
                                                MissingDict(float, {hash(value): 1 / len(assignment.simple_sets) for
                                                                    value in assignment}))
            distribution = UnivariateDiscreteLeaf(distribution, probabilistic_circuit=result)

        # create uniform distribution for integer variables
        elif isinstance(variable, Integer):
            distribution = IntegerDistribution(variable,
                                               MissingDict(float, {value.lower: 1 / len(assignment.simple_sets) for
                                                                   value in assignment}))
            distribution = UnivariateDiscreteLeaf(distribution, probabilistic_circuit=result)

        else:
            raise NotImplementedError

        # mount the distribution on the root
        uniform_model.add_subcircuit(distribution)

    return result


def fully_factorized(variables: Iterable[Variable],
                     means: Optional[Dict[Continuous, float]] = None,
                     variances: Optional[Dict[Continuous, float]] = None) -> ProbabilisticCircuit:
    """
    Create a fully factorized distribution over a set of variables.
    For symbolic variables, the distribution is uniform.
    For continuous variables, the distribution is normal.

    :param variables: The variables.

    :param means: The means of the normal distributions.
    Defaults to 0 for every not specified variable.

    :param variances: The variances of the normal distributions.
    Defaults to 1 for every not specified variable.

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

        # create a normal distribution for every continuous variable
        if isinstance(variable, Continuous):
            distribution = GaussianDistribution(variable, means.get(variable, 0.), variances.get(variable, 1.))
            distribution = leaf(distribution, pc)

        # create uniform distribution for symbolic variables
        elif isinstance(variable, Symbolic):
            domain_elements = list(variable.domain.simple_sets)
            distribution = SymbolicDistribution(variable, MissingDict(float, {hash(v): 1 / len(domain_elements)
                                                                              for v in domain_elements}))
            distribution = leaf(distribution, pc)
        else:
            raise NotImplementedError(f"Variable type not supported: {variable}")
        root.add_subcircuit(distribution)

    return root.probabilistic_circuit

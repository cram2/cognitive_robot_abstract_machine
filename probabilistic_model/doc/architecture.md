# Technical Guide

This document gives a structural overview of the pm package.
The abstract base class ProbabilisticModel defines an interface for probabilistic models that support all tractable
query types.

The class inheritance diagram for parametric distributions is shown below.


```{eval-rst}
    .. autoclasstree:: probabilistic_model.probabilistic_model probabilistic_model.distributions.distributions probabilistic_model.distributions.uniform probabilistic_model.distributions.gaussian probabilistic_model.distributions.multinomial
        :zoom:
        :namespace: probabilistic_model
        :strict:
        :caption: Inheritance Diagram for parametric distributions.
```

For bayesian networks, the next class diagram is relevant.

```{eval-rst}
    .. autoclasstree:: probabilistic_model.bayesian_network.bayesian_network
        :zoom:
        :namespace: probabilistic_model
        :strict:
        :caption: Inheritance Diagram for baysian networks.
```

For networkx based probabilistic circuits the next class diagram is relevant.

```{eval-rst}
    .. autoclasstree:: probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit probabilistic_model.probabilistic_circuit.rx.helper
        :zoom:
        :namespace: probabilistic_model
        :strict:
        :caption: Inheritance Diagram for probabilistic circuits implemented with rustworkx.
```

Finally, for jax based faster circuits with limited inference, this class diagram is relevant.

```{eval-rst}
    .. autoclasstree:: probabilistic_model.probabilistic_circuit.jax probabilistic_model.probabilistic_circuit.jax.coupling_circuit probabilistic_model.probabilistic_circuit.jax.gaussian_layer probabilistic_model.probabilistic_circuit.jax.discrete_layer probabilistic_model.probabilistic_circuit.jax.uniform_layer
        :zoom:
        :namespace: probabilistic_model.probabilistic_circuit.jax
        :caption: Inheritance Diagram for probabilistic circuits implemented with jax.
```
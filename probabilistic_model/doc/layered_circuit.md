---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Representations of Circuits

While understanding the concepts of a probabilistic circuit is subject to math, implementing it is a whole different
story.
This section discusses different approaches to represent circuits.
This package has three implementations: one that stores the circuit as a graph in rustworkx, and two that store it in
layers, one in NumPy and one in JAX.

## The DAG (rustworkx) Way

The easiest and naive way of implementing a circuit is using a directed acyclic graph (DAG).
The graph directly follows definition {prf:ref}`def-probabilistic-circuit`.

Let's look at an example.

```{code-cell} ipython3
import numpy as np
import plotly
plotly.offline.init_notebook_mode()
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from random_events.interval import SimpleInterval
from random_events.variable import Continuous
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit, SumUnit, ProductUnit, leaf


x = Continuous("x")
y = Continuous("y")
model = ProbabilisticCircuit()
sum1, sum2, sum3 = SumUnit(probabilistic_circuit=model), SumUnit(probabilistic_circuit=model), SumUnit(probabilistic_circuit=model)
sum4, sum5 = SumUnit(probabilistic_circuit=model), SumUnit(probabilistic_circuit=model)
prod1, prod2 = ProductUnit(probabilistic_circuit=model), ProductUnit(probabilistic_circuit=model)

sum1.add_subcircuit(prod1, np.log(0.5))
sum1.add_subcircuit(prod2, np.log(0.5))
prod1.add_subcircuit(sum2)
prod1.add_subcircuit(sum4)
prod2.add_subcircuit(sum3)
prod2.add_subcircuit(sum5)

d_x1 = leaf(UniformDistribution(variable=x, interval=SimpleInterval.from_data(0, 1)), probabilistic_circuit=model)
d_x2 = leaf(UniformDistribution(variable=x, interval=SimpleInterval.from_data(2, 3)), probabilistic_circuit=model)
d_y1 = leaf(UniformDistribution(variable=y, interval=SimpleInterval.from_data(0, 1)), probabilistic_circuit=model)
d_y2 = leaf(UniformDistribution(variable=y, interval=SimpleInterval.from_data(3, 4)), probabilistic_circuit=model)

sum2.add_subcircuit(d_x1, np.log(0.8))
sum2.add_subcircuit(d_x2, np.log(0.2))
sum3.add_subcircuit(d_x1, np.log(0.7))
sum3.add_subcircuit(d_x2, np.log(0.3))

sum4.add_subcircuit(d_y1, np.log(0.5))
sum4.add_subcircuit(d_y2, np.log(0.5))
sum5.add_subcircuit(d_y1, np.log(0.1))
sum5.add_subcircuit(d_y2, np.log(0.9))

model.plot_structure()
plt.show()
```

```{code-cell} ipython3
fig = go.Figure(model.plot(), model.plotly_layout())
fig.show()
```

The benefits of the DAG representation are:
- Understandability
- Simplicity of implementation
- Great extendability
- Great for structure learning
- Great for teaching

The drawbacks are:
- Every query visits the nodes one by one in Python, which is slow for large circuits
- Nodes that do the same operation are not computed together, so rustworkx does not benefit from SIMD instructions
  or hardware acceleration the way a layered circuit does


## The Layered Way
Modern literature suggests representing circuits in a way that is compatible with modern hardware
acceleration. {cite}`liu2024scaling`, {cite}`peharz2020einsum`.

Doing so requires a topological sorting of the circuit. In that topological sorting, each layer represents a set of
nodes at the same depth (distance to the root) that can be computed in parallel.
These nodes have to be of the same type, such that their operations (weighted sum, product, density, etc.) can be computed in parallel.

The way a layered circuit is structured is shown in the figure below.

![Grouped operations in a layered circuit](layered_example.png)

We can see that similar operations have been grouped together.
Now they can be executed as one array operation instead of a for loop in Python.
The benefits of the layered representation are:
- One array operation per layer instead of one Python call per node
- Compatible with modern machine learning frameworks
- Compatible with modern hardware acceleration
- Most likely the future of probabilistic circuits


The drawbacks are:
- Harder to understand
- Harder to maintain
- Harder to extend with new kinds of nodes or to change in structure

The two layered implementations of this package, NumPy and JAX, are built for different things.

## NumPy Implementation

The NumPy implementation is the `LayeredProbabilisticCircuit` in
`probabilistic_model.probabilistic_circuit.tensorized`.
It answers every query the rustworkx implementation answers, including marginals,
truncation and conditioning, which return a NumPy circuit again.

```{code-cell} ipython3
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import RustworkxCircuitToLayeredCircuitConverter

numpy_model = RustworkxCircuitToLayeredCircuitConverter.convert(model)
event = SimpleEvent.from_data({x: closed(0.25, 2.5)}).as_composite_set()
truncated, probability = numpy_model.truncated(event)
print(probability)
```

## JAX Implementation

The JAX implementation, the `DifferentiableLayeredCircuit` in `probabilistic_model.probabilistic_circuit.jax`, calculates the log-likelihood and learns the
parameters of a circuit by gradient descent.
It answers no other query; a trained JAX circuit is converted into a NumPy circuit for those.
A JAX circuit is created from a NumPy circuit, so the example from above looks as follows:

```{code-cell} ipython3
from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import DifferentiableLayeredCircuit

jax_model = CircuitRepresentations().convert(numpy_model, DifferentiableLayeredCircuit)
print(jax_model.root)
```

The JAX implementation uses equinox to aid with an OOP approach to the circuit.
It uses sparse matrices to represent edges between the layers and hence does not suffer from extreme memory consumption
like EinsumNetworks.

JAX circuits are approximately **5** times faster than the rustworkx implementation in calculating the
log-likelihood of a joint probability tree on a CPU, and hence are a great tool for doing deep learning with circuits.
For the speed-up to kick in, the JAX computational graph that describes the circuit has to be compiled.
This is expensive, so don't do it more than needed.
However, for a fixed circuit, the speed-up is immense.

`probabilistic_model/scripts/jpt_speed_comparison.py` reproduces this measurement.

Be aware that the JAX implementation is still in development and might not be as stable as the rustworkx implementation.
I would be happy to get support here if someone is interested in it.

## Which One to Use

| implementation | built for |
| --- | --- |
| rustworkx | trying out new circuit structures, and understanding and developing new algorithms |
| NumPy | speed: answering queries, including truncation, conditioning and marginals |
| JAX | learning the parameters of a circuit by gradient descent |

All three can be converted into each other, and every conversion goes through NumPy.
`RustworkxCircuitToLayeredCircuitConverter` and `LayeredCircuitToRustworkxCircuitConverter` in
`probabilistic_model.adapters.rustworkx_tensorized` convert between rustworkx and NumPy,
`LayeredCircuitToDifferentiableLayeredCircuitConverter` and `DifferentiableLayeredCircuitToLayeredCircuitConverter` in
`probabilistic_model.adapters.jax_tensorized` between NumPy and JAX.
`CircuitRepresentations` in `probabilistic_model.adapters.circuit_representations` chains these converters,
so that it converts a circuit into any representation, for instance rustworkx into JAX.
Both layered circuits group their nodes the same way, so the conversion between them copies arrays layer by layer.
To learn the parameters of a NumPy circuit, convert it to JAX, train it there and
convert it back.
Classification circuits and coupling circuits only exist in JAX and are not converted.

The NumPy circuit is faster than rustworkx for the probability of an event, for
truncation and for conditioning, by between about 2 and 60 times on a joint probability
tree and on Gaussian mixtures, and more the larger the event or the mixture. Rustworkx stays
faster for the likelihood of large batches of events on circuits whose leaves have small,
disjoint supports, like those of a joint probability tree.
`experiments/src/experiments/probabilistic_model_experiments/layered_circuit_speed.py`
and `gaussian_mixture_speed.py` in the same folder measure both.

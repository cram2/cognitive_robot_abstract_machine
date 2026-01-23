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

# Probabilistic Circuits User Guide

This tutorial walks you through the basics of using the Probabilistic Circuits (PC) from this package.
First, let's start by importing the necessary modules.

```{code-cell} ipython3
import plotly
plotly.offline.init_notebook_mode()
import plotly.graph_objs as go

from random_events.interval import closed_open, closed
from random_events.product_algebra import Continuous, Event, SimpleEvent
from probabilistic_model.distributions import *
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import *
```

Let's have a look at input units.
Input units (Leaves) are nodes that are at the end of the probabilistic circuit.
These leaves can contain any distribution and will try to call it for the respective inference tasks.
Hence, the distributions need to implement the methods required for the inference tasks.
Leaf distributions have to be wrapped in a leaf node.

Currently supported distributions are:
- (Truncated) Normal Distributions
- Uniform Distributions
- Dirac Delta Distributions
- Multinomial Distributions
- Integer Distributions

You can create them by calling the corresponding class constructors.
For this tutorial, we will stick to normal distributions.

```{code-cell} ipython3
x = Continuous("x")
y = Continuous("y")
pc = ProbabilisticCircuit()
p_x_1 = leaf(GaussianDistribution(x, 0, 1), pc)
```

We can always look at the graph that we have by calling the plot_structure method.

```{code-cell} ipython3
pc.plot_structure()
```

One node only is pretty boring, so let us create a gaussian mixture model.

```{code-cell} ipython3
p_x_2 = leaf(GaussianDistribution(x, 2, 1), pc)
p_x = SumUnit(probabilistic_circuit=pc)
p_x.add_subcircuit(p_x_1, np.log(0.3))
p_x.add_subcircuit(p_x_2, np.log(0.7))
pc.plot_structure()
```

Now we got a more interesting model. 
We can see the nodes we created and the connections between them. 
Even the log_weights are indicated by the opacity of an edge.
Let's create a more complex model by becoming multivariate through product units.

```{code-cell} ipython3
p_y = leaf(GaussianDistribution(y, 1, 2), pc)
p_xy = ProductUnit(probabilistic_circuit=pc)
p_xy.add_subcircuit(p_x)
p_xy.add_subcircuit(p_y)
pc.plot_structure()
```

Now we have a model that is a bit more complex. 
We can now observe how conditioning modifies the structure of a probabilistic circuit.

```{code-cell} ipython3
e = SimpleEvent({x: closed_open(0, 1),
                 y: closed_open(0, 0.5 ) | closed(1, 1.5)}).as_composite_set()
p_xy_conditioned, _ = p_xy.probabilistic_circuit.truncated(e)
p_xy_conditioned.plot_structure()
```

We can also look at the distributions on the data level.

```{code-cell} ipython3
fig = go.Figure(p_xy_conditioned.plot(), p_xy_conditioned.plotly_layout())
fig.show()
```

We can also create a circuit from a bayesian network.

```{code-cell} ipython3
from probabilistic_model.bayesian_network.bayesian_network import *
from random_events.set import *
from random_events.variable import *
from random_events.interval import *
from enum import IntEnum

# Declare variable types and variables
class Success(IntEnum):
    FAILURE = 0
    SUCCESS = 1
    
class ObjectPosition(IntEnum):
    LEFT = 0
    RIGHT = 1
    CENTER = 2
    
class Mood(IntEnum):
    HAPPY = 0
    SAD = 1

success = Symbolic("Success", Set.from_iterable(Success))
object_position = Symbolic("ObjectPosition", Set.from_iterable(ObjectPosition))
mood = Symbolic("Mood", Set.from_iterable(Mood))

# construct Bayesian network
bn = BayesianNetwork()

# create root
cpd_success = Root(SymbolicDistribution(success, MissingDict(float, {hash(Success.FAILURE): 0.8, hash(Success.SUCCESS): 0.2})), bayesian_network=bn)

# create P(ObjectPosition | Success)
cpd_object_position = ConditionalProbabilityTable(bayesian_network=bn)
cpd_object_position.conditional_probability_distributions[Success.FAILURE] = SymbolicDistribution(object_position, 
                                                                                                       MissingDict(float, {ObjectPosition.LEFT: 0.3, 
                                                                                                                           ObjectPosition.RIGHT: 0.3, 
                                                                                                                           ObjectPosition.CENTER: 0.4}))
cpd_object_position.conditional_probability_distributions[Success.SUCCESS] = SymbolicDistribution(object_position,
                                                                                                        MissingDict(float, {ObjectPosition.LEFT: 0.3, 
                                                                                                                            ObjectPosition.RIGHT: 0.3, 
                                                                                                                            ObjectPosition.CENTER: 0.4}))
bn.add_edge(cpd_success, cpd_object_position)

# create P(Mood | Success)
cpd_mood = ConditionalProbabilityTable(bayesian_network=bn)
cpd_mood.conditional_probability_distributions[Success.FAILURE] = SymbolicDistribution(mood, 
                                                                                            MissingDict(float, {Mood.HAPPY: 0.2, 
                                                                                                                Mood.SAD: 0.8}))
cpd_mood.conditional_probability_distributions[Success.SUCCESS] = SymbolicDistribution(mood, 
                                                                                            MissingDict(float, {Mood.HAPPY: 0.9, 
                                                                                                                Mood.SAD: 0.1}))
bn.add_edge(cpd_success, cpd_mood)

# create P(X, Y | ObjectPosition)
cpd_xy = ConditionalProbabilisticCircuit(bayesian_network=bn)
default_circuit = ProbabilisticCircuit()
product_unit = ProductUnit(probabilistic_circuit=default_circuit)
product_unit.add_subcircuit(leaf(GaussianDistribution(x, 0, 1), default_circuit))
product_unit.add_subcircuit(leaf(GaussianDistribution(y, 0, 1), default_circuit))

cpd_xy.conditional_probability_distributions[hash(ObjectPosition.LEFT)] = default_circuit.truncated(SimpleEvent({x: closed(-np.inf, -0.5)}).as_composite_set())[0]
cpd_xy.conditional_probability_distributions[hash(ObjectPosition.RIGHT)] = default_circuit.truncated(SimpleEvent({x: open(0.5, np.inf)}).as_composite_set())[0]
cpd_xy.conditional_probability_distributions[hash(ObjectPosition.CENTER)] = default_circuit.truncated(SimpleEvent({x: open_closed(-0.5, 0.5)}).as_composite_set())[0]

bn.add_edge(cpd_object_position, cpd_xy)

bn.plot()
```

```{code-cell} ipython3
pc_bn = bn.as_probabilistic_circuit()
pc_bn.plot_structure()

```

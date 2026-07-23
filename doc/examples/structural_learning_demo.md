---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Probabilistic Circuit Structural Learning Demo

This demo presents the structural learning functionality implemented for
RX probabilistic circuits.

The implementation follows the pruning and growing strategy described in:

Dang, M., Liu, A., and Van den Broeck, G.  
"Sparse Probabilistic Circuits via Pruning and Growing."  
NeurIPS 2022.

Structural learning adapts the structure of a probabilistic circuit based
on observed data.

The implemented process consists of:

1. Computing edge flows.
2. Removing low-contribution edges (pruning).
3. Increasing capacity by duplicating SumUnits (growing).
4. Alternating pruning and growing operations.


## Imports

```{code-cell} python
import sys
from pathlib import Path

ROOT = Path("../../").resolve()

sys.path.append(
    str(ROOT / "probabilistic_model" / "src")
)

import numpy as np

from probabilistic_model.probabilistic_circuit.rx.learning import (
    calculate_edge_flows,
    prune_edges,
    grow_nodes,
    sparse_probabilistic_circuit_learning,
)

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    ProductUnit,
    leaf,
    Continuous,
)

from probabilistic_model.distributions.gaussian import GaussianDistribution
```


# Creating a small probabilistic circuit

We create a simple Gaussian mixture model represented as a
probabilistic circuit.

The circuit contains two Gaussian components combined by a SumUnit.

```{code-cell} python
def create_demo_circuit():

    variable = Continuous("x")

    circuit = ProbabilisticCircuit()

    root = SumUnit(
        probabilistic_circuit=circuit
    )

    product_1 = ProductUnit(
        probabilistic_circuit=circuit
    )

    product_2 = ProductUnit(
        probabilistic_circuit=circuit
    )

    gaussian_1 = leaf(
        GaussianDistribution(
            variable=variable,
            location=0,
            scale=1,
        ),
        circuit,
    )

    gaussian_2 = leaf(
        GaussianDistribution(
            variable=variable,
            location=5,
            scale=1,
        ),
        circuit,
    )

    product_1.add_subcircuit(
        gaussian_1
    )

    product_2.add_subcircuit(
        gaussian_2
    )

    root.add_subcircuit(
        product_1,
        np.log(0.5),
    )

    root.add_subcircuit(
        product_2,
        np.log(0.5),
    )

    return circuit


circuit = create_demo_circuit()

circuit
```


# Dataset

The dataset contains observations generated from the two Gaussian
components.

```{code-cell} python
data = np.array(
    [
        [0.0],
        [5.0],
    ]
)
```


# Edge Flow Calculation

The edge flow calculation measures the contribution of each weighted
connection in a SumUnit.

Edges with low contribution are candidates for pruning.

The implementation is available in:

```{eval-rst}
.. autofunction:: probabilistic_model.probabilistic_circuit.rx.learning.calculate_edge_flows
```

```{code-cell} python
flows = calculate_edge_flows(
    circuit,
    data,
)

flows
```


# Pruning Edges

Pruning removes edges with low probability contribution while keeping
the most relevant paths in the circuit.

The pruning procedure:

1. Computes edge flows.
2. Ranks edges according to their contribution.
3. Removes the least relevant edges.
4. Normalizes the remaining SumUnit weights.

The implementation is available in:

```{eval-rst}
.. autofunction:: probabilistic_model.probabilistic_circuit.rx.learning.prune_edges
```

```{code-cell} python
edges_before = len(
    list(circuit.edges())
)

prune_edges(
    circuit,
    data,
    prune_fraction=0.2,
)

edges_after = len(
    list(circuit.edges())
)

print(
    "Edges before pruning:",
    edges_before,
)

print(
    "Edges after pruning:",
    edges_after,
)
```


# Growing the Circuit

Growing increases the representational capacity of the probabilistic
circuit.

Selected SumUnits are duplicated and their weights are slightly
perturbed to create new structural alternatives.

The implementation is available in:

```{eval-rst}
.. autofunction:: probabilistic_model.probabilistic_circuit.rx.learning.grow_nodes
```

```{code-cell} python
nodes_before = len(
    list(circuit.graph.nodes())
)

grow_nodes(
    circuit,
    fraction=0.5,
    noise_scale=1e-3,
)

nodes_after = len(
    list(circuit.graph.nodes())
)

print(
    "Nodes before growing:",
    nodes_before,
)

print(
    "Nodes after growing:",
    nodes_after,
)
```


# Complete Structural Learning Pipeline

The complete structural learning algorithm alternates pruning and growing
operations.

The implementation is available in:

```{eval-rst}
.. autofunction:: probabilistic_model.probabilistic_circuit.rx.learning.sparse_probabilistic_circuit_learning
```

```{code-cell} python
circuit = create_demo_circuit()

result = sparse_probabilistic_circuit_learning(
    circuit,
    data,
    prune_fraction=0.2,
    grow_fraction=0.5,
    noise_scale=1e-3,
    iterations=2,
)

result
```


# When to Use Structural Learning

Structural learning is useful when:

- The initial probabilistic circuit contains redundant structure.
- The architecture is difficult to design manually.
- The model needs to adapt its complexity according to data.
- A compact but expressive probabilistic circuit is required.

Pruning improves efficiency by removing unnecessary connections.

Growing increases modelling capacity by introducing additional structural
components.

Together, both operations allow the probabilistic circuit to automatically
adapt its structure.
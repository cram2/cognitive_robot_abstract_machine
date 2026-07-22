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

This notebook demonstrates the structural learning functionality
implemented for RX probabilistic circuits.

The demo follows the pruning and growing strategy described in:

Dang, M., Liu, A., and Van den Broeck, G.
"Sparse Probabilistic Circuits via Pruning and Growing."
NeurIPS 2022.

The process consists of:

1. Creating an initial probabilistic circuit.
2. Computing edge flows.
3. Removing low-contribution connections.
4. Growing the circuit by duplicating SumUnits.
5. Applying the complete structural adaptation process.

```{code-cell} python
import sys
from pathlib import Path

ROOT = Path("../../").resolve()

sys.path.append(
    str(ROOT / "probabilistic_model" / "src")
)

sys.path.append(
    str(ROOT / "krrood" / "src")
)

sys.path.append(
    str(ROOT / "semantic_digital_twin" / "src")
)
```

```{code-cell} python
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

## Creating a small probabilistic circuit

We create a simple Gaussian mixture model represented as a
probabilistic circuit.

The initial circuit contains two Gaussian components combined
by a SumUnit.

```{code-cell} python
def create_demo_circuit():

    x = Continuous("x")

    probabilistic_circuit = ProbabilisticCircuit()

    root = SumUnit(
        probabilistic_circuit=probabilistic_circuit
    )

    product_unit_1 = ProductUnit(
        probabilistic_circuit=probabilistic_circuit
    )

    product_unit_2 = ProductUnit(
        probabilistic_circuit=probabilistic_circuit
    )

    leaf_unit_1 = leaf(
        GaussianDistribution(
            variable=x,
            location=0,
            scale=1,
        ),
        probabilistic_circuit,
    )

    leaf_unit_2 = leaf(
        GaussianDistribution(
            variable=x,
            location=5,
            scale=1,
        ),
        probabilistic_circuit,
    )

    product_unit_1.add_subcircuit(leaf_unit_1)

    product_unit_2.add_subcircuit(leaf_unit_2)

    root.add_subcircuit(
        product_unit_1,
        np.log(0.5)
    )

    root.add_subcircuit(
        product_unit_2,
        np.log(0.5)
    )

    return probabilistic_circuit


circuit = create_demo_circuit()

circuit
```

## 1. Computing Edge Flows

Edge flows estimate how much every weighted connection contributes
to the probability distribution.

Connections with low contribution are candidates for pruning.

```{code-cell} python
data = np.array(
    [
        [0.0],
        [5.0],
    ]
)


flows = calculate_edge_flows(
    circuit,
    data,
)


flows
```

## 2. Pruning Low-Contribution Edges

Pruning removes unnecessary connections while keeping the most
important probability paths.

The remaining SumUnit weights are normalized afterwards.

```{code-cell} python
before_edges = len(
    list(circuit.edges())
)


prune_edges(
    circuit,
    data,
    prune_fraction=0.2,
)


after_edges = len(
    list(circuit.edges())
)


print("Edges before pruning:", before_edges)
print("Edges after pruning:", after_edges)
```

## 3. Growing the Circuit

Growing increases the capacity of the probabilistic circuit.

Selected SumUnits are duplicated and their weights are slightly
perturbed to create new structural alternatives.

```{code-cell} python
before_nodes = len(
    list(circuit.graph.nodes())
)


grow_nodes(
    circuit,
    fraction=0.5,
    noise_scale=1e-3,
)


after_nodes = len(
    list(circuit.graph.nodes())
)


print("Nodes before growing:", before_nodes)
print("Nodes after growing:", after_nodes)
```

## 4. Complete Structural Learning Pipeline

The complete algorithm alternates pruning and growing operations.

It automatically reduces unnecessary structure and increases
capacity where required by the data.

```{code-cell} python
circuit = create_demo_circuit()


result = sparse_probabilistic_circuit_learning(
    circuit,
    data,
    prune_fraction=0.2,
    grow_fraction=0.5,
    iterations=2,
)


result
```

## When should structural learning be used?

Structural learning is useful when:

- the initial probabilistic circuit contains redundant structure,
- manually designing the architecture is difficult,
- the model needs to adapt its complexity to the data,
- a compact but expressive probabilistic circuit is desired.

Pruning improves efficiency by removing unnecessary connections,
while growing increases modelling capacity by introducing new
structural components.

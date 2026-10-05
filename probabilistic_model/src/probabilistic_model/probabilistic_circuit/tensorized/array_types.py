"""
Names for the kinds of numpy arrays that the layers of a layered circuit pass around.

They are plain aliases of :class:`numpy.ndarray`; they only tell the reader what an
array holds and which shape it has.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

SampleArray: TypeAlias = npt.NDArray[np.float64]
"""
Samples of all variables of a circuit, shape (#samples, #variables of the circuit).
"""

SampleColumn: TypeAlias = npt.NDArray[np.float64]
"""
The values of one variable in a set of samples, shape (#samples,).
"""

SampleRows: TypeAlias = npt.NDArray[np.int64]
"""
Indices of rows of a :data:`SampleArray`.
"""

SampleValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per sample, shape (#samples,).
"""

NodeIndices: TypeAlias = npt.NDArray[np.int64]
"""
Indices of nodes of a layer.
"""

NodeValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per node of a layer, shape (#nodes,).
"""

NodeMask: TypeAlias = npt.NDArray[np.bool_]
"""
One flag per node of a layer, shape (#nodes,).
"""

NodeSelection: TypeAlias = NodeIndices | NodeMask
"""
Which nodes of a layer to keep: their indices, or one flag per node.
"""

SampleNodeValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per sample and node of a layer, shape (#samples, #nodes).
"""

SampleNodeMask: TypeAlias = npt.NDArray[np.bool_]
"""
One flag per sample and node of a layer, shape (#samples, #nodes).
"""

NodeVariableValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per node of a layer and variable of the circuit, shape (#nodes, #variables of
the circuit).
"""

NodeIntervals: TypeAlias = npt.NDArray[np.float64]
"""
The lower and upper bound of one interval per node of a layer, shape (#nodes, 2).
"""

NodeIntervalBounds: TypeAlias = npt.NDArray[np.int64]
"""
Whether the lower and upper bound of one interval per node of a layer are open or
closed, as :class:`random_events.interval.Bound` values of shape (#nodes, 2).
"""

NodeScopeValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per node of a layer and variable in the scope of the layer, shape (#nodes,
#variables of the layer).
"""

NodeScopeMatrices: TypeAlias = npt.NDArray[np.float64]
"""
One square matrix over the variables in the scope of a layer per node of the layer,
shape (#nodes, #variables of the layer, #variables of the layer).
"""

NodeScopeLowerTriangles: TypeAlias = npt.NDArray[np.float64]
"""
The entries on and below the diagonal of one square matrix over the variables in the
scope of a layer per node of the layer, row by row, shape (#nodes, n * (n + 1) // 2) for
n variables of the layer.
"""

NodeScopeIntervals: TypeAlias = npt.NDArray[np.float64]
"""
The lower and upper bound of one interval per node of a layer and variable in its scope,
shape (#nodes, #variables of the layer, 2).
"""

NodeScopeIntervalBounds: TypeAlias = npt.NDArray[np.int64]
"""
Whether the bounds of a :data:`NodeScopeIntervals` array are open or closed, as
:class:`random_events.interval.Bound` values of shape (#nodes, #variables of the layer,
2).
"""

ScopeValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per variable of a subset of the scope of a layer, shape (#those variables,).
"""

SampleScopeValues: TypeAlias = npt.NDArray[np.float64]
"""
The values of the variables in the scope of a layer in a set of samples, shape
(#samples, #variables of the layer).
"""

VariableValues: TypeAlias = npt.NDArray[np.number]
"""
One number per variable of a circuit, shape (#variables of the circuit,).
"""

VariableMask: TypeAlias = npt.NDArray[np.bool_]
"""
One flag per variable of a circuit, shape (#variables of the circuit,).
"""

VariableIndices: TypeAlias = npt.NDArray[np.int64]
"""
Indices of variables of a circuit.
"""

EdgeValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per edge of an inner layer, in the order the layer stores its edges.
"""

EdgeMask: TypeAlias = npt.NDArray[np.bool_]
"""
One flag per edge of an inner layer, in the order the layer stores its edges.
"""

States: TypeAlias = npt.NDArray[np.int64]
"""
The states of a discrete variable, sorted ascending: the hash of the domain element for
a symbolic variable and the value itself for an integer variable.
"""

StateIndices: TypeAlias = npt.NDArray[np.int64]
"""
Indices into the states of a discrete variable.
"""

StateMask: TypeAlias = npt.NDArray[np.bool_]
"""
One flag per state of a discrete variable.
"""

NodeStateValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per node of a layer and state of a discrete variable, shape (#nodes, #states).
"""

StateValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per state of a discrete variable, shape (#states,), or k values per state,
shape (#states, k).
"""

TableEntryValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per entry of a probability table, in the order the table lists its entries.
"""

"""
Names for the kinds of arrays that the layers of a layered circuit pass around.

They tell the reader what an array holds and which shape it has. The values numeric
queries take and return may be arrays of any backend; structure, and the values of the
parts that compute on numpy only, are numpy arrays on the host.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing_extensions import Any, TypeAlias

# %% values of any backend

BackendArray: TypeAlias = Any
"""
An array of any backend that implements the Python Array API standard, which defines no
type to annotate such an array with.
"""

SampleArray: TypeAlias = BackendArray
"""
Samples of all variables of a circuit, shape (#samples, #variables of the circuit).
"""

SampleColumn: TypeAlias = BackendArray
"""
The values of one variable in a set of samples, shape (#samples,).
"""

SampleValues: TypeAlias = BackendArray
"""
One value per sample, shape (#samples,).
"""

NodeValues: TypeAlias = BackendArray
"""
One value per node of a layer, shape (#nodes,).
"""

SampleNodeValues: TypeAlias = BackendArray
"""
One value per sample and node of a layer, shape (#samples, #nodes).
"""

SampleNodeMask: TypeAlias = BackendArray
"""
One flag per sample and node of a layer, shape (#samples, #nodes).
"""

NodeVariableValues: TypeAlias = BackendArray
"""
One value per node of a layer and variable of the circuit, shape (#nodes, #variables of
the circuit).
"""

NodeIntervals: TypeAlias = BackendArray
"""
The lower and upper bound of one interval per node of a layer, shape (#nodes, 2).
"""

EdgeValues: TypeAlias = BackendArray
"""
One value per edge of an inner layer, in the order of the edge matrix of the layer.
"""

NodeStateValues: TypeAlias = BackendArray
"""
One value per node of a layer and state of a discrete variable, shape (#nodes, #states).
"""

# %% numpy arrays on the host

SampleRows: TypeAlias = npt.NDArray[np.int64]
"""
Indices of rows of a :data:`SampleArray`.
"""

NodeIndices: TypeAlias = npt.NDArray[np.int64]
"""
Indices of nodes of a layer.
"""

NodeMask: TypeAlias = npt.NDArray[np.bool_]
"""
One flag per node of a layer, shape (#nodes,).
"""

NodeSelection: TypeAlias = NodeIndices | NodeMask
"""
Which nodes of a layer to keep: their indices, or one flag per node.
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

EdgeMask: TypeAlias = npt.NDArray[np.bool_]
"""
One flag per edge of an inner layer, in the order of the edge matrix of the layer.
"""

DenseEdgeValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per node of a layer and node of one of its child layers, shape (#nodes, #child
nodes): the edges between the two layers as a dense matrix.
"""

NodeColumnMask: TypeAlias = npt.NDArray[np.bool_]
"""
One flag per node of an inner layer and node of its child layers, shape (#nodes, total
number of nodes of the child layers).
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

StateValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per state of a discrete variable, shape (#states,), or k values per state,
shape (#states, k).
"""

TableEntryValues: TypeAlias = npt.NDArray[np.float64]
"""
One value per entry of a probability table, in the order the table lists its entries.
"""

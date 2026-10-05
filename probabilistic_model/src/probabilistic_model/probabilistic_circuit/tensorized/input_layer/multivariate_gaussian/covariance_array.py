from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
from typing_extensions import List, Self, Sequence

from probabilistic_model.distributions.multivariate_gaussian import Covariance
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
    NodeSelection,
    NodeScopeLowerTriangles,
    NodeScopeMatrices,
    NodeScopeValues,
)


@dataclass
class CovarianceArray:
    """
    The covariance matrices of the nodes of a layer, one per node and all over the same
    variables.

    A covariance matrix is symmetric, so only the entries on and below its diagonal are
    stored, the way
    :class:`~probabilistic_model.distributions.multivariate_gaussian.Covariance` stores
    a single matrix.
    """

    lower_triangles: NodeScopeLowerTriangles
    """
    The entries of every covariance matrix on and below its diagonal, row by row.
    """

    def __post_init__(self):
        self.lower_triangles = np.asarray(self.lower_triangles, dtype=float)
        self.validate()

    def validate(self):
        """
        :raises ShapeMismatchError: If the entries are not one lower triangle of a
            square matrix per row.
        """
        if self.lower_triangles.ndim != 2:
            raise ShapeMismatchError(self.lower_triangles.shape, (None, None))
        entries = self.dimension * (self.dimension + 1) // 2
        expected = (self.number_of_matrices, entries)
        if self.lower_triangles.shape != expected:
            raise ShapeMismatchError(self.lower_triangles.shape, expected)

    @classmethod
    def from_matrices(cls, matrices: NodeScopeMatrices) -> Self:
        """
        :param matrices: One covariance matrix per node, of which only the entries on
            and below the diagonal are read.
        :return: The covariances.
        :raises ShapeMismatchError: If the matrices are not square.
        """
        matrices = np.asarray(matrices, dtype=float)
        dimension = matrices.shape[-1]
        expected = (len(matrices), dimension, dimension)
        if matrices.shape != expected:
            raise ShapeMismatchError(matrices.shape, expected)
        rows, columns = np.tril_indices(dimension)
        return cls(matrices[:, rows, columns])

    @classmethod
    def from_covariances(cls, covariances: Sequence[Covariance]) -> Self:
        """
        :param covariances: Covariances of the same dimension.
        :return: The covariances, stacked in that order.
        """
        return cls(np.array([covariance.lower_triangle for covariance in covariances]))

    @property
    def number_of_matrices(self) -> int:
        """
        :return: How many covariance matrices there are.
        """
        return len(self.lower_triangles)

    @property
    def dimension(self) -> int:
        """
        :return: How many rows, and columns, every matrix has.
        """
        return int((math.isqrt(8 * self.lower_triangles.shape[1] + 1) - 1) // 2)

    @functools.cached_property
    def matrices(self) -> NodeScopeMatrices:
        """
        :return: The full, symmetric matrices.
        """
        rows, columns = np.tril_indices(self.dimension)
        matrices = np.empty((self.number_of_matrices, self.dimension, self.dimension))
        matrices[:, rows, columns] = self.lower_triangles
        matrices[:, columns, rows] = self.lower_triangles
        return matrices

    @property
    def variances(self) -> NodeScopeValues:
        """
        :return: The diagonal of every matrix.
        """
        indices = np.arange(self.dimension)
        return self.lower_triangles[:, indices * (indices + 1) // 2 + indices]

    def covariance_at(self, index: int) -> Covariance:
        """
        :param index: The index of a matrix.
        :return: That covariance on its own.
        """
        return Covariance(self.lower_triangles[index].copy())

    def between(self, rows: NodeIndices, columns: NodeIndices) -> np.ndarray:
        """
        :param rows: Indices of rows.
        :param columns: Indices of columns.
        :return: The entries of every matrix at those rows and columns, with shape
            (#matrices, #rows, #columns).
        """
        return self.matrices[:, rows][:, :, columns]

    def marginal(self, indices: NodeIndices) -> Self:
        """
        :param indices: The rows and columns to keep, in the order to keep them in.
        :return: The covariances of only those.
        """
        return self.from_matrices(self.between(indices, indices))

    def scaled(self, factors: np.ndarray) -> Self:
        """
        :param factors: What to multiply each index by.
        :return: The covariances after every index is multiplied by its factor, which
            scales each entry once per index it relates.
        """
        rows, columns = np.tril_indices(self.dimension)
        return type(self)(self.lower_triangles * factors[rows] * factors[columns])

    def select(self, indices: NodeSelection) -> Self:
        """
        :param indices: A mask or index array over the matrices.
        :return: The covariances of only those matrices.
        """
        return type(self)(self.lower_triangles[indices])

    @classmethod
    def concatenate(cls, arrays: List[Self]) -> Self:
        """
        :param arrays: Covariances of the same dimension.
        :return: All of their matrices, in order.
        """
        return cls(np.concatenate([array.lower_triangles for array in arrays]))

    def copy(self) -> Self:
        """
        :return: The same covariances, sharing no array with these.
        """
        return type(self)(self.lower_triangles.copy())

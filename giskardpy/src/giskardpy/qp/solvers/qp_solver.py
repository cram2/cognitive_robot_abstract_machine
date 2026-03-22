from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import numpy as np
from typing_extensions import TypeVar, Generic, get_args

from giskardpy.qp.qp_data import QPDataExplicit, QPDataTwoSidedInequality

T = TypeVar("T", QPDataExplicit, QPDataTwoSidedInequality)


@dataclass
class QPSolver(Generic[T]):

    @classmethod
    @property
    def qp_data_type(cls) -> Type[T]:
        """
        The semDT type for which this converter handles conversion.
        """
        return get_args(cls.__orig_bases__[0])[0]

    def solver_call(self, qp_data: T) -> np.ndarray:
        raise NotImplementedError()

    def solver_call_explicit_interface(self, qp_data: QPDataExplicit) -> np.ndarray:
        """
        min_x 0.5 x^T H x + g^T x
        s.t.  lb <= x <= ub     (box constraints)
                   Ex <= bE     (equality constraints)
            lbA <= Ax <= ubA    (lower/upper inequality constraints)
        """
        raise NotImplementedError()

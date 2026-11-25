"""
Comparison of the results a query computes on an array backend with those numpy
computes.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
from array_api_compat import array_namespace
from typing_extensions import Any

# %% comparison with numpy


def assert_result_of_backend_equals(
    result: Any, expected: np.ndarray, namespace: ModuleType
) -> None:
    """
    :param result: The result of a query of a layer with parameters of a backend.
    :param expected: The result of the same query of the layer with numpy parameters.
    :param namespace: The array namespace of the backend.
    """
    assert array_namespace(result) is array_namespace(namespace.asarray(0.0))
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-12)

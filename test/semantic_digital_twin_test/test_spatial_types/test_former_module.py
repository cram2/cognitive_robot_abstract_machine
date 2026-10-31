from __future__ import annotations

import inspect

import pytest

import krrood_spatial_math.exceptions
import krrood_spatial_math.spatial_types
import semantic_digital_twin.exceptions
import semantic_digital_twin.spatial_types.spatial_types

# %% names kept importable from where the spatial types used to live


@pytest.mark.parametrize(
    "name",
    [
        name
        for name, value in vars(krrood_spatial_math.spatial_types).items()
        if not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and value.__module__ == krrood_spatial_math.spatial_types.__name__
    ],
)
def test_a_spatial_type_is_importable_from_its_former_module(name):
    assert (
        vars(semantic_digital_twin.spatial_types.spatial_types)[name]
        is vars(krrood_spatial_math.spatial_types)[name]
    )


@pytest.mark.parametrize(
    "exception",
    [
        krrood_spatial_math.exceptions.SpatialTypesError,
        krrood_spatial_math.exceptions.InsufficientVectorsError,
        krrood_spatial_math.exceptions.ReferenceFrameMismatchError,
    ],
)
def test_a_spatial_types_exception_is_importable_from_its_former_module(exception):
    assert vars(semantic_digital_twin.exceptions)[exception.__name__] is exception

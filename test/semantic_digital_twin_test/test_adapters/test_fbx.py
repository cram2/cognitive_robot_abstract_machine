import numpy as np
import pytest

from semantic_digital_twin.adapters.exceptions import (
    UnexpectedArrayShape,
    UnsupportedFBXAxis,
)
from semantic_digital_twin.adapters.fbx import CoordinateAxis, FBXParser


def test_axis_from_fbx_settings():
    assert CoordinateAxis.from_fbx(2, -1) == CoordinateAxis.NEGATIVE_Z


@pytest.mark.parametrize("axis_index, sign", [(3, 1), (0, 0)])
def test_axis_that_does_not_exist_is_rejected(axis_index, sign):
    with pytest.raises(UnsupportedFBXAxis):
        CoordinateAxis.from_fbx(axis_index, sign)


def test_vertices_that_are_not_three_dimensional_are_rejected():
    vertices = np.zeros((4, 2))
    with pytest.raises(UnexpectedArrayShape) as error:
        FBXParser.transform_vertices(vertices, np.eye(4))
    assert error.value.actual_shape == list(vertices.shape)


def test_transformation_that_is_not_homogeneous_is_rejected():
    transformation = np.eye(3)
    with pytest.raises(UnexpectedArrayShape) as error:
        FBXParser.transform_vertices(np.zeros((4, 3)), transformation)
    assert error.value.actual_shape == list(transformation.shape)

import pytest

from krrood.symbolic_math.exceptions import (
    SymbolicMathExpressionNotRegisteredError,
    NoFreeVariablesError,
    SymbolicMathExpressionAlreadyRegisteredError,
    FloatVariableAlreadyHasResolveError,
)
from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable, Vector
import numpy as np


def test_add_variable():
    data = FloatVariableData()
    v1 = FloatVariable("v1")
    v2 = FloatVariable("v2")
    data.register_expression(v1)
    data.register_expression(v2)
    assert len(data.variables) == 2
    data.set_value(v1, 1.0)
    assert data.data[0] == 1.0 == v1.evaluate()
    assert data.data[1] == 0.0 == v2.evaluate()
    data.set_value(v2, 2.0)
    assert data.data[0] == 1.0 == v1.evaluate()
    assert data.data[1] == 2.0 == v2.evaluate()


def test_add_variable_twice():
    data = FloatVariableData()
    v1 = FloatVariable("v1")
    data.register_expression(v1)
    with pytest.raises(SymbolicMathExpressionAlreadyRegisteredError):
        data.register_expression(v1)
    assert len(data.variables) == 1
    data.set_value(v1, 1.0)
    assert data.data[0] == 1.0 == v1.evaluate()


def test_add_variable_that_has_resolve():
    data1 = FloatVariableData()
    v1 = FloatVariable("v1")
    v1.resolve = lambda: 1
    with pytest.raises(FloatVariableAlreadyHasResolveError):
        data1.register_expression(v1)
    assert len(data1.variables) == 0


def test_add_vector():
    data = FloatVariableData()
    vector = Vector([FloatVariable("v1"), FloatVariable("v2")])
    data.register_expression(vector)
    assert len(data.variables) == 2
    data.set_value(vector, [1.0, 2.0])
    assert np.allclose(vector.evaluate(), [1.0, 2.0])
    assert np.allclose(data.data, vector.evaluate())


def test_update_non_managed_expression():
    data = FloatVariableData()
    vector = Vector([FloatVariable("v1"), FloatVariable("v2")])
    vector2 = Vector([FloatVariable("v3"), FloatVariable("v4")])
    data.register_expression(vector)
    assert len(data.variables) == 2
    with pytest.raises(SymbolicMathExpressionNotRegisteredError):
        data.set_value(vector2, [1.0, 2.0])
    assert np.allclose(vector.evaluate(), [0.0, 0.0])
    assert np.allclose(data.data, vector.evaluate())


def test_register_constant_expression():
    data = FloatVariableData()
    vector = Vector([1.0, 2.0])
    with pytest.raises(NoFreeVariablesError):
        data.register_expression(vector)
    assert len(data.data) == 0

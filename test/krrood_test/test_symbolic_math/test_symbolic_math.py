import operator

import casadi as ca
import numpy as np
import pytest
import scipy

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.exceptions import (
    HasFreeVariablesError,
    NotSquareMatrixError,
)
from krrood.symbolic_math.symbolic_math import VariableParameters
from test.krrood_test.test_symbolic_math.reference_implementations import (
    normalize_angle_positive,
    shortest_angular_distance,
    normalize_angle,
)

TrinaryTrue = 1
TrinaryFalse = 0
TrinaryUnknown = 0.5

bool_values = [True, False]
numbers = [-69, 23]
quaternions = [np.array([1.0, 0, 0, 0]), np.array([0.0, 1, 0, 0])]


def logic_not(a):
    if a == TrinaryTrue:
        return TrinaryFalse
    elif a == TrinaryFalse:
        return TrinaryTrue
    elif a == TrinaryUnknown:
        return TrinaryUnknown
    else:
        raise ValueError(f"Invalid truth value: {a}")


def logic_and(a, b):
    if a == TrinaryFalse or b == TrinaryFalse:
        return TrinaryFalse
    elif a == TrinaryTrue and b == TrinaryTrue:
        return TrinaryTrue
    elif a == TrinaryUnknown or b == TrinaryUnknown:
        return TrinaryUnknown
    else:
        raise ValueError(f"Invalid truth values: {a}, {b}")


def logic_or(a, b):
    if a == TrinaryTrue or b == TrinaryTrue:
        return TrinaryTrue
    elif a == TrinaryFalse and b == TrinaryFalse:
        return TrinaryFalse
    elif a == TrinaryUnknown or b == TrinaryUnknown:
        return TrinaryUnknown
    else:
        raise ValueError(f"Invalid truth values: {a}, {b}")


def test_to_sx():
    v = sm.FloatVariable(name="muh")
    v_sx = v.casadi_sx
    # constants
    assert ca.is_equal(sm.to_sx(1), ca.SX(1))
    assert ca.is_equal(sm.to_sx(np.array([1, 2])), ca.SX([1, 2]))
    assert ca.is_equal(sm.to_sx([1, 2]), ca.SX([1, 2]))
    assert ca.is_equal(sm.to_sx([ca.SX(1), 2]), ca.SX([1, 2]))
    assert ca.is_equal(sm.to_sx(sm.Vector([1, 2, 3])), ca.SX([1, 2, 3]))

    # with variable
    assert ca.is_equal(sm.to_sx(v), v_sx)

    sx_list = sm.to_sx([v, 2])
    assert ca.is_equal(sx_list[0], v_sx)
    assert ca.is_equal(sx_list[1], 2)

    sx_list = sm.to_sx([v_sx, 2])
    assert ca.is_equal(sx_list[0], v_sx)
    assert ca.is_equal(sx_list[1], 2)

    sx_list = sm.to_sx(sm.Vector([v, 2, 3]))
    assert ca.is_equal(sx_list[0], v_sx)
    assert ca.is_equal(sx_list[1], 2)
    assert ca.is_equal(sx_list[2], 3)


class TestLogic3:
    values = [
        TrinaryTrue,
        TrinaryFalse,
        TrinaryUnknown,
    ]

    def test_and3(self):
        for i in self.values:
            for j in self.values:
                expected = logic_and(i, j)
                actual = sm.trinary_logic_and(sm.Scalar(i), sm.Scalar(j))
                assert (
                    expected == actual
                ), f"a={i}, b={j}, expected {expected}, actual {actual}"

    def test_or3(self):
        for i in self.values:
            for j in self.values:
                expected = logic_or(i, j)
                actual = sm.trinary_logic_or(sm.Scalar(i), sm.Scalar(j))
                assert expected == float(
                    actual
                ), f"a={i}, b={j}, expected {expected}, actual {actual}"

    def test_not3(self):
        for i in self.values:
            expected = logic_not(i)
            actual = sm.trinary_logic_not(sm.Scalar(i))
            assert expected == actual, f"a={i}, expected {expected}, actual {actual}"

    def test_trinary_logic_to_str(self):
        a = sm.FloatVariable(name="a")
        b = sm.FloatVariable(name="b")
        c = sm.FloatVariable(name="c")
        expression = sm.trinary_logic_and(
            a, sm.trinary_logic_or(b, sm.trinary_logic_not(c))
        )
        expression_str = sm.trinary_logic_to_str(expression)
        assert expression_str == '("a" and ("b" or not "c"))'

        const_expr = sm.trinary_logic_and(
            sm.Scalar.const_true(),
            sm.trinary_logic_or(a, sm.Scalar.const_trinary_unknown()),
        )
        const_expr_str = sm.trinary_logic_to_str(const_expr)
        assert const_expr_str == '("a" or Unknown)'


class TestIfElse:
    def test_if_one_arg(self):
        inputs = [
            (sm.FloatVariable(name="muh"), sm.FloatVariable(name="muh2")),
            (sm.Scalar(1), sm.Scalar(12)),
        ]
        if_functions = [
            sm.if_else,
            sm.if_eq_zero,
            sm.if_greater_eq_zero,
            sm.if_greater_zero,
        ]
        c = sm.FloatVariable(name="c")
        for if_result, else_result in inputs:
            for if_function in if_functions:
                result = if_function(c, if_result, else_result)
                result_type = type(result)
                if isinstance(if_result, sm.FloatVariable):
                    assert result_type == sm.Scalar
                    continue
                assert isinstance(
                    result, result_type
                ), f"{type(result)} != {result_type} for {if_function}"

    def test_if_two_arg(self):
        inputs = [
            (sm.FloatVariable(name="muh"), sm.FloatVariable(name="muh2")),
            (sm.Scalar(1), sm.Scalar(12)),
        ]
        if_functions = [
            sm.if_eq,
            sm.if_greater,
            sm.if_greater_eq,
            sm.if_less,
            sm.if_less_eq,
        ]
        a = sm.FloatVariable(name="a")
        b = sm.FloatVariable(name="b")
        for if_result, else_result in inputs:
            for if_function in if_functions:
                result = if_function(a, b, if_result, else_result)
                result_type = type(result)
                if isinstance(if_result, sm.FloatVariable):
                    assert result_type == sm.Scalar
                    continue
                assert isinstance(
                    result, result_type
                ), f"{type(result)} != {result_type} for {if_function}"

    @pytest.mark.parametrize("condition", bool_values)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_greater_zero(self, condition, if_result, else_result):
        assert np.allclose(
            sm.if_greater_zero(condition, if_result, else_result),
            float(if_result if condition > 0 else else_result),
        )

    @pytest.mark.parametrize("condition", bool_values)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_greater_eq_zero(self, condition, if_result, else_result):
        assert np.allclose(
            sm.if_greater_eq_zero(condition, if_result, else_result),
            float(if_result if condition >= 0 else else_result),
        )

    @pytest.mark.parametrize("condition", bool_values)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_eq_zero(self, condition, if_result, else_result):
        assert np.allclose(
            sm.if_eq_zero(condition, if_result, else_result),
            float(if_result if condition == 0 else else_result),
        )

    @pytest.mark.parametrize("a", numbers)
    @pytest.mark.parametrize("b", numbers)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_greater_eq(self, a, b, if_result, else_result):
        assert np.allclose(
            sm.if_greater_eq(a, b, if_result, else_result),
            float(if_result if a >= b else else_result),
        )

    @pytest.mark.parametrize("a", numbers)
    @pytest.mark.parametrize("b", numbers)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_less_eq(self, a, b, if_result, else_result):
        assert np.allclose(
            sm.if_less_eq(a, b, if_result, else_result),
            float(if_result if a <= b else else_result),
        )

    @pytest.mark.parametrize("a", numbers)
    @pytest.mark.parametrize("b", numbers)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_eq(self, a, b, if_result, else_result):
        assert np.allclose(
            sm.if_eq(a, b, if_result, else_result),
            float(if_result if a == b else else_result),
        )

    @pytest.mark.parametrize("a", numbers)
    @pytest.mark.parametrize("b", numbers)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_greater(self, a, b, if_result, else_result):
        assert np.allclose(
            sm.if_greater(a, b, if_result, else_result),
            float(if_result if a > b else else_result),
        )

    @pytest.mark.parametrize("a", numbers)
    @pytest.mark.parametrize("b", numbers)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_less(self, a, b, if_result, else_result):
        assert np.allclose(
            sm.if_less(a, b, if_result, else_result),
            float(if_result if a < b else else_result),
        )

    @pytest.mark.parametrize("a", [1, 3, 4, -1, 0.5, -0.5, 0])
    def test_if_eq_cases(self, a):
        b_result_cases = [
            (1, sm.Scalar(1)),
            (3, sm.Scalar(3)),
            (4, sm.Scalar(4)),
            (-1, sm.Scalar(-1)),
            (0.5, sm.Scalar(0.5)),
            (-0.5, sm.Scalar(-0.5)),
        ]

        def reference(a_, b_result_cases_, else_result):
            for b, if_result in b_result_cases_:
                if a_ == b:
                    return if_result.to_np()[0]
            return else_result

        actual = sm.if_eq_cases(a, b_result_cases, sm.Scalar(0))
        expected = float(reference(a, b_result_cases, 0))
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("a", numbers)
    def test_if_eq_cases_set(self, a):
        b_result_cases = [
            (1, sm.Scalar(1)),
            (3, sm.Scalar(3)),
            (4, sm.Scalar(4)),
            (-1, sm.Scalar(-1)),
            (0.5, sm.Scalar(0.5)),
            (-0.5, sm.Scalar(-0.5)),
        ]

        def reference(a_, b_result_cases_, else_result):
            for b, if_result in b_result_cases_:
                if a_ == b:
                    return if_result.to_np()[0]
            return else_result

        actual = sm.if_eq_cases(a, b_result_cases, sm.Scalar(0))
        expected = float(reference(a, b_result_cases, 0))
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("a", numbers)
    def test_if_less_eq_cases(self, a):
        b_result_cases = [
            (-1, sm.Scalar(-1)),
            (-0.5, sm.Scalar(-0.5)),
            (0.5, sm.Scalar(0.5)),
            (1, sm.Scalar(1)),
            (3, sm.Scalar(3)),
            (4, sm.Scalar(4)),
        ]

        def reference(a_, b_result_cases_, else_result):
            for b, if_result in b_result_cases_:
                if a_ <= b:
                    return if_result.to_np()[0]
            return else_result

        assert np.allclose(
            sm.if_less_eq_cases(a, b_result_cases, sm.Scalar(0)),
            float(reference(a, b_result_cases, 0)),
        )


class TestFloatVariable:
    def test_bool_casting(self):
        v = sm.FloatVariable(name="v")
        with pytest.raises(HasFreeVariablesError):
            bool(v)

    def test_float_casting(self):
        v = sm.FloatVariable(name="v")
        with pytest.raises(HasFreeVariablesError):
            # noinspection PyTypeChecker
            float(v)

    def test_arithmetic_operations(self):
        operators = [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.pow,
            operator.floordiv,
            operator.mod,
        ]
        s1 = sm.FloatVariable(name="muh")
        s2 = sm.FloatVariable(name="kikariki")
        for op in operators:
            result = op(s1, s2)
            assert isinstance(result, sm.Scalar), f"{op.__name__} result is not Scalar"
            f = result.compile()
            assert f.call_with_kwargs(muh=1, kikariki=2) == op(
                1, 2
            ), f"{op.__name__} result is wrong"

    def test_comparison_operations(self):
        operators = [
            operator.eq,
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
        ]
        s1 = sm.FloatVariable(name="muh")
        s2 = sm.FloatVariable(name="kikariki")
        for op in operators:
            result = op(s1, s2)
            assert isinstance(result, sm.Scalar), f"{op.__name__} result is not Scalar"
            f = result.compile()
            assert f.call_with_kwargs(muh=1, kikariki=2) == op(
                1, 2
            ), f"{op.__name__} result is wrong"

    def test_back_reference(self):
        v = sm.FloatVariable(name="asdf")
        v2 = v.free_variables()[0]
        assert id(v2) == id(v)
        assert id(v2.casadi_sx) == id(v.casadi_sx)

        v3 = sm.Scalar(v).free_variables()[0]
        assert id(v3) == id(v)

    def test_float_variable_unique(self):
        v1 = sm.FloatVariable(name="asdf")
        v2 = sm.FloatVariable(name="asdf")
        e = v1 + v2
        e.compile(VariableParameters.from_lists([v1, v2]))

    def test_from_name(self):
        s = sm.FloatVariable(name="muh")
        assert isinstance(s, sm.FloatVariable)
        assert str(s) == "muh"

    def test_to_np(self):
        s1 = sm.FloatVariable(name="s1")
        with pytest.raises(HasFreeVariablesError):
            s1.to_np()

    def test_hash(self):
        s = sm.FloatVariable(name="muh")
        d = {s: 1}
        assert d[s] == 1


class TestExpression:

    def test_free_variables(self):
        m = sm.Vector(sm.create_float_variables(["a", "b", "c", "d"]))
        assert len(m.free_variables()) == 4
        a = sm.FloatVariable(name="a")
        assert a.equivalent(a.free_variables()[0])

    def test_pretty_str(self):
        e = sm.Matrix.eye(4)
        e.pretty_str()

    def test_to_np(self):
        e = sm.Scalar(1)
        assert np.allclose(e.to_np(), np.array([1]))
        e = sm.Vector([1, 2])
        assert np.allclose(e.to_np(), np.array([1, 2]))
        e = sm.Matrix([[1, 2], [3, 4]])
        assert np.allclose(e.to_np(), np.array([[1, 2], [3, 4]]))

    def test_to_np_fail(self):
        s1, s2 = sm.FloatVariable(name="s1"), sm.FloatVariable(name="s2")
        e = s1 + s2
        with pytest.raises(HasFreeVariablesError):
            e.to_np()

    def test_jacobian(self):
        a = sm.FloatVariable(name="a")
        b = sm.FloatVariable(name="b")
        m = sm.Vector(data=[a + b, a**2, b**2])
        jac = m.jacobian([a, b])
        expected = sm.Matrix([[1, 1], [2 * a, 0], [0, 2 * b]])
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                assert jac[i, j].equivalent(expected[i, j])

    def test_jacobian_dot(self):
        a, ad, b, bd = 1.0, 2.0, 3.0, 4.0
        kwargs = {
            "a": a,
            "ad": ad,
            "b": b,
            "bd": bd,
        }
        a_s = sm.FloatVariable(name="a")
        ad_s = sm.FloatVariable(name="ad")
        b_s = sm.FloatVariable(name="b")
        bd_s = sm.FloatVariable(name="bd")
        m = sm.Vector(
            data=[
                a_s**3 * b_s**3,
                -a_s * sm.cos(b_s),
            ]
        )
        jac = m.jacobian_dot([a_s, b_s], [ad_s, bd_s])
        expected_expr = sm.Matrix(
            data=[
                [
                    6 * ad_s * a_s * b_s**3 + 9 * a_s**2 * bd_s * b_s**2,
                    9 * ad_s * a_s**2 * b_s**2 + 6 * a_s**3 * bd_s * b,
                ],
                [bd_s * sm.sin(b_s), ad_s * sm.sin(b_s) + a_s * bd_s * sm.cos(b_s)],
            ]
        )
        actual = jac.compile().call_with_kwargs(**kwargs)
        expected = expected_expr.compile().call_with_kwargs(**kwargs)
        assert np.allclose(actual, expected)

    def test_jacobian_ddot(self):
        a, ad, add, b, bd, bdd = 1, 2, 3, 4, 5, 6
        kwargs = {
            "a": a,
            "ad": ad,
            "add": add,
            "b": b,
            "bd": bd,
            "bdd": bdd,
        }
        a_s = sm.FloatVariable(name="a")
        ad_s = sm.FloatVariable(name="ad")
        add_s = sm.FloatVariable(name="add")
        b_s = sm.FloatVariable(name="b")
        bd_s = sm.FloatVariable(name="bd")
        bdd_s = sm.FloatVariable(name="bdd")
        m = sm.Vector(
            [
                a_s**3 * b_s**3,
                b_s**2,
                -a_s * sm.cos(b_s),
            ]
        )
        jac = m.jacobian_ddot([a_s, b_s], [ad_s, bd_s], [add_s, bdd_s])
        expected = np.array(
            [
                [
                    add * 6 * b**3 + bdd * 18 * a**2 * b + 2 * ad * bd * 18 * a * b**2,
                    bdd * 6 * a**3 + add * 18 * b**2 * a + 2 * ad * bd * 18 * b * a**2,
                ],
                [0, 0],
                [bdd * np.cos(b), bdd * -a * np.sin(b) + 2 * ad * bd * np.cos(b)],
            ]
        )
        actual = jac.compile().call_with_kwargs(**kwargs)
        assert np.allclose(actual, expected)

    def test_total_derivative2(self):
        a, b, ad, bd, add, bdd = 1, 2, 3, 4, 5, 6
        kwargs = {
            "a": a,
            "ad": ad,
            "add": add,
            "b": b,
            "bd": bd,
            "bdd": bdd,
        }
        a_s = sm.FloatVariable(name="a")
        ad_s = sm.FloatVariable(name="ad")
        add_s = sm.FloatVariable(name="add")
        b_s = sm.FloatVariable(name="b")
        bd_s = sm.FloatVariable(name="bd")
        bdd_s = sm.FloatVariable(name="bdd")
        m = a_s * b_s**2
        jac = m.second_order_total_derivative([a_s, b_s], [ad_s, bd_s], [add_s, bdd_s])
        actual = jac.compile().call_with_kwargs(**kwargs)
        expected = bdd * 2 * a + 2 * ad * bd * 2 * b
        assert np.allclose(actual, expected)

    def test_total_derivative2_2(self):
        a, b, c, ad, bd, cd, add, bdd, cdd = 1, 2, 3, 4, 5, 6, 7, 8, 9
        kwargs = {
            "a": a,
            "ad": ad,
            "add": add,
            "b": b,
            "bd": bd,
            "bdd": bdd,
            "c": c,
            "cd": cd,
            "cdd": cdd,
        }
        a_s = sm.FloatVariable(name="a")
        ad_s = sm.FloatVariable(name="ad")
        add_s = sm.FloatVariable(name="add")
        b_s = sm.FloatVariable(name="b")
        bd_s = sm.FloatVariable(name="bd")
        bdd_s = sm.FloatVariable(name="bdd")
        c_s = sm.FloatVariable(name="c")
        cd_s = sm.FloatVariable(name="cd")
        cdd_s = sm.FloatVariable(name="cdd")
        m = data = a_s * b_s**2 * c_s**3
        jac = m.second_order_total_derivative(
            [a_s, b_s, c_s], [ad_s, bd_s, cd_s], [add_s, bdd_s, cdd_s]
        )
        actual = jac.compile().call_with_kwargs(**kwargs)
        expected = (
            bdd * 2 * a * c**3
            + cdd * 6 * a * b**2 * c
            + 4 * ad * bd * b * c**3
            + 6 * ad * b**2 * cd * c**2
            + 12 * a * bd * b * cd * c**2
        )
        assert np.allclose(actual, expected)


class TestSymbolicType:
    def test_substitute(self):
        a, b, c, d = sm.create_float_variables(["a", "b", "c", "d"])
        expr = a * b
        expr_substituted = expr.substitute([a, b], [c, d])
        assert expr_substituted == c * d

    def test_create_variables(self):
        result = sm.create_float_variables(["a", "b", "c"])
        assert str(result[0]) == "a"
        assert str(result[1]) == "b"
        assert str(result[2]) == "c"

    def test_create_variables2(self):
        result = sm.create_float_variables(3)
        assert str(result[0]) == "s_0"
        assert str(result[1]) == "s_1"
        assert str(result[2]) == "s_2"

    def test_to_str(self):
        expr = sm.FloatVariable(name="muh") * sm.Scalar(23)
        assert expr.pretty_str() == [["(23*muh)"]]


class TestCompiledFunction:

    def test_empty_compiled_function(self):
        expected = np.array([1, 2, 3])
        e = sm.Vector(expected)
        f = e.compile(sparse=False)
        assert np.allclose(f(), expected)
        assert np.allclose(f(np.array([], dtype=float)), expected)

    def test_empty_compiled_function_sparse(self):
        expected = np.array([1, 2, 3], ndmin=2)
        e = sm.Matrix(expected)
        f = e.compile(sparse=True)
        assert np.allclose(f().toarray(), expected)
        assert np.allclose(f(np.array([], dtype=float)).toarray(), expected)

    def test_dense(self):
        s1_value = 420.0
        s2_value = 69.0
        s1, s2 = sm.create_float_variables(["s1", "s2"])
        e = sm.sqrt(sm.cos(s1) + sm.sin(s2))
        e_f = e.compile()
        actual = e_f(np.array([s1_value, s2_value]))
        expected = np.sqrt(np.cos(s1_value) + np.sin(s2_value))
        assert np.allclose(actual, expected)

    def test_dense_two_params(self):
        s1_value = 420.0
        s2_value = 69.0
        s1, s2 = sm.create_float_variables(["s1", "s2"])
        e = sm.sqrt(sm.cos(s1) + sm.sin(s2))
        e_f = e.compile(parameters=VariableParameters.from_lists([s1], [s2]))
        actual = e_f(np.array([s1_value]), np.array([s2_value]))
        expected = np.sqrt(np.cos(s1_value) + np.sin(s2_value))
        assert np.allclose(actual, expected)

    def test_sparse(self):
        s1_value = 420.0
        s2_value = 69.0
        s1, s2 = sm.create_float_variables(["s1", "s2"])
        e = sm.sqrt(sm.cos(s1) + sm.sin(s2))
        e_f = e.compile(sparse=True)
        actual = e_f(np.array([s1_value, s2_value]))
        assert isinstance(actual, scipy.sparse.csc_matrix)
        expected = np.sqrt(np.cos(s1_value) + np.sin(s2_value))
        assert np.allclose(actual.toarray(), expected)

    def test_stacked_compiled_function_dense(self):
        s1_value = 420.0
        s2_value = 69.0
        s1, s2 = sm.create_float_variables(["s1", "s2"])
        e1 = sm.sqrt(sm.cos(s1) + sm.sin(s2))
        e2 = s1 + s2
        e_f = sm.CompiledFunctionWithViews(
            expressions=[e1, e2], parameters=VariableParameters.from_lists([s1, s2])
        )
        actual_e1, actual_e2 = e_f(np.array([s1_value, s2_value]))
        expected_e1 = np.sqrt(np.cos(s1_value) + np.sin(s2_value))
        expected_e2 = s1_value + s2_value
        assert np.allclose(actual_e1, expected_e1)
        assert np.allclose(actual_e2, expected_e2)

    def test_single_args(self):
        size = 10_000
        variables = sm.create_float_variables([str(i) for i in range(size)])
        expr = sm.sum(*variables)
        f = expr.compile()
        for i in range(10):
            data = np.random.rand(size)
            assert np.isclose(f(data), np.sum(data))

    def test_single_args_with_bind(self):
        size = 10_000
        data = np.random.rand(size)
        variables = sm.create_float_variables([str(i) for i in range(size)])
        expr = sm.sum(*variables)
        f = expr.compile()
        f.bind_args_to_memory_view(0, data)
        for i in range(10):
            np.copyto(data, np.random.rand(size))
            assert np.isclose(f.evaluate(), np.sum(data))

    def test_multiple_args(self):
        size = 10_000
        n = 10
        element_size = size // n
        variables = sm.create_float_variables([str(i) for i in range(size)])
        expr = sm.sum(*variables)
        args = [variables[i * element_size : (i + 1) * element_size] for i in range(n)]
        f = expr.compile(parameters=VariableParameters.from_lists(*args))
        for i in range(100):
            args_values = [np.ones(element_size)] * n
            assert f(*args_values) == size

    def test_multiple_args_with_bind(self):
        size = 10_000
        n = 10
        element_size = size // n
        variables = sm.create_float_variables([str(i) for i in range(size)])
        expr = sm.sum(*variables)
        args = [variables[i * element_size : (i + 1) * element_size] for i in range(n)]
        f = expr.compile(parameters=VariableParameters.from_lists(*args))

        datas = []
        for i in range(n):
            datas.append(np.random.rand(element_size))
            f.bind_args_to_memory_view(i, datas[i])
        for i in range(100):
            for i in range(n):
                datas[i][:] = np.random.rand(element_size)
            assert np.isclose(f.evaluate(), np.sum(datas))

    def test_missing_free_variables(self):
        s1, s2 = sm.create_float_variables(["s1", "s2"])
        e = sm.sqrt(sm.cos(s1) + sm.sin(s2))
        with pytest.raises(HasFreeVariablesError):
            e.compile(parameters=VariableParameters.from_lists([s1]))


class TestScalar:
    def test_bool_casting(self):
        v = sm.FloatVariable(name="muh")
        v2 = sm.FloatVariable(name="kikariki")
        v3 = sm.FloatVariable(name="oink")
        false_ = sm.Scalar.const_false()
        true_ = sm.Scalar.const_true()
        assert not false_
        assert true_

        # "and" and "or" are smart and will simply const True/False away.
        with pytest.raises(HasFreeVariablesError):
            assert not (v and false_)

        # the == calls __eq__ which returns an expression.
        assert v == v
        assert v != v2

        # "in" is calling __eq__ which creates, e.g., v == v, bool casting of eq works and will return False for first eq and True for second
        assert v2 in [v, v2, v3]
        assert v not in [v2, v3]

        # const logical expressions can be evaluated
        assert sm.Scalar(10) > sm.Scalar(5)

        # this calls __bool__ on v and not __or__ and therefore raises exception
        with pytest.raises(HasFreeVariablesError):
            assert (v or v2) == v

        if (v | v2) == v:
            raise Exception("This should not happen")

    def test_arithmetic_operations(self):
        operators = [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.pow,
            operator.floordiv,
            operator.mod,
        ]
        s1 = sm.Scalar(1)
        s2 = sm.Scalar(2)
        f = 23.5
        for op in operators:
            result = op(s1, s2)
            assert isinstance(result, sm.Scalar), f"{op.__name__} result is not Scalar"
            assert result == op(1, 2), f"{op.__name__} result is wrong"

            result = op(s1, f)
            assert isinstance(result, sm.Scalar), f"{op.__name__} result is not Scalar"
            assert result == op(1, f), f"{op.__name__} result is wrong"

            result = op(f, s1)
            assert isinstance(result, sm.Scalar), f"{op.__name__} result is not Scalar"
            assert result == op(f, 1), f"{op.__name__} result is wrong"

    def test_comparisons(self):
        operators = [
            operator.lt,
            operator.le,
            operator.eq,
            operator.ge,
            operator.gt,
        ]
        f1 = 23
        f2 = 69
        e1_cas = sm.Scalar(23)
        e2_cas = sm.Scalar(69)
        for f in operators:
            r_np = f(f1, f2)
            r_cas = f(e1_cas, e2_cas)
            assert isinstance(r_cas, bool), f"{f.__name__} result is not Scalar"
            assert r_np == r_cas, f"{f.__name__} result is wrong"

    def test_comparisons_with_variable(self):
        operators = [
            operator.lt,
            operator.le,
            operator.eq,
            operator.ge,
            operator.gt,
        ]
        v = sm.FloatVariable(name="muh")
        e1_cas = sm.Scalar(23)
        for f in operators:
            r_cas = f(e1_cas, v)
            assert isinstance(r_cas, sm.Scalar), f"{f.__name__} result is not Scalar"

    def test_float_casting(self):
        assert float(sm.Scalar(1)) == 1.0

    def test_abs(self):
        f1 = sm.Scalar(-23)
        assert np.allclose(abs(f1), abs(float(f1)))
        assert isinstance(abs(f1), sm.Scalar)

    def test_max(self):
        f1 = sm.Scalar(23)
        f2 = sm.Scalar(69)
        assert np.allclose(sm.max(f1, f2), max(f1, f2))

    def test_min(self):
        f1 = sm.Scalar(23)
        f2 = sm.Scalar(69)
        assert np.allclose(sm.min(f1, f2), min(f1, f2))

    def test_negative(self):
        f1 = sm.Scalar(23)
        assert np.allclose(-f1, -float(f1))
        assert isinstance(-f1, sm.Scalar)

    def test_save_division(self):
        f1, f2 = 23, 69
        assert np.allclose(sm.Scalar(f1).safe_division(f2), f1 / f2 if f2 != 0 else 0)

    @pytest.mark.parametrize("x", [-23, 0, 23])
    def test_sign(self, x):
        actual = sm.sign(x)
        assert np.allclose(actual, np.sign(x))
        assert isinstance(actual, sm.Scalar)

    @pytest.mark.parametrize("x", [-23, 0, 23])
    def test_exp(self, x):
        actual = sm.exp(x)
        assert np.allclose(actual, np.exp(x))
        assert isinstance(actual, sm.Scalar)

    @pytest.mark.parametrize("x", [-23, 0, 23])
    def test_log(self, x):
        actual = sm.log(x)
        assert np.allclose(actual, np.log(x), equal_nan=True)
        assert isinstance(actual, sm.Scalar)

    @pytest.mark.parametrize("x", numbers)
    @pytest.mark.parametrize("lower_limit", numbers)
    @pytest.mark.parametrize("upper_limit", numbers)
    def test_limit(self, x, lower_limit, upper_limit):
        r1 = sm.limit(sm.Scalar(x), lower_limit, upper_limit)
        r2 = max(lower_limit, min(upper_limit, x))
        assert isinstance(r1, sm.Scalar)
        assert np.allclose(r1, r2)

    @pytest.mark.parametrize("a", numbers)
    @pytest.mark.parametrize("b", numbers)
    def test_fmod(self, a, b):
        ref_r = np.fmod(a, b)
        actual = sm.fmod(sm.Scalar(a), sm.Scalar(b))
        assert isinstance(actual, sm.Scalar)
        assert np.allclose(actual, ref_r, equal_nan=True)

    @pytest.mark.parametrize("a", numbers)
    def test_normalize_angle_positive(self, a):
        expected = normalize_angle_positive(a)
        actual = sm.normalize_angle_positive(a)
        assert isinstance(actual, sm.Scalar)
        assert np.allclose(
            shortest_angular_distance(actual.to_np(), expected),
            0.0,
        )

    @pytest.mark.parametrize("a", numbers)
    def test_normalize_angle(self, a):
        ref_r = normalize_angle(a)
        actual = sm.normalize_angle(a)
        assert isinstance(actual, sm.Scalar)
        assert np.allclose(actual, ref_r)

    @pytest.mark.parametrize("angle1", numbers)
    @pytest.mark.parametrize("angle2", numbers)
    def test_shorted_angular_distance(self, angle1, angle2):
        try:
            expected = shortest_angular_distance(angle1, angle2)
        except ValueError:
            expected = np.nan
        actual = sm.shortest_angular_distance(angle1, angle2)
        assert isinstance(actual, sm.Scalar)
        assert np.allclose(actual, expected, equal_nan=True)


class TestVector:
    def test_vector_to_list(self):
        data = np.array([1.0, 2.0, 3.5])
        v = sm.Vector(data)
        assert v.to_list() == data.tolist()

    def test_to_list_raises_on_variables(self):
        v = sm.Vector([sm.FloatVariable(name="a"), 2.0])
        with pytest.raises(HasFreeVariablesError):
            _ = v.to_list()

    def test_leq_on_array(self):
        a = sm.Vector(np.array([1, 2, 3, 4]))
        b = sm.Vector(np.array([2, 2, 2, 2]))
        assert not sm.logic_all(a <= b)

    def test_logic_any(self):
        a = sm.Vector(np.array([1, 2, 3, 4]))
        b = sm.Vector(np.array([2, 2, 2, 2]))
        result = sm.logic_any(a <= b)
        assert isinstance(result, sm.Scalar)
        assert result == 1

    def test_logic_all(self):
        a = sm.Vector(np.array([1, 2, 3, 4]))
        b = sm.Vector(np.array([2, 2, 2, 2]))
        result = sm.logic_all(a <= b)
        assert isinstance(result, sm.Scalar)
        assert result == 0

    def test_float_casting(self):
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            assert float(sm.Vector([1, 2]))

    def test_iterable(self):
        data = np.array([3, 1, 4, 1, 5])
        v = sm.Vector(data)
        iter_values = [float(x) for x in v]
        assert np.allclose(iter_values, list(data))
        assert all(isinstance(x, sm.Scalar) for x in v)

    def test_iterable_slice(self):
        data = np.array([10, 20, 30, 40])
        v = sm.Vector(data)
        s = v[1:3]
        iter_values = [float(x) for x in s]
        assert np.allclose(iter_values, list(data[1:3]))
        assert all(isinstance(x, sm.Scalar) for x in s)

    def test_arithmetic_operations(self):
        operators = [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.pow,
            operator.floordiv,
            operator.mod,
        ]
        v1_np = np.array([1, 2, 3])
        v2_np = np.array([3, 2, 1])
        s1_float = 1
        v1 = sm.Vector(v1_np)
        v2 = sm.Vector(v2_np)
        s1 = sm.Scalar(s1_float)

        for op in operators:
            result = op(v1, s1)
            expected = op(v1_np, s1_float)
            assert isinstance(result, sm.Vector), f"{op.__name__} result is not Vector"
            assert np.allclose(result, expected), f"{op.__name__} result is wrong"
            assert (
                result.shape[0] == expected.shape[0]
            ), f"{op.__name__} result shape is wrong"

            result = op(v1, v2)
            expected = op(v1_np, v2_np)
            assert isinstance(result, sm.Vector), f"{op.__name__} result is not Vector"
            assert np.allclose(result, expected), f"{op.__name__} result is wrong"
            assert (
                result.shape[0] == expected.shape[0]
            ), f"{op.__name__} result shape is wrong"

    def test_abs(self):
        f1 = sm.Vector([1, -1, -23])
        assert np.allclose(abs(f1), abs(f1.to_np()))
        assert isinstance(abs(f1), sm.Vector)

    def test_max(self):
        v = sm.Vector([23, 69])
        assert np.allclose(sm.max(v), max(v.to_np()))

    def test_min(self):
        v = sm.Vector([23, 69])
        assert np.allclose(sm.min(v), min(v.to_np()))

    def test_comparisons(self):
        operators = [
            operator.lt,
            operator.le,
            operator.eq,
            operator.ge,
            operator.gt,
        ]
        e1_np = np.array([1, 2, 3, -1])
        e1_cas = sm.Vector(e1_np)
        # todo test without float variable
        e2_cas = sm.Vector([1, 1, -1, sm.FloatVariable(name="muh")])
        for f in operators:
            r_cas = f(e1_cas, e2_cas)
            assert isinstance(r_cas, sm.Vector), f"{f.__name__} result is not Vector"

    def test_get_item(self):
        v = sm.Vector(np.array([1, 2, 3]))
        assert v[0] == 1
        assert isinstance(v[0], sm.Scalar)
        assert np.allclose(v[0:2], sm.Vector(np.array([1, 2])))
        assert isinstance(v[0:2], sm.Vector)

    def test_norm(self):
        v = np.array([1, 2, 3])
        actual = sm.Vector(data=v).norm()
        expected = np.linalg.norm(v)
        assert np.allclose(actual, expected)

    def test_len(self):
        assert len(sm.Vector([1, 2, 3, 4])) == 4

    def test_matmul(self):
        v = sm.Vector(np.array([1, 2]))
        v2 = sm.Vector(np.array([1, 2, 3]))
        m = sm.Matrix(np.array([[1, 2, 3], [4, 5, 6]]))

        expected = np.matmul(v2, v2)
        actual = v2 @ v2
        assert np.allclose(actual, expected)
        assert isinstance(actual, sm.Scalar)

        expected = np.matmul(v, m)
        actual = v @ m
        assert np.allclose(actual, expected)
        assert isinstance(actual, sm.Vector)

        expected = np.matmul(m, v2)
        actual = m @ v2
        assert np.allclose(actual, expected)
        assert isinstance(actual, sm.Vector)

    def test_matmul_vector_vector_dot(self):
        # (1,n) @ (n,1) -> (1,1) Scalar, numpy-compatible
        v1_np = np.ones((1, 5))
        v2_np = np.ones((5, 1))
        v1 = sm.Vector(v1_np)
        v2 = sm.Vector(v2_np)
        # all vectors are column 1
        assert v1.shape == (5, 1)
        assert v2.shape == (5, 1)
        actual = v1 @ v2
        expected = np.matmul(v1_np, v2_np)
        assert isinstance(actual, sm.Scalar)
        assert np.allclose(actual, expected)
        assert actual.shape == expected.shape

    def test_scale(self):
        v, a = np.array([1, 2, 3]), 2
        if np.linalg.norm(v) == 0:
            expected = [0, 0, 0]
        else:
            expected = v / np.linalg.norm(v) * a
        actual = sm.Vector(data=v).scale(a)
        assert np.allclose(actual, expected)

    def test_filter1(self):
        e_np = np.arange(16) * 2
        e = sm.Vector(data=e_np)
        filter_ = np.zeros(16, dtype=bool)
        filter_[3] = True
        filter_[5] = True
        actual = e[filter_].to_np()
        expected = e_np[filter_]
        assert np.all(actual == expected)

    def test_concatenate(self):
        v1 = sm.Vector([1, 2, 3])
        v2 = sm.Vector([4, 5, 6, 7])
        assert np.allclose(v1.concatenate(v2), np.concatenate([v1, v2]))

    def test_concatenate2(self):
        v1 = sm.Vector([1, 2, 3])
        assert np.allclose(sm.concatenate(v1, v1, v1), np.concatenate([v1, v1, v1]))


class TestMatrix:
    @pytest.mark.parametrize("shape", [(1, 1), (1, 5), (3, 1), (2, 3), (4, 4)])
    def test_create_filled_with_variables(self, shape):
        m = sm.Matrix.create_filled_with_variables(shape, name="muh")
        for row in range(shape[0]):
            for col in range(shape[1]):
                v = m[row, col].free_variables()[0]
                assert v.name == f"muh_{row}_{col}"

    @pytest.mark.parametrize("shape", [(1, 1), (1, 5), (3, 1), (2, 3), (4, 4)])
    def test_matrix_flatten_matches_numpy(self, shape):
        data = np.arange(shape[0] * shape[1]).reshape(shape)
        m = sm.Matrix(data)
        flat = m.flatten()
        np_flat = data.flatten()
        # Values should match exactly
        assert np.allclose(np.array(flat), np_flat)
        # Shape should be (n,1) like Vector
        assert flat.shape == (np_flat.size, 1)

    def test_flatten_empty(self):
        data = np.eye(0)
        m = sm.Matrix(data)
        flat = m.flatten()
        assert isinstance(flat, sm.Vector)
        assert flat.shape == (0, 0) or flat.shape == (0, 1)  # allow empty handling
        assert np.size(np.array(flat)) == 0

    def test_matrix_to_list(self):
        data = np.arange(12).reshape((3, 4))
        m = sm.Matrix(data)
        assert m.to_list() == data.tolist()

    def test_filter2(self):
        e_np = np.arange(16) * 2
        e_np = e_np.reshape((4, 4))
        e = sm.Matrix(data=e_np)
        filter_ = np.zeros(4, dtype=bool)
        filter_[1] = True
        filter_[2] = True
        actual = e[filter_]
        expected = e_np[filter_]
        assert np.allclose(actual, expected)

    def test_reshape(self):
        np_arr = np.arange(16)
        mat = sm.Matrix(np_arr).reshape((4, 4))
        assert mat.shape == (4, 4)
        assert np.allclose(mat.to_np(), np_arr.reshape((4, 4)))

        mat = sm.Matrix(np_arr).reshape((2, 8))
        assert mat.shape == (2, 8)
        assert np.allclose(mat.to_np(), np_arr.reshape((2, 8)))

    def test_trace(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        actual = sm.Matrix(m).trace()
        assert isinstance(actual, sm.Scalar)
        assert np.allclose(actual, np.trace(m))

        m2 = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(NotSquareMatrixError):
            sm.Matrix(m2).trace()

    def test_sum(self):
        m = np.arange(16, dtype=float).reshape((4, 4))
        actual_sum = sm.Matrix(m).sum()
        expected_sum = np.sum(m)
        assert isinstance(actual_sum, sm.Scalar)
        assert np.allclose(actual_sum, expected_sum, rtol=1.0e-4)

    def test_sum_row(self):
        m = np.arange(16, dtype=float).reshape((4, 4))
        actual_sum = sm.Matrix(data=m).sum_row()
        expected_sum = np.sum(m, axis=0)
        assert isinstance(actual_sum, sm.Vector)
        assert np.allclose(actual_sum, expected_sum)

    def test_sum_column(self):
        m = np.arange(16, dtype=float).reshape((4, 4))
        actual_sum = sm.Matrix(data=m).sum_column()
        expected_sum = np.sum(m, axis=1)
        assert isinstance(actual_sum, sm.Vector)
        assert np.allclose(actual_sum, expected_sum)

    def test_vstack(self):
        m = np.eye(4)
        m1 = sm.Matrix(m)
        e = sm.Matrix.vstack([m1, m1])
        r1 = e
        r2 = np.vstack([m, m])
        assert isinstance(r1, sm.Matrix)
        assert np.allclose(r1, r2)

    def test_vstack_empty(self):
        m = np.eye(0)
        m1 = sm.Matrix(data=m)
        e = sm.Matrix.vstack([m1, m1])
        r1 = e
        r2 = np.vstack([m, m])
        assert isinstance(r1, sm.Matrix)
        assert np.allclose(r1, r2)

    def test_hstack(self):
        m = np.eye(4)
        m1 = sm.Matrix(data=m)
        e = sm.Matrix.hstack([m1, m1])
        r1 = e
        r2 = np.hstack([m, m])
        assert isinstance(r1, sm.Matrix)
        assert np.allclose(r1, r2)

    def test_hstack_empty(self):
        m = np.eye(0)
        m1 = sm.Matrix(data=m)
        e = sm.Matrix.hstack([m1, m1])
        r1 = e
        r2 = np.hstack([m, m])
        assert isinstance(r1, sm.Matrix)
        assert np.allclose(r1, r2)

    def test_diag_stack(self):
        m1_np = np.eye(4)
        m2_np = np.ones((2, 5))
        m3_np = np.ones((5, 3))
        m1_e = sm.Matrix(data=m1_np)
        m2_e = sm.Matrix(data=m2_np)
        m3_e = sm.Matrix(data=m3_np)
        e = sm.Matrix.diag_stack([m1_e, m2_e, m3_e])
        assert isinstance(e, sm.Matrix)
        r1 = e
        combined_matrix = np.zeros((4 + 2 + 5, 4 + 5 + 3))
        row_counter = 0
        column_counter = 0
        for matrix in [m1_np, m2_np, m3_np]:
            combined_matrix[
                row_counter : row_counter + matrix.shape[0],
                column_counter : column_counter + matrix.shape[1],
            ] = matrix
            row_counter += matrix.shape[0]
            column_counter += matrix.shape[1]
        assert np.allclose(r1, combined_matrix)

    def test_iterable_rows(self):
        m_np = np.array([[1, 2, 3], [4, 5, 6]])
        m = sm.Matrix(m_np)
        rows = list(m)
        assert len(rows) == len(m_np)
        assert all(isinstance(r, sm.Vector) for r in rows)
        for i, r in enumerate(rows):
            assert np.allclose(r, m_np[i, :])

    def test_iterable_shapes(self):
        # 1xN should iterate once, yielding a Vector of length N
        m1 = sm.Matrix(np.array([[7, 8, 9]]))
        rows1 = list(m1)
        assert len(rows1) == 1
        assert isinstance(rows1[0], sm.Vector)
        assert np.allclose(rows1[0], [7, 8, 9])
        # Nx1 should iterate N times, each a Vector with one element
        m2 = sm.Matrix(np.array([[1], [2], [3]]))
        rows2 = list(m2)
        assert len(rows2) == 3
        assert all(isinstance(r, sm.Vector) for r in rows2)
        assert np.allclose(np.array([float(r[0]) for r in rows2]), [1, 2, 3])

    def test_arithmetic_operations(self):
        operators = [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.pow,
            operator.floordiv,
            operator.mod,
        ]
        v1_np = np.array([1, 2, 3])
        m1_np = np.array([[3, 2, 1], [23, 23, 23], [4, 5, 6]])
        m2_np = np.array([[6, 5, 4], [1, 2, 3], [0, 0, 0]])
        s1_float = 1
        s1 = sm.Scalar(s1_float)
        v1 = sm.Vector(v1_np)
        m1 = sm.Matrix(m1_np)
        m2 = sm.Matrix(m2_np)

        for op in operators:
            expected = op(m1_np, s1_float)
            result = op(m1, s1)
            assert isinstance(result, sm.Matrix), f"{op.__name__} result is not Matrix"
            assert np.allclose(result, expected), f"{op.__name__} result is wrong"
            assert np.allclose(
                result, op(m1, s1_float)
            ), f"{op.__name__} result is wrong"
            assert (
                result.shape == expected.shape
            ), f"{op.__name__} result shape is wrong"

            expected = op(m1_np, v1_np)
            result = op(m1, v1)
            assert isinstance(result, sm.Matrix), f"{op.__name__} result is not Matrix"
            assert np.allclose(result, expected), f"{op.__name__} result is wrong"
            assert (
                result.shape == expected.shape
            ), f"{op.__name__} result shape is wrong"

            expected = op(m1_np, v1_np)
            result = op(m1, v1)
            assert isinstance(result, sm.Matrix), f"{op.__name__} result is not Matrix"
            assert np.allclose(result, expected), f"{op.__name__} result is wrong"
            assert (
                result.shape == expected.shape
            ), f"{op.__name__} result shape is wrong"

            expected = op(m1_np, m2_np)
            result = op(m1, m2)
            assert isinstance(result, sm.Matrix), f"{op.__name__} result is not Matrix"
            assert np.allclose(result, expected), f"{op.__name__} result is wrong"
            assert (
                result.shape == expected.shape
            ), f"{op.__name__} result shape is wrong"

    def test_abs(self):
        f1 = -sm.Matrix.eye(3)
        assert np.allclose(abs(f1), abs(f1.to_np()))
        assert isinstance(abs(f1), sm.Matrix)

    def test_matmul_scalar(self):
        m = sm.Matrix([[23]])
        s = sm.Scalar(5)
        actual = m @ s
        expected = np.matmul(m, s)
        assert np.allclose(actual, expected)
        assert isinstance(actual, sm.Scalar)

    def test_matmul_vector(self):
        m = sm.Matrix(np.random.rand(3, 3))
        v = sm.Vector(np.random.rand(3))
        actual = m @ v
        expected = np.matmul(m, v)
        assert np.allclose(actual, expected)
        assert isinstance(actual, sm.Vector)

    def test_matmul_matrix(self):
        m = sm.Matrix(np.random.rand(3, 3))
        actual = m @ m
        expected = np.matmul(m, m)
        assert np.allclose(actual, expected)
        assert isinstance(actual, sm.Matrix)

    def test_matmul_vector_right(self):
        # Matrix @ Vector -> Vector
        m_np = np.random.rand(3, 4)
        v_np = np.random.rand(4, 1)
        m = sm.Matrix(m_np)
        v = sm.Vector(v_np)
        actual = m @ v
        expected = np.matmul(m_np, v_np)
        assert isinstance(actual, sm.Vector)
        # Compare flattened values because Vector.to_np() flattens 2D vectors like numpy 1-D
        assert np.allclose(np.array(actual), expected.ravel())
        # And verify the 2-D vector shape matches numpy's (n,1)
        assert actual.shape == expected.shape

    def test_len(self):
        m_np = np.eye(4)
        assert len(sm.Matrix(m_np)) == len(m_np)

    def test_kron(self):
        m1_np = np.eye(4)
        m1_cas = sm.Matrix(m1_np)
        r1 = m1_cas.kron(m1_cas)
        r2 = np.kron(m1_np, m1_np)
        assert np.allclose(r1, r2)
        assert isinstance(r1, sm.Matrix)

    def test_diag(self):
        v_np = np.arange(23)
        m_np = np.diag(v_np)
        m_cas = sm.Matrix.diag(v_np)
        assert np.allclose(m_cas, m_np)
        assert isinstance(m_cas, sm.Matrix)

    def test_get_item(self):
        m = sm.Matrix(data=np.eye(4))
        assert m[0, 0] == 1
        assert m[1, 1] == 1
        assert m[1, 0] == 0
        assert isinstance(m[0, 0], sm.Scalar)
        assert np.allclose(m[2, :], sm.Vector([0, 0, 1, 0]))
        assert isinstance(m[2, :], sm.Vector)
        assert np.allclose(m[:2, :2], np.eye(2))
        assert isinstance(m[:2, :2], sm.Matrix)

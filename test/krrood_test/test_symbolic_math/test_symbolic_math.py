import numpy as np
import pytest
import scipy

import krrood.symbolic_math.symbolic_math as cas
from krrood.symbolic_math.exceptions import HasFreeVariablesError, NotScalerError
from test.krrood_test.test_symbolic_math.reference_implementations import (
    normalize_angle_positive,
    shortest_angular_distance,
    normalize_angle,
    rotation_matrix_from_quaternion,
)

TrinaryTrue = cas.TrinaryTrue.to_np()[0]
TrinaryFalse = cas.TrinaryFalse.to_np()[0]
TrinaryUnknown = cas.TrinaryUnknown.to_np()[0]

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


class TestLogic3:
    values = [
        TrinaryTrue,
        TrinaryFalse,
        TrinaryUnknown,
    ]

    def test_and3(self):
        s = cas.FloatVariable(name="a")
        s2 = cas.FloatVariable(name="b")
        expr = cas.trinary_logic_and(s, s2)
        f = expr.compile()
        for i in self.values:
            for j in self.values:
                expected = logic_and(i, j)
                actual = f(np.array([i, j]))
                assert (
                    expected == actual
                ), f"a={i}, b={j}, expected {expected}, actual {actual}"

    def test_or3(self):
        s = cas.FloatVariable(name="a")
        s2 = cas.FloatVariable(name="b")
        expr = cas.trinary_logic_or(s, s2)
        f = expr.compile()
        for i in self.values:
            for j in self.values:
                expected = logic_or(i, j)
                actual = f(np.array([i, j]))
                assert (
                    expected == actual
                ), f"a={i}, b={j}, expected {expected}, actual {actual}"

    def test_not3(self):
        s = cas.FloatVariable(name="muh")
        expr = cas.trinary_logic_not(s)
        f = expr.compile()
        for i in self.values:
            expected = logic_not(i)
            actual = f(np.array([i]))
            assert expected == actual, f"a={i}, expected {expected}, actual {actual}"

    def test_trinary_logic_to_str(self):
        a = cas.FloatVariable(name="a")
        b = cas.FloatVariable(name="b")
        c = cas.FloatVariable(name="c")
        expression = cas.trinary_logic_and(
            a, cas.trinary_logic_or(b, cas.trinary_logic_not(c))
        )
        expression_str = cas.trinary_logic_to_str(expression)
        assert expression_str == '("a" and ("b" or not "c"))'


class TestBinaryLogic:
    def test_bool_casting(self):
        v = cas.FloatVariable(name="v")
        v2 = cas.FloatVariable(name="v2")
        v3 = cas.FloatVariable(name="v3")

        # simple casting cases of constants
        assert cas.BinaryTrue
        assert not cas.BinaryFalse

        # "and" and "or" are smart and will simply const True/False away.
        assert not (v and cas.BinaryFalse)
        assert v or cas.BinaryTrue

        # the == calls __eq__ which returns an expression.
        assert v == v
        assert v != v2

        # "in" is calling __eq__ which creates, e.g., v == v, bool casting of eq works and will return False for first eq and True for second
        assert v2 in [v, v2, v3]
        assert v not in [v2, v3]

        # const logical expressions can be evaluated
        assert cas.Expression(10) > cas.Expression(5)

        # here bool is called on v and returns it if it is True, otherwise it would return v2
        # this is the "normal" behavior for python objects
        assert (v or v2) == v


class TestIfElse:
    def test_if_one_arg(self):
        inputs = [
            (cas.FloatVariable(name="muh"), cas.FloatVariable(name="muh2")),
            (cas.Expression(1), cas.Expression(12)),
        ]
        if_functions = [
            cas.if_else,
            cas.if_eq_zero,
            cas.if_greater_eq_zero,
            cas.if_greater_zero,
        ]
        c = cas.FloatVariable(name="c")
        for if_result, else_result in inputs:
            for if_function in if_functions:
                result = if_function(c, if_result, else_result)
                result_type = type(result)
                if isinstance(if_result, cas.FloatVariable):
                    assert result_type == cas.Expression
                    continue
                assert isinstance(
                    result, result_type
                ), f"{type(result)} != {result_type} for {if_function}"

    def test_if_two_arg(self):
        inputs = [
            (cas.FloatVariable(name="muh"), cas.FloatVariable(name="muh2")),
            (cas.Expression(1), cas.Expression(12)),
        ]
        if_functions = [
            cas.if_eq,
            cas.if_greater,
            cas.if_greater_eq,
            cas.if_less,
            cas.if_less_eq,
        ]
        a = cas.FloatVariable(name="a")
        b = cas.FloatVariable(name="b")
        for if_result, else_result in inputs:
            for if_function in if_functions:
                result = if_function(a, b, if_result, else_result)
                result_type = type(result)
                if isinstance(if_result, cas.FloatVariable):
                    assert result_type == cas.Expression
                    continue
                assert isinstance(
                    result, result_type
                ), f"{type(result)} != {result_type} for {if_function}"

    @pytest.mark.parametrize("condition", bool_values)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_greater_zero(self, condition, if_result, else_result):
        assert np.allclose(
            cas.if_greater_zero(condition, if_result, else_result),
            float(if_result if condition > 0 else else_result),
        )

    @pytest.mark.parametrize("condition", bool_values)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_greater_eq_zero(self, condition, if_result, else_result):
        assert np.allclose(
            cas.if_greater_eq_zero(condition, if_result, else_result),
            float(if_result if condition >= 0 else else_result),
        )

    @pytest.mark.parametrize("condition", bool_values)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_eq_zero(self, condition, if_result, else_result):
        assert np.allclose(
            cas.if_eq_zero(condition, if_result, else_result),
            float(if_result if condition == 0 else else_result),
        )

    @pytest.mark.parametrize("a", numbers)
    @pytest.mark.parametrize("b", numbers)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_greater_eq(self, a, b, if_result, else_result):
        assert np.allclose(
            cas.if_greater_eq(a, b, if_result, else_result),
            float(if_result if a >= b else else_result),
        )

    @pytest.mark.parametrize("a", numbers)
    @pytest.mark.parametrize("b", numbers)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_less_eq(self, a, b, if_result, else_result):
        assert np.allclose(
            cas.if_less_eq(a, b, if_result, else_result),
            float(if_result if a <= b else else_result),
        )

    @pytest.mark.parametrize("a", numbers)
    @pytest.mark.parametrize("b", numbers)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_eq(self, a, b, if_result, else_result):
        assert np.allclose(
            cas.if_eq(a, b, if_result, else_result),
            float(if_result if a == b else else_result),
        )

    @pytest.mark.parametrize("a", numbers)
    @pytest.mark.parametrize("b", numbers)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_greater(self, a, b, if_result, else_result):
        assert np.allclose(
            cas.if_greater(a, b, if_result, else_result),
            float(if_result if a > b else else_result),
        )

    @pytest.mark.parametrize("a", numbers)
    @pytest.mark.parametrize("b", numbers)
    @pytest.mark.parametrize("if_result", numbers)
    @pytest.mark.parametrize("else_result", numbers)
    def test_if_less(self, a, b, if_result, else_result):
        assert np.allclose(
            cas.if_less(a, b, if_result, else_result),
            float(if_result if a < b else else_result),
        )

    @pytest.mark.parametrize("a", [1, 3, 4, -1, 0.5, -0.5, 0])
    def test_if_eq_cases(self, a):
        b_result_cases = [
            (1, cas.Expression(data=1)),
            (3, cas.Expression(data=3)),
            (4, cas.Expression(data=4)),
            (-1, cas.Expression(data=-1)),
            (0.5, cas.Expression(data=0.5)),
            (-0.5, cas.Expression(data=-0.5)),
        ]

        def reference(a_, b_result_cases_, else_result):
            for b, if_result in b_result_cases_:
                if a_ == b:
                    return if_result.to_np()[0]
            return else_result

        actual = cas.if_eq_cases(a, b_result_cases, cas.Expression(data=0))
        expected = float(reference(a, b_result_cases, 0))
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("a", numbers)
    def test_if_eq_cases_set(self, a):
        b_result_cases = {
            (1, cas.Expression(data=1)),
            (3, cas.Expression(data=3)),
            (4, cas.Expression(data=4)),
            (-1, cas.Expression(data=-1)),
            (0.5, cas.Expression(data=0.5)),
            (-0.5, cas.Expression(data=-0.5)),
        }

        def reference(a_, b_result_cases_, else_result):
            for b, if_result in b_result_cases_:
                if a_ == b:
                    return if_result.to_np()[0]
            return else_result

        actual = cas.if_eq_cases(a, b_result_cases, cas.Expression(data=0))
        expected = float(reference(a, b_result_cases, 0))
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("a", numbers)
    def test_if_less_eq_cases(self, a):
        b_result_cases = [
            (-1, cas.Expression(data=-1)),
            (-0.5, cas.Expression(data=-0.5)),
            (0.5, cas.Expression(data=0.5)),
            (1, cas.Expression(data=1)),
            (3, cas.Expression(data=3)),
            (4, cas.Expression(data=4)),
        ]

        def reference(a_, b_result_cases_, else_result):
            for b, if_result in b_result_cases_:
                if a_ <= b:
                    return if_result.to_np()[0]
            return else_result

        assert np.allclose(
            cas.if_less_eq_cases(a, b_result_cases, cas.Expression(data=0)),
            float(reference(a, b_result_cases, 0)),
        )


class TestFloatVariable:
    def test_back_reference(self):
        v = cas.FloatVariable(name="asdf")
        v2 = v.free_variables()[0]
        assert id(v2) == id(v)
        assert id(v2.casadi_sx) == id(v.casadi_sx)

        v3 = cas.Expression(v).free_variables()[0]
        assert id(v3) == id(v)

    def test_float_variable_unique(self):
        v1 = cas.FloatVariable(name="asdf")
        v2 = cas.FloatVariable(name="asdf")
        e = v1 + v2
        e.compile([[v1, v2]])

    def test_from_name(self):
        s = cas.FloatVariable(name="muh")
        assert isinstance(s, cas.FloatVariable)
        assert str(s) == "muh"

    def test_to_np(self):
        s1 = cas.FloatVariable(name="s1")
        with pytest.raises(HasFreeVariablesError):
            s1.to_np()

    def test_add(self):
        s = cas.FloatVariable(name="muh")
        # int float addition is fine
        assert isinstance(s + 1, cas.Expression)
        assert isinstance(1 + s, cas.Expression)
        assert isinstance(s + 1.0, cas.Expression)
        assert isinstance(1.0 + s, cas.Expression)

        assert isinstance(s + s, cas.Expression)

        e = cas.Expression(data=1)
        assert isinstance(e + s, cas.Expression)
        assert isinstance(s + e, cas.Expression)

    def test_sub(self):
        s = cas.FloatVariable(name="muh")
        # int float addition is fine
        assert isinstance(s - 1, cas.Expression)
        assert isinstance(1 - s, cas.Expression)
        assert isinstance(s - 1.0, cas.Expression)
        assert isinstance(1.0 - s, cas.Expression)

        assert isinstance(s - s, cas.Expression)

        e = cas.Expression(data=1)
        assert isinstance(e - s, cas.Expression)
        assert isinstance(s - e, cas.Expression)

    def test_mul(self):
        s = cas.FloatVariable(name="muh")
        # int float addition is fine
        assert isinstance(s * 1, cas.Expression)
        assert isinstance(1 * s, cas.Expression)
        assert isinstance(s * 1.0, cas.Expression)
        assert isinstance(1.0 * s, cas.Expression)

        assert isinstance(s * s, cas.Expression)

        e = cas.Expression()
        assert isinstance(e * s, cas.Expression)
        assert isinstance(s * e, cas.Expression)

    def test_truediv(self):
        s = cas.FloatVariable(name="muh")
        # int float addition is fine
        assert isinstance(s / 1, cas.Expression)
        assert isinstance(1 / s, cas.Expression)
        assert isinstance(s / 1.0, cas.Expression)
        assert isinstance(1.0 / s, cas.Expression)

        assert isinstance(s / s, cas.Expression)

        e = cas.Expression(data=1)
        assert isinstance(e / s, cas.Expression)
        assert isinstance(s / e, cas.Expression)

    def test_lt(self):
        s = cas.FloatVariable(name="muh")
        # int float addition is fine
        assert isinstance(s < 1, cas.Expression)
        assert isinstance(1 < s, cas.Expression)
        assert isinstance(s < 1.0, cas.Expression)
        assert isinstance(1.0 < s, cas.Expression)

        assert isinstance(s < s, cas.Expression)

        e = cas.Expression(data=1)
        assert isinstance(e < s, cas.Expression)
        assert isinstance(s < e, cas.Expression)

    def test_pow(self):
        s = cas.FloatVariable(name="muh")
        # int float addition is fine
        assert isinstance(s**1, cas.Expression)
        assert isinstance(1**s, cas.Expression)
        assert isinstance(s**1.0, cas.Expression)
        assert isinstance(1.0**s, cas.Expression)

        assert isinstance(s**s, cas.Expression)

        e = cas.Expression()
        assert isinstance(e**s, cas.Expression)
        assert isinstance(s**e, cas.Expression)

    def test_simple_math(self):
        s = cas.FloatVariable(name="muh")
        e = s + s
        assert isinstance(e, cas.Expression)
        e = s - s
        assert isinstance(e, cas.Expression)
        e = s * s
        assert isinstance(e, cas.Expression)
        e = s / s
        assert isinstance(e, cas.Expression)
        e = s**s
        assert isinstance(e, cas.Expression)

    def test_comparisons(self):
        s = cas.FloatVariable(name="muh")
        e = s > s
        assert isinstance(e, cas.Expression)
        e = s >= s
        assert isinstance(e, cas.Expression)
        e = s < s
        assert isinstance(e, cas.Expression)
        e = s <= s
        assert isinstance(e, cas.Expression)
        e = s == s
        assert isinstance(e, cas.Expression)

    def test_logic(self):
        s1 = cas.FloatVariable(name="s1")
        s2 = cas.FloatVariable(name="s2")
        s3 = cas.FloatVariable(name="s3")
        e = s1 | s2
        assert isinstance(e, cas.Expression)
        e = s1 & s2
        assert isinstance(e, cas.Expression)
        e = ~s1
        assert isinstance(e, cas.Expression)
        e = s1 & (s2 | ~s3)
        assert isinstance(e, cas.Expression)

    def test_hash(self):
        s = cas.FloatVariable(name="muh")
        d = {s: 1}
        assert d[s] == 1


class TestExpression:
    def test_kron(self):
        m1 = np.eye(4)
        r1 = cas.Expression(data=m1).kron(cas.Expression(data=m1))
        r2 = np.kron(m1, m1)
        assert np.allclose(r1, r2)

    def test_jacobian(self):
        a = cas.FloatVariable(name="a")
        b = cas.FloatVariable(name="b")
        m = cas.Expression(data=[a + b, a**2, b**2])
        jac = m.jacobian([a, b])
        expected = cas.Expression(data=[[1, 1], [2 * a, 0], [0, 2 * b]])
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
        a_s = cas.FloatVariable(name="a")
        ad_s = cas.FloatVariable(name="ad")
        b_s = cas.FloatVariable(name="b")
        bd_s = cas.FloatVariable(name="bd")
        m = cas.Expression(
            data=[
                a_s**3 * b_s**3,
                -a_s * cas.cos(b_s),
            ]
        )
        jac = m.jacobian_dot([a_s, b_s], [ad_s, bd_s])
        expected_expr = cas.Expression(
            data=[
                [
                    6 * ad_s * a_s * b_s**3 + 9 * a_s**2 * bd_s * b_s**2,
                    9 * ad_s * a_s**2 * b_s**2 + 6 * a_s**3 * bd_s * b,
                ],
                # [0, 2 * bd_s],
                [bd_s * cas.sin(b_s), ad_s * cas.sin(b_s) + a_s * bd_s * cas.cos(b_s)],
                # [4 * bd * b ** 3, 4 * ad * b ** 3 + 12 * a * bd * b ** 2]
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
        a_s = cas.FloatVariable(name="a")
        ad_s = cas.FloatVariable(name="ad")
        add_s = cas.FloatVariable(name="add")
        b_s = cas.FloatVariable(name="b")
        bd_s = cas.FloatVariable(name="bd")
        bdd_s = cas.FloatVariable(name="bdd")
        m = cas.Expression(
            data=[
                a_s**3 * b_s**3,
                b_s**2,
                -a_s * cas.cos(b_s),
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
        a_s = cas.FloatVariable(name="a")
        ad_s = cas.FloatVariable(name="ad")
        add_s = cas.FloatVariable(name="add")
        b_s = cas.FloatVariable(name="b")
        bd_s = cas.FloatVariable(name="bd")
        bdd_s = cas.FloatVariable(name="bdd")
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
        a_s = cas.FloatVariable(name="a")
        ad_s = cas.FloatVariable(name="ad")
        add_s = cas.FloatVariable(name="add")
        b_s = cas.FloatVariable(name="b")
        bd_s = cas.FloatVariable(name="bd")
        bdd_s = cas.FloatVariable(name="bdd")
        c_s = cas.FloatVariable(name="c")
        cd_s = cas.FloatVariable(name="cd")
        cdd_s = cas.FloatVariable(name="cdd")
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

    def test_free_variables(self):
        m = cas.Expression(data=cas.create_float_variables(["a", "b", "c", "d"]))
        assert len(m.free_variables()) == 4
        a = cas.FloatVariable(name="a")
        assert a.equivalent(a.free_variables()[0])

    def test_diag(self):
        result = cas.Expression.diag([1, 2, 3])
        assert result[0, 0] == 1
        assert result[0, 1] == 0
        assert result[0, 2] == 0

        assert result[1, 0] == 0
        assert result[1, 1] == 2
        assert result[1, 2] == 0

        assert result[2, 0] == 0
        assert result[2, 1] == 0
        assert result[2, 2] == 3
        assert cas.diag(cas.Expression(data=[1, 2, 3])).equivalent(cas.diag([1, 2, 3]))

    def test_dot(self):
        u, v = np.array([1, 2, 3]), np.array([4, 5, 6])
        result = cas.Expression(data=u) @ cas.Expression(data=v)
        u = np.array(u)
        v = np.array(v)
        assert np.allclose(result, np.dot(u, v))

    def test_dot_matrix(self):
        u, v = np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])
        result = cas.Expression(data=u) @ cas.Expression(data=v)
        expected = np.dot(u, v)
        assert np.allclose(result, expected)

    def test_pretty_str(self):
        e = cas.Expression.eye(4)
        e.pretty_str()

    def test_norm(self):
        v = np.array([1, 2, 3])
        actual = cas.Expression(data=v).norm()
        expected = np.linalg.norm(v)
        assert np.allclose(actual, expected, equal_nan=True)

    def test_create(self):
        cas.Expression(data=cas.FloatVariable(name="muh"))
        cas.Expression(data=[cas._ca.SX(1), cas._ca.SX.sym("muh")])
        m = cas.Expression(data=np.eye(4))
        m = cas.Expression(data=m)
        assert np.allclose(m, np.eye(4))
        m = cas.Expression(cas._ca.SX(np.eye(4)))
        assert np.allclose(m, np.eye(4))
        m = cas.Expression(data=[1, 1])
        assert np.allclose(m, np.array([1, 1]))
        m = cas.Expression(data=[np.array([1, 1])])
        assert np.allclose(m, np.array([1, 1]))
        m = cas.Expression(data=1)
        assert m.to_np() == 1
        m = cas.Expression(data=[[1, 1], [2, 2]])
        assert np.allclose(m, np.array([[1, 1], [2, 2]]))
        m = cas.Expression(data=[])
        assert m.shape[0] == m.shape[1] == 0

        m = cas.Expression()
        assert m.shape[0] == m.shape[1] == 0

    def test_filter1(self):
        e_np = np.arange(16) * 2
        e = cas.Expression(data=e_np)
        filter_ = np.zeros(16, dtype=bool)
        filter_[3] = True
        filter_[5] = True
        actual = e[filter_].to_np()
        expected = e_np[filter_]
        assert np.all(actual == expected)

    def test_filter2(self):
        e_np = np.arange(16) * 2
        e_np = e_np.reshape((4, 4))
        e = cas.Expression(data=e_np)
        filter_ = np.zeros(4, dtype=bool)
        filter_[1] = True
        filter_[2] = True
        actual = e[filter_]
        expected = e_np[filter_]
        assert np.allclose(actual, expected)

    def test_add(self):
        f1, f2 = 23, 69
        expected = f1 + f2
        r1 = cas.Expression(data=f2) + f1
        assert np.allclose(r1, expected)
        r1 = f1 + cas.Expression(data=f2)
        assert np.allclose(r1, expected)
        r1 = cas.Expression(data=f1) + cas.Expression(data=f2)
        assert np.allclose(r1, expected)

    def test_sub(self):
        f1, f2 = 23, 69
        expected = f1 - f2
        r1 = cas.Expression(data=f1) - f2
        assert np.allclose(r1, expected)
        r1 = f1 - cas.Expression(data=f2)
        assert np.allclose(r1, expected)
        r1 = cas.Expression(data=f1) - cas.Expression(data=f2)
        assert np.allclose(r1, expected)

    def test_len(self):
        m = cas.Expression(data=np.eye(4))
        assert len(m) == len(np.eye(4))

    def test_simple_math(self):
        m = cas.Expression(data=[1, 1])
        s = cas.FloatVariable(name="muh")
        e = m + s
        e = m + 1
        e = 1 + m
        assert isinstance(e, cas.Expression)
        e = m - s
        e = m - 1
        e = 1 - m
        assert isinstance(e, cas.Expression)
        e = m * s
        e = m * 1
        e = 1 * m
        assert isinstance(e, cas.Expression)
        e = m / s
        e = m / 1
        e = 1 / m
        assert isinstance(e, cas.Expression)
        e = m**s
        e = m**1
        e = 1**m
        assert isinstance(e, cas.Expression)

    def test_to_np(self):
        e = cas.Expression(data=1)
        assert np.allclose(e.to_np(), np.array([1]))
        e = cas.Expression(data=[1, 2])
        assert np.allclose(e.to_np(), np.array([1, 2]))
        e = cas.Expression(data=[[1, 2], [3, 4]])
        assert np.allclose(e.to_np(), np.array([[1, 2], [3, 4]]))

    def test_to_np_fail(self):
        s1, s2 = cas.FloatVariable(name="s1"), cas.FloatVariable(name="s2")
        e = s1 + s2
        with pytest.raises(HasFreeVariablesError):
            e.to_np()

    def test_scale(self):
        v, a = np.array([1, 2, 3]), 2
        if np.linalg.norm(v) == 0:
            expected = [0, 0, 0]
        else:
            expected = v / np.linalg.norm(v) * a
        actual = cas.Expression(data=v).scale(a)
        assert np.allclose(actual, expected)

    def test_get_attr(self):
        m = cas.Expression(data=np.eye(4))
        assert m[0, 0] == cas.Expression(data=1)
        assert m[1, 1] == cas.Expression(data=1)
        assert m[1, 0] == cas.Expression(data=0)
        assert isinstance(m[0, 0], cas.Expression)
        print(m.shape)

    def test_comparisons(self):
        logic_functions = [
            lambda a, b: a > b,
            lambda a, b: a >= b,
            lambda a, b: a < b,
            lambda a, b: a <= b,
            lambda a, b: a == b,
        ]
        e1_np = np.array([1, 2, 3, -1])
        e2_np = np.array([1, 1, -1, 3])
        e1_cas = cas.Expression(data=e1_np)
        e2_cas = cas.Expression(data=e2_np)
        for f in logic_functions:
            r_np = f(e1_np, e2_np)
            r_cas = f(e1_cas, e2_cas)
            assert isinstance(r_cas, cas.Expression)
            assert np.all(r_np == r_cas)

    def test_logic_and(self):
        s1 = cas.FloatVariable(name="s1")
        s2 = cas.FloatVariable(name="s2")
        expr = cas.logic_and(cas.BinaryTrue, s1)
        assert not cas.is_const_binary_true(expr) and not cas.is_const_binary_false(
            expr
        )
        expr = cas.logic_and(cas.BinaryFalse, s1)
        assert cas.is_const_binary_false(expr)
        expr = cas.logic_and(cas.BinaryTrue, cas.BinaryTrue)
        assert cas.is_const_binary_true(expr)
        expr = cas.logic_and(cas.BinaryFalse, cas.BinaryTrue)
        assert cas.is_const_binary_false(expr)
        expr = cas.logic_and(cas.BinaryFalse, cas.BinaryFalse)
        assert cas.is_const_binary_false(expr)
        expr = cas.logic_and(s1, s2)
        assert not cas.is_const_binary_true(expr) and not cas.is_const_binary_false(
            expr
        )

    def test_logic_or(self):
        s1 = cas.FloatVariable(name="s1")
        s2 = cas.FloatVariable(name="s2")
        s3 = cas.FloatVariable(name="s3")
        expr = cas.logic_or(cas.BinaryFalse, s1)
        assert not cas.is_const_binary_true(expr) and not cas.is_const_binary_false(
            expr
        )
        expr = cas.logic_or(cas.BinaryTrue, s1)
        assert cas.is_const_binary_true(expr)
        expr = cas.logic_or(cas.BinaryTrue, cas.BinaryTrue)
        assert cas.is_const_binary_true(expr)
        expr = cas.logic_or(cas.BinaryFalse, cas.BinaryTrue)
        assert cas.is_const_binary_true(expr)
        expr = cas.logic_or(cas.BinaryFalse, cas.BinaryFalse)
        assert cas.is_const_binary_false(expr)
        expr = cas.logic_or(s1, s2)
        assert not cas.is_const_binary_true(expr) and not cas.is_const_binary_false(
            expr
        )

        expr = cas.logic_or(s1, s2, s3)
        assert not cas.is_const_binary_true(expr) and not cas.is_const_binary_false(
            expr
        )

    def test_lt(self):
        e1 = cas.Expression(data=[1, 2, 3, -1])
        e2 = cas.Expression(data=[1, 1, -1, 3])
        gt_result = e1 < e2
        assert isinstance(gt_result, cas.Expression)
        assert cas.logic_all(gt_result == cas.Expression(data=[0, 0, 0, 1])).to_np()


class TestScalarMathFunctions:
    def test_float_casting(self):
        assert float(cas.Expression(data=1)) == 1.0
        with pytest.raises(NotScalerError):
            assert float(cas.Expression(data=[1, 2]))
        with pytest.raises(HasFreeVariablesError):
            assert float(cas.FloatVariable(name="muh"))

    def test_abs(self):
        f1 = 23
        assert np.allclose(cas.abs(f1), abs(f1))

    def test_max(self):
        f1, f2 = 23, 69
        assert np.allclose(cas.max(f1, f2), max(f1, f2))

    def test_save_division(self):
        f1, f2 = 23, 69
        assert np.allclose(
            cas.Expression(data=f1).safe_division(f2), f1 / f2 if f2 != 0 else 0
        )

    def test_min(self):
        f1, f2 = 23, 69
        assert np.allclose(cas.min(f1, f2), min(f1, f2))

    def test_sign(self):
        f1 = 23
        assert np.allclose(cas.sign(f1), np.sign(f1))

    @pytest.mark.parametrize("x", numbers)
    @pytest.mark.parametrize("lower_limit", numbers)
    @pytest.mark.parametrize("upper_limit", numbers)
    def test_limit(self, x, lower_limit, upper_limit):
        r1 = cas.limit(x, lower_limit, upper_limit)
        r2 = max(lower_limit, min(upper_limit, x))
        assert np.allclose(r1, r2)

    @pytest.mark.parametrize("a", numbers)
    @pytest.mark.parametrize("b", numbers)
    def test_fmod(self, a, b):
        ref_r = np.fmod(a, b)
        assert np.allclose(cas.fmod(a, b), ref_r, equal_nan=True)

    @pytest.mark.parametrize("a", numbers)
    def test_normalize_angle_positive(self, a):
        expected = normalize_angle_positive(a)
        actual = cas.normalize_angle_positive(a)
        assert np.allclose(
            shortest_angular_distance(actual.to_np(), expected),
            0.0,
        )

    @pytest.mark.parametrize("a", numbers)
    def test_normalize_angle(self, a):
        ref_r = normalize_angle(a)
        assert np.allclose(cas.normalize_angle(a), ref_r)

    @pytest.mark.parametrize("angle1", numbers)
    @pytest.mark.parametrize("angle2", numbers)
    def test_shorted_angular_distance(self, angle1, angle2):
        try:
            expected = shortest_angular_distance(angle1, angle2)
        except ValueError:
            expected = np.nan
        actual = cas.shortest_angular_distance(angle1, angle2)
        assert np.allclose(actual, expected, equal_nan=True)


class TestArrayMathFunctions:
    def test_leq_on_array(self):
        a = cas.Expression(data=np.array([1, 2, 3, 4]))
        b = cas.Expression(data=np.array([2, 2, 2, 2]))
        assert not cas.logic_all(a <= b).to_np()

    def test_trace(self):
        m = rotation_matrix_from_quaternion(0, 1, 0, 0)
        assert np.allclose(m.trace(), np.trace(m))

    @pytest.mark.parametrize("q1", quaternions)
    @pytest.mark.parametrize("q2", quaternions)
    def test_entrywise_product(self, q1, q2):
        m1 = rotation_matrix_from_quaternion(*q1)
        m2 = rotation_matrix_from_quaternion(*q2)
        r1 = cas.Expression(data=m1).entrywise_product(m2)
        r2 = m1 * m2
        assert np.allclose(r1, r2)

    def test_sum(self):
        m = np.arange(16, dtype=float).reshape((4, 4))
        actual_sum = m.sum()
        expected_sum = np.sum(m)
        assert np.allclose(actual_sum, expected_sum, rtol=1.0e-4)

    def test_sum_row(self):
        m = np.arange(16, dtype=float).reshape((4, 4))
        actual_sum = cas.Expression(data=m).sum_row()
        expected_sum = np.sum(m, axis=0)
        assert np.allclose(actual_sum, expected_sum)

    def test_sum_column(self):
        m = np.arange(16, dtype=float).reshape((4, 4))
        actual_sum = cas.Expression(data=m).sum_column()
        expected_sum = np.sum(m, axis=1)
        assert np.allclose(actual_sum, expected_sum)

    def test_vstack(self):
        m = np.eye(4)
        m1 = cas.Expression(data=m)
        e = cas.Expression.vstack([m1, m1])
        r1 = e
        r2 = np.vstack([m, m])
        assert np.allclose(r1, r2)

    def test_vstack_empty(self):
        m = np.eye(0)
        m1 = cas.Expression(data=m)
        e = cas.Expression.vstack([m1, m1])
        r1 = e
        r2 = np.vstack([m, m])
        assert np.allclose(r1, r2)

    def test_hstack(self):
        m = np.eye(4)
        m1 = cas.Expression(data=m)
        e = cas.Expression.hstack([m1, m1])
        r1 = e
        r2 = np.hstack([m, m])
        assert np.allclose(r1, r2)

    def test_hstack_empty(self):
        m = np.eye(0)
        m1 = cas.Expression(data=m)
        e = cas.Expression.hstack([m1, m1])
        r1 = e
        r2 = np.hstack([m, m])
        assert np.allclose(r1, r2)

    def test_diag_stack(self):
        m1_np = np.eye(4)
        m2_np = np.ones((2, 5))
        m3_np = np.ones((5, 3))
        m1_e = cas.Expression(data=m1_np)
        m2_e = cas.Expression(data=m2_np)
        m3_e = cas.Expression(data=m3_np)
        e = cas.Expression.diag_stack([m1_e, m2_e, m3_e])
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


class TestSymbolicType:
    def test_substitute(self):
        a, b, c, d = cas.create_float_variables(["a", "b", "c", "d"])
        expr = a * b
        expr_substituted = expr.substitute([a, b], [c, d])
        assert expr_substituted == c * d

    def test_empty_compiled_function(self):
        expected = np.array([1, 2, 3])
        e = cas.Expression(data=expected)
        f = e.compile(sparse=False)
        assert np.allclose(f(), expected)
        assert np.allclose(f(np.array([], dtype=float)), expected)

    def test_empty_compiled_function_sparse(self):
        expected = np.array([1, 2, 3], ndmin=2)
        e = cas.Expression(data=expected)
        f = e.compile(sparse=True)
        assert np.allclose(f().toarray(), expected)
        assert np.allclose(f(np.array([], dtype=float)).toarray(), expected)

    def test_create_variables(self):
        result = cas.create_float_variables(["a", "b", "c"])
        assert str(result[0]) == "a"
        assert str(result[1]) == "b"
        assert str(result[2]) == "c"

    def test_create_variables2(self):
        result = cas.create_float_variables(3)
        assert str(result[0]) == "s_0"
        assert str(result[1]) == "s_1"
        assert str(result[2]) == "s_2"

    def test_to_str(self):
        expr = cas.FloatVariable(name="muh") * cas.Expression(23)
        assert expr.pretty_str() == [["(23*muh)"]]


class TestCompiledFunction:
    def test_dense(self):
        s1_value = 420.0
        s2_value = 69.0
        s1, s2 = cas.create_float_variables(["s1", "s2"])
        e = cas.sqrt(cas.cos(s1) + cas.sin(s2))
        e_f = e.compile()
        actual = e_f(np.array([s1_value, s2_value]))
        expected = np.sqrt(np.cos(s1_value) + np.sin(s2_value))
        assert np.allclose(actual, expected)

    def test_dense_two_params(self):
        s1_value = 420.0
        s2_value = 69.0
        s1, s2 = cas.create_float_variables(["s1", "s2"])
        e = cas.sqrt(cas.cos(s1) + cas.sin(s2))
        e_f = e.compile(parameters=[[s1], [s2]])
        actual = e_f(np.array([s1_value]), np.array([s2_value]))
        expected = np.sqrt(np.cos(s1_value) + np.sin(s2_value))
        assert np.allclose(actual, expected)

    def test_sparse(self):
        s1_value = 420.0
        s2_value = 69.0
        s1, s2 = cas.create_float_variables(["s1", "s2"])
        e = cas.sqrt(cas.cos(s1) + cas.sin(s2))
        e_f = e.compile(sparse=True)
        actual = e_f(np.array([s1_value, s2_value]))
        assert isinstance(actual, scipy.sparse.csc_matrix)
        expected = np.sqrt(np.cos(s1_value) + np.sin(s2_value))
        assert np.allclose(actual.toarray(), expected)

    def test_stacked_compiled_function_dense(self):
        s1_value = 420.0
        s2_value = 69.0
        s1, s2 = cas.create_float_variables(["s1", "s2"])
        e1 = cas.sqrt(cas.cos(s1) + cas.sin(s2))
        e2 = s1 + s2
        e_f = cas.CompiledFunctionWithViews(
            expressions=[e1, e2], variable_parameters=[[s1, s2]]
        )
        actual_e1, actual_e2 = e_f(np.array([s1_value, s2_value]))
        expected_e1 = np.sqrt(np.cos(s1_value) + np.sin(s2_value))
        expected_e2 = s1_value + s2_value
        assert np.allclose(actual_e1, expected_e1)
        assert np.allclose(actual_e2, expected_e2)

    def test_single_args(self):
        size = 10_000
        variables = cas.create_float_variables([str(i) for i in range(size)])
        expr = cas.sum(*variables)
        f = expr.compile()
        for i in range(10):
            data = np.random.rand(size)
            assert np.isclose(f(data), np.sum(data))

    def test_single_args_with_bind(self):
        size = 10_000
        data = np.random.rand(size)
        variables = cas.create_float_variables([str(i) for i in range(size)])
        expr = cas.sum(*variables)
        f = expr.compile()
        f.bind_args_to_memory_view(0, data)
        for i in range(10):
            np.copyto(data, np.random.rand(size))
            assert np.isclose(f.evaluate(), np.sum(data))

    def test_multiple_args(self):
        size = 10_000
        n = 10
        element_size = size // n
        variables = cas.create_float_variables([str(i) for i in range(size)])
        expr = cas.sum(*variables)
        args = [variables[i * element_size : (i + 1) * element_size] for i in range(n)]
        f = expr.compile(parameters=args)
        for i in range(100):
            args_values = [np.ones(element_size)] * n
            assert f(*args_values) == size

    def test_multiple_args_with_bind(self):
        size = 10_000
        n = 10
        element_size = size // n
        variables = cas.create_float_variables([str(i) for i in range(size)])
        expr = cas.sum(*variables)
        args = [variables[i * element_size : (i + 1) * element_size] for i in range(n)]
        f = expr.compile(parameters=args)

        datas = []
        for i in range(n):
            datas.append(np.random.rand(element_size))
            f.bind_args_to_memory_view(i, datas[i])
        for i in range(100):
            for i in range(n):
                datas[i][:] = np.random.rand(element_size)
            assert np.isclose(f.evaluate(), np.sum(datas))

    def test_missing_free_variables(self):
        s1, s2 = cas.create_float_variables(["s1", "s2"])
        e = cas.sqrt(cas.cos(s1) + cas.sin(s2))
        with pytest.raises(HasFreeVariablesError):
            e.compile(parameters=[[s1]])

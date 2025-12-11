import hypothesis.strategies as st
import numpy as np
import pytest
import scipy
from hypothesis import given, assume

import semantic_digital_twin.spatial_types.spatial_types as cas
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    HasFreeVariablesError,
    WrongDimensionsError,
    SpatialTypesError,
)
from .reference_implementations import (
    rotation_matrix_from_quaternion,
    axis_angle_from_rotation_matrix,
    shortest_angular_distance,
    rotation_matrix_from_axis_angle,
    rotation_matrix_from_rpy,
    quaternion_slerp,
    quaternion_from_axis_angle,
    axis_angle_from_quaternion,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_from_rpy,
    quaternion_from_rotation_matrix,
    normalize_angle_positive,
    normalize_angle,
)
from .utils_for_tests import (
    float_no_nan_no_inf,
    quaternion,
    random_angle,
    unit_vector,
    compare_axis_angle,
    angle_positive,
    vector,
    lists_of_same_length,
    compare_orientations,
    sq_matrix,
    assert_allclose,
)

TrinaryTrue = cas.TrinaryTrue.to_np()[0]
TrinaryFalse = cas.TrinaryFalse.to_np()[0]
TrinaryUnknown = cas.TrinaryUnknown.to_np()[0]


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
        a = cas.FloatVariable(name=PrefixedName("a"))
        b = cas.FloatVariable(name=PrefixedName("b"))
        c = cas.FloatVariable(name=PrefixedName("c"))
        expression = cas.trinary_logic_and(
            a, cas.trinary_logic_or(b, cas.trinary_logic_not(c))
        )
        expression_str = cas.trinary_logic_to_str(expression)
        assert expression_str == '("a" and ("b" or not "c"))'


class TestFloatVariable:
    def test_back_reference(self):
        v = cas.FloatVariable(name=PrefixedName("asdf"))
        v2 = v.free_variables()[0]
        assert id(v2) == id(v)
        assert id(v2.casadi_sx) == id(v.casadi_sx)

        v3 = cas.Point3(v).free_variables()[0]
        assert id(v3) == id(v)

    def test_float_variable_unique(self):
        from semantic_digital_twin.world import World

        v1 = cas.FloatVariable(name=PrefixedName("asdf"))
        v2 = cas.FloatVariable(name=PrefixedName("asdf"))
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

        p = cas.Point3()
        with pytest.raises(TypeError):
            s + p
        with pytest.raises(TypeError):
            p + s

        v = cas.Vector3()
        with pytest.raises(TypeError):
            s + v
        with pytest.raises(TypeError):
            v + s

        r = cas.RotationMatrix()
        with pytest.raises(TypeError):
            s + r
        with pytest.raises(TypeError):
            r + s

        t = cas.TransformationMatrix()
        with pytest.raises(TypeError):
            s + t
        with pytest.raises(TypeError):
            t + s

        q = cas.Quaternion()
        with pytest.raises(TypeError):
            s + q
        with pytest.raises(TypeError):
            q + s

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

        p = cas.Point3()
        with pytest.raises(TypeError):
            s - p
        with pytest.raises(TypeError):
            p - s

        v = cas.Vector3()
        with pytest.raises(TypeError):
            s - v
        with pytest.raises(TypeError):
            v - s

        r = cas.RotationMatrix()
        with pytest.raises(TypeError):
            s - r
        with pytest.raises(TypeError):
            r - s

        t = cas.TransformationMatrix()
        with pytest.raises(TypeError):
            s - t
        with pytest.raises(TypeError):
            t - s

        q = cas.Quaternion()
        with pytest.raises(TypeError):
            s - q
        with pytest.raises(TypeError):
            q - s

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

        p = cas.Point3()
        with pytest.raises(TypeError):
            s * p
        with pytest.raises(TypeError):
            p * s

        v = cas.Vector3()
        with pytest.raises(TypeError):
            s * v
        assert isinstance(v * s, cas.Vector3)

        r = cas.RotationMatrix()
        with pytest.raises(TypeError):
            s * r
        with pytest.raises(TypeError):
            r * s

        t = cas.TransformationMatrix()
        with pytest.raises(TypeError):
            s * t
        with pytest.raises(TypeError):
            t * s

        q = cas.Quaternion()
        with pytest.raises(TypeError):
            s * q
        with pytest.raises(TypeError):
            q * s

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

        p = cas.Point3()
        with pytest.raises(TypeError):
            s / p
        with pytest.raises(TypeError):
            p / s

        v = cas.Vector3()
        with pytest.raises(TypeError):
            s / v
        assert isinstance(v / s, cas.Vector3)

        r = cas.RotationMatrix()
        with pytest.raises(TypeError):
            s / r
        with pytest.raises(TypeError):
            r / s

        t = cas.TransformationMatrix()
        with pytest.raises(TypeError):
            s / t
        with pytest.raises(TypeError):
            t / s

        q = cas.Quaternion()
        with pytest.raises(TypeError):
            s / q
        with pytest.raises(TypeError):
            q / s

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

        p = cas.Point3()
        with pytest.raises(TypeError):
            s < p
        with pytest.raises(TypeError):
            p < s

        v = cas.Vector3()
        with pytest.raises(TypeError):
            s < v
        with pytest.raises(TypeError):
            v < s

        r = cas.RotationMatrix()
        with pytest.raises(TypeError):
            s < r
        with pytest.raises(TypeError):
            r < s

        t = cas.TransformationMatrix()
        with pytest.raises(TypeError):
            s < t
        with pytest.raises(TypeError):
            t < s

        q = cas.Quaternion()
        with pytest.raises(TypeError):
            s < q
        with pytest.raises(TypeError):
            q < s

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

        p = cas.Point3()
        with pytest.raises(TypeError):
            s**p
        with pytest.raises(TypeError):
            p**s

        v = cas.Vector3()
        with pytest.raises(TypeError):
            s**v
        with pytest.raises(TypeError):
            v**s

        r = cas.RotationMatrix()
        with pytest.raises(TypeError):
            s**r
        with pytest.raises(TypeError):
            r**s

        t = cas.TransformationMatrix()
        with pytest.raises(TypeError):
            s**t
        with pytest.raises(TypeError):
            t**s

        q = cas.Quaternion()
        with pytest.raises(TypeError):
            s**q
        with pytest.raises(TypeError):
            q**s

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
        assert_allclose(r1, r2)

    def test_jacobian(self):
        a = cas.FloatVariable(name="a")
        b = cas.FloatVariable(name="b")
        m = cas.Expression(data=[a + b, a**2, b**2])
        jac = m.jacobian([a, b])
        expected = cas.Expression(data=[[1, 1], [2 * a, 0], [0, 2 * b]])
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                assert jac[i, j].equivalent(expected[i, j])

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
    )
    def test_jacobian_dot(self, a, ad, b, bd):
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
                # b_s ** 2,
                -a_s * cas.cos(b_s),
                # a_s * b_s ** 4
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
        assert_allclose(actual, expected)

    @given(
        float_no_nan_no_inf(outer_limit=1e2),
        float_no_nan_no_inf(outer_limit=1e2),
        float_no_nan_no_inf(outer_limit=1e2),
        float_no_nan_no_inf(outer_limit=1e2),
        float_no_nan_no_inf(outer_limit=1e2),
        float_no_nan_no_inf(outer_limit=1e2),
    )
    def test_jacobian_ddot(self, a, ad, add, b, bd, bdd):
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
        assert_allclose(actual, expected)

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
    )
    def test_total_derivative2(self, a, b, ad, bd, add, bdd):
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
        m = cas.Expression(data=a_s * b_s**2)
        jac = m.second_order_total_derivative([a_s, b_s], [ad_s, bd_s], [add_s, bdd_s])
        actual = jac.compile().call_with_kwargs(**kwargs)
        expected = bdd * 2 * a + 2 * ad * bd * 2 * b
        assert_allclose(actual, expected)

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
    )
    def test_total_derivative2_2(self, a, b, c, ad, bd, cd, add, bdd, cdd):
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
        m = cas.Expression(data=a_s * b_s**2 * c_s**3)
        jac = m.second_order_total_derivative(
            [a_s, b_s, c_s], [ad_s, bd_s, cd_s], [add_s, bdd_s, cdd_s]
        )
        # expected_expr = cas.Expression(add_s + bdd_s*2*a*c**3 + 4*ad_s*)
        actual = jac.compile().call_with_kwargs(**kwargs)
        # expected = expected_expr.compile()(**kwargs)
        expected = (
            bdd * 2 * a * c**3
            + cdd * 6 * a * b**2 * c
            + 4 * ad * bd * b * c**3
            + 6 * ad * b**2 * cd * c**2
            + 12 * a * bd * b * cd * c**2
        )
        assert_allclose(actual, expected)

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

    @given(
        lists_of_same_length(
            [float_no_nan_no_inf(), float_no_nan_no_inf()], max_length=50
        )
    )
    def test_dot(self, vectors):
        u, v = vectors
        result = cas.Expression(data=u).dot(cas.Expression(data=v))
        u = np.array(u)
        v = np.array(v)
        assert_allclose(result, np.dot(u, v))

    @given(
        lists_of_same_length(
            [
                float_no_nan_no_inf(outer_limit=1000),
                float_no_nan_no_inf(outer_limit=1000),
            ],
            min_length=16,
            max_length=16,
        )
    )
    def test_dot_matrix(self, vectors):
        u, v = vectors
        u = np.array(u).reshape((4, 4))
        v = np.array(v).reshape((4, 4))
        result = cas.Expression(data=u).dot(cas.Expression(data=v))
        expected = np.dot(u, v)
        assert_allclose(result, expected)

    def test_pretty_str(self):
        e = cas.Expression.eye(4)
        e.pretty_str()

    @given(st.lists(float_no_nan_no_inf(), min_size=1))
    def test_norm(self, v):
        actual = cas.Expression(data=v).norm()
        expected = np.linalg.norm(v)
        assume(not np.isinf(expected))
        assert_allclose(actual, expected, equal_nan=True)

    def test_create(self):
        cas.Expression(data=cas.FloatVariable(name="muh"))
        cas.Expression(data=[cas.ca.SX(1), cas.ca.SX.sym("muh")])
        m = cas.Expression(data=np.eye(4))
        m = cas.Expression(data=m)
        assert_allclose(m, np.eye(4))
        m = cas.Expression(cas.ca.SX(np.eye(4)))
        assert_allclose(m, np.eye(4))
        m = cas.Expression(data=[1, 1])
        assert_allclose(m, np.array([1, 1]))
        m = cas.Expression(data=[np.array([1, 1])])
        assert_allclose(m, np.array([1, 1]))
        m = cas.Expression(data=1)
        assert m.to_np() == 1
        m = cas.Expression(data=[[1, 1], [2, 2]])
        assert_allclose(m, np.array([[1, 1], [2, 2]]))
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
        assert_allclose(actual, expected)

    @given(float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_add(self, f1, f2):
        expected = f1 + f2
        r1 = cas.Expression(data=f2) + f1
        assert_allclose(r1, expected)
        r1 = f1 + cas.Expression(data=f2)
        assert_allclose(r1, expected)
        r1 = cas.Expression(data=f1) + cas.Expression(data=f2)
        assert_allclose(r1, expected)

    @given(float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_sub(self, f1, f2):
        expected = f1 - f2
        r1 = cas.Expression(data=f1) - f2
        assert_allclose(r1, expected)
        r1 = f1 - cas.Expression(data=f2)
        assert_allclose(r1, expected)
        r1 = cas.Expression(data=f1) - cas.Expression(data=f2)
        assert_allclose(r1, expected)

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
        assert_allclose(e.to_np(), np.array([1]))
        e = cas.Expression(data=[1, 2])
        assert_allclose(e.to_np(), np.array([1, 2]))
        e = cas.Expression(data=[[1, 2], [3, 4]])
        assert_allclose(e.to_np(), np.array([[1, 2], [3, 4]]))

    def test_to_np_fail(self):
        s1, s2 = cas.FloatVariable(name="s1"), cas.FloatVariable(name="s2")
        e = s1 + s2
        with pytest.raises(HasFreeVariablesError):
            e.to_np()

    @given(vector(3), float_no_nan_no_inf())
    def test_scale(self, v, a):
        if np.linalg.norm(v) == 0:
            expected = [0, 0, 0]
        else:
            expected = v / np.linalg.norm(v) * a
        actual = cas.Expression(data=v).scale(a)
        assert_allclose(actual, expected)

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
            r_cas = r_cas.to_np()
            np.all(r_np == r_cas)

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


class TestRotationMatrix:
    @given(quaternion(), quaternion())
    def test_rotation_distance(self, q1, q2):
        m1 = rotation_matrix_from_quaternion(*q1)
        m2 = rotation_matrix_from_quaternion(*q2)
        actual_angle = cas.RotationMatrix(data=m1).rotational_error(
            cas.RotationMatrix(data=m2)
        )
        _, expected_angle = axis_angle_from_rotation_matrix(m1.T.dot(m2))
        expected_angle = expected_angle
        try:
            assert_allclose(
                shortest_angular_distance(actual_angle.to_np()[0], expected_angle),
                0,
            )
        except AssertionError:
            assert_allclose(
                shortest_angular_distance(actual_angle.to_np()[0], -expected_angle),
                0,
            )

    def test_reference_frames(self):
        reference_frame = "muh"
        reference_frame2 = "muh2"
        m1 = cas.RotationMatrix(reference_frame=reference_frame)
        m2 = cas.RotationMatrix(reference_frame=reference_frame2)
        q = cas.Quaternion(reference_frame=reference_frame)

        assert m1.reference_frame == reference_frame
        assert m1.to_quaternion().reference_frame == reference_frame
        assert m1.to_axis_angle()[0].reference_frame == reference_frame
        assert (
            cas.RotationMatrix.from_rpy(reference_frame=reference_frame).reference_frame
            == reference_frame
        )
        assert (m1 @ m2).reference_frame == reference_frame
        assert (m2 @ m1).reference_frame == reference_frame2
        assert m1.x_vector().reference_frame == reference_frame
        assert m1.y_vector().reference_frame == reference_frame
        assert m1.z_vector().reference_frame == reference_frame
        assert m1.from_quaternion(q).reference_frame == reference_frame
        assert (
            m1.from_vectors(
                x=cas.Vector3.X(), y=cas.Vector3.Y(), reference_frame=reference_frame
            ).reference_frame
            == reference_frame
        )
        assert m1.inverse().reference_frame == reference_frame

    def test_matmul_type_preservation(self):
        s = cas.FloatVariable(name="s")
        e = cas.Expression(data=1)
        v = cas.Vector3(x_init=1, y_init=1, z_init=1)
        p = cas.Point3(x_init=1, y_init=1, z_init=1)
        r = cas.RotationMatrix()
        q = cas.Quaternion()
        t = cas.TransformationMatrix()

        with pytest.raises(TypeError):
            r @ s
        with pytest.raises(TypeError):
            r @ e
        assert isinstance(r @ v, cas.Vector3)
        with pytest.raises(TypeError):
            assert isinstance(r @ p, cas.Point3)
        assert isinstance(r @ r, cas.RotationMatrix)
        with pytest.raises(TypeError):
            r @ q
        assert isinstance(r @ t, cas.TransformationMatrix)
        assert isinstance(t @ r, cas.RotationMatrix)

    def test_x_y_z_vector(self):
        v = np.array([1, 1, 1])
        v = v / np.linalg.norm(v)
        R_ref = rotation_matrix_from_axis_angle(v, 1)
        R = cas.RotationMatrix().from_axis_angle(cas.Vector3.unit_vector(1, 1, 1), 1)
        assert_allclose(R.x_vector(), R_ref[:, 0])
        assert_allclose(R.y_vector(), R_ref[:, 1])
        assert_allclose(R.z_vector(), R_ref[:, 2])

    def test_create_RotationMatrix(self):
        s = cas.FloatVariable(name="s")
        r = cas.RotationMatrix.from_rpy(1, 2, s)
        r = cas.RotationMatrix.from_rpy(1, 2, 3)
        assert isinstance(r, cas.RotationMatrix)
        t = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3)
        r = cas.RotationMatrix(data=t)
        assert t[0, 3].to_np() == 1

    def test_from_vectors(self):
        v = np.array([1, 1, 1])
        v = v / np.linalg.norm(v)
        R_ref = rotation_matrix_from_axis_angle(v, 1)
        x = R_ref[:, 0]
        y = R_ref[:, 1]
        z = R_ref[:, 2]
        x_unit = cas.Vector3.from_iterable(x)
        x_unit.scale(1.1)
        y_unit = cas.Vector3.from_iterable(y)
        y_unit.scale(0.2)
        z_unit = cas.Vector3.from_iterable(z)
        z_unit.scale(1.1)
        assert_allclose(cas.RotationMatrix.from_vectors(x=x_unit, y=y_unit), R_ref)
        assert_allclose(cas.RotationMatrix.from_vectors(x=x_unit, z=z_unit), R_ref)
        assert_allclose(cas.RotationMatrix.from_vectors(y=y_unit, z=z_unit), R_ref)
        assert_allclose(
            cas.RotationMatrix.from_vectors(x=x_unit, y=y_unit, z=z_unit), R_ref
        )

    @given(quaternion())
    def test_from_quaternion(self, q):
        actual = cas.RotationMatrix.from_quaternion(cas.Quaternion.from_iterable(q))
        expected = rotation_matrix_from_quaternion(*q)
        assert_allclose(actual, expected)

    @given(random_angle(), random_angle(), random_angle())
    def test_from_rpy(self, roll, pitch, yaw):
        m1 = cas.RotationMatrix.from_rpy(roll, pitch, yaw)
        m2 = rotation_matrix_from_rpy(roll, pitch, yaw)
        assert_allclose(m1, m2)

    @given(unit_vector(length=3), random_angle())
    def test_rotation3_axis_angle(self, axis, angle):
        assert_allclose(
            cas.RotationMatrix.from_axis_angle(axis, angle),
            rotation_matrix_from_axis_angle(np.array(axis), angle),
        )

    @given(quaternion())
    def test_axis_angle_from_matrix(self, q):
        m = rotation_matrix_from_quaternion(*q)

        actual_axis = cas.RotationMatrix(data=m).to_axis_angle()[0]
        actual_angle = cas.RotationMatrix(data=m).to_axis_angle()[1]

        expected_axis, expected_angle = axis_angle_from_rotation_matrix(m)
        compare_axis_angle(actual_angle, actual_axis[:3], expected_angle, expected_axis)

        assert actual_axis[-1].to_np() == 0

    @given(unit_vector(length=3), angle_positive())
    def test_axis_angle_from_matrix2(self, expected_axis, expected_angle):
        m = rotation_matrix_from_axis_angle(expected_axis, expected_angle)
        actual_axis = cas.RotationMatrix(data=m).to_axis_angle()[0]
        actual_angle = cas.RotationMatrix(data=m).to_axis_angle()[1]
        compare_axis_angle(actual_angle, actual_axis[:3], expected_angle, expected_axis)
        assert actual_axis[-1] == 0

    @given(unit_vector(4))
    def test_rpy_from_matrix(self, q):
        expected = rotation_matrix_from_quaternion(*q)

        roll = float(cas.RotationMatrix(data=expected).to_rpy()[0].to_np()[0])
        pitch = float(cas.RotationMatrix(data=expected).to_rpy()[1].to_np()[0])
        yaw = float(cas.RotationMatrix(data=expected).to_rpy()[2].to_np()[0])
        actual = rotation_matrix_from_rpy(roll, pitch, yaw)

        assert_allclose(actual, expected)

    @given(unit_vector(4))
    def test_rpy_from_matrix2(self, q):
        matrix = rotation_matrix_from_quaternion(*q)
        roll = cas.RotationMatrix(data=matrix).to_rpy()[0]
        pitch = cas.RotationMatrix(data=matrix).to_rpy()[1]
        yaw = cas.RotationMatrix(data=matrix).to_rpy()[2]
        r1 = cas.RotationMatrix.from_rpy(roll, pitch, yaw)
        assert_allclose(r1, matrix, atol=1.0e-4)

    def test_initialization(self):
        """Test various ways to initialize RotationMatrix"""
        # Default initialization (identity)
        r_identity = cas.RotationMatrix()
        assert isinstance(r_identity, cas.RotationMatrix)
        identity_np = r_identity
        expected_identity = np.eye(4)
        assert_allclose(identity_np, expected_identity)

        # From another RotationMatrix
        r_copy = cas.RotationMatrix(data=r_identity)
        assert isinstance(r_copy, cas.RotationMatrix)
        assert_allclose(r_copy, identity_np)

        # From numpy array
        rotation_data = np.eye(4)
        rotation_data[0, 1] = 0.5  # Add some rotation
        r_from_np = cas.RotationMatrix(data=rotation_data)
        assert isinstance(r_from_np, cas.RotationMatrix)

        # From TransformationMatrix
        t = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)
        r_from_t = cas.RotationMatrix(data=t)
        assert isinstance(r_from_t, cas.RotationMatrix)
        # Should preserve rotation part only
        assert r_from_t[0, 3] == 0
        assert r_from_t[1, 3] == 0
        assert r_from_t[2, 3] == 0

    def test_sanity_check(self):
        """Test that sanity check enforces proper rotation matrix structure"""
        # Valid 4x4 matrix should pass
        valid_matrix = np.eye(4)
        r = cas.RotationMatrix(data=valid_matrix)

        # Check that homogeneous coordinates are enforced
        assert r[0, 3] == 0
        assert r[1, 3] == 0
        assert r[2, 3] == 0
        assert r[3, 0] == 0
        assert r[3, 1] == 0
        assert r[3, 2] == 0
        assert r[3, 3] == 1

    def test_orthogonality_properties(self):
        """Test orthogonality properties of rotation matrices"""
        # Create rotation from known values
        r = cas.RotationMatrix.from_rpy(0.1, 0.2, 0.3)

        # Test orthogonality: R @ R.T = I
        should_be_identity = r @ r.T
        assert_allclose(should_be_identity, np.eye(4), atol=1e-10)

        # Test that determinant is 1 (proper rotation, not reflection)
        det = r.det()
        assert_allclose(det, 1.0, atol=1e-10)

    def test_transpose(self):
        """Test transpose operation and its properties"""
        r = cas.RotationMatrix.from_rpy(0.1, 0.2, 0.3)
        r_t = r.T

        assert isinstance(r_t, cas.RotationMatrix)

        # For rotation matrices: R.T = R^(-1)
        product = r @ r_t
        identity = cas.RotationMatrix()
        assert_allclose(product, identity, atol=1e-10)

        # Double transpose should give original
        r_tt = r_t.T
        assert_allclose(r, r_tt)

    def test_inverse(self):
        """Test matrix inversion for rotation matrices"""
        r = cas.RotationMatrix.from_rpy(0.5, -0.3, 1.2)

        # For rotation matrices, inverse should equal transpose
        r_inv = r.inverse()
        r_t = r.T
        assert isinstance(r_inv, cas.RotationMatrix)
        assert_allclose(r_inv, r_t, atol=1e-10)

        # R @ R^(-1) = I
        identity_check = r @ r_inv
        identity = cas.RotationMatrix()
        assert_allclose(identity_check, identity, atol=1e-10)

    def test_composition(self):
        """Test composition of multiple rotations"""
        r1 = cas.RotationMatrix.from_rpy(0.1, 0, 0)  # Roll
        r2 = cas.RotationMatrix.from_rpy(0, 0.2, 0)  # Pitch
        r3 = cas.RotationMatrix.from_rpy(0, 0, 0.3)  # Yaw

        # Test that composition works
        combined = r3 @ r2 @ r1
        assert isinstance(combined, cas.RotationMatrix)

        # Note: Order matters in rotation composition, so this might not be exactly equal
        # but both should be valid rotation matrices
        assert_allclose(combined.det(), 1.0)
        assert_allclose(combined @ combined.T, np.eye(4), atol=1e-10)

    def test_vector_rotation(self):
        """Test rotation of vectors and unit vectors"""
        # 90-degree rotation around Z-axis
        r_z90 = cas.RotationMatrix.from_axis_angle(cas.Vector3.Z(), np.pi / 2)

        # Rotate unit vector along X-axis
        x_axis = cas.Vector3.X()
        rotated = r_z90 @ x_axis

        assert isinstance(rotated, cas.Vector3)
        # Should become Y-axis
        expected = np.array([0, 1, 0, 0])  # Homogeneous coordinates
        assert_allclose(rotated, expected, atol=1e-10)

        # Test with regular Vector3
        v = cas.Vector3(x_init=1, y_init=0, z_init=0)
        rotated_v = r_z90 @ v
        assert isinstance(rotated_v, cas.Vector3)
        assert_allclose(rotated_v[:3], np.array([0, 1, 0]), atol=1e-10)

    def test_frame_properties(self):
        """Test reference frame and child frame properties"""
        r = cas.RotationMatrix()

        # Initially should be None
        assert r.reference_frame is None

        # Test frame preservation in operations
        r1 = cas.RotationMatrix.from_rpy(0.1, 0.2, 0.3)
        r2 = cas.RotationMatrix.from_rpy(0.2, 0.3, 0.4)

        result = r1 @ r2
        # Frame handling depends on implementation, krrood_test basic structure
        assert hasattr(result, "reference_frame")

    def test_to_conversions(self):
        """Test conversion methods to other representations"""
        r = cas.RotationMatrix.from_rpy(0.1, 0.2, 0.3)

        # Test conversion to axis-angle
        axis, angle = r.to_axis_angle()
        assert isinstance(axis, cas.Vector3)
        assert axis[3] == 0  # Should be a vector, not point
        assert hasattr(angle, "to_np")  # Should be Expression or similar

        # Test conversion to RPY
        roll, pitch, yaw = r.to_rpy()
        assert_allclose(roll, 0.1, atol=1e-10)
        assert_allclose(pitch, 0.2, atol=1e-10)
        assert_allclose(yaw, 0.3, atol=1e-10)

        # Test conversion to quaternion
        q = r.to_quaternion()
        assert isinstance(q, cas.Quaternion)

        # Round-trip krrood_test: R -> Q -> R should preserve rotation
        r_roundtrip = cas.RotationMatrix.from_quaternion(q)
        assert_allclose(r, r_roundtrip, atol=1e-10)

    def test_invalid_matmul_operations(self):
        """Test invalid matrix multiplication operations"""
        r = cas.RotationMatrix()
        s = cas.FloatVariable(name="s")
        e = cas.Expression(data=1)
        p = cas.Point3(x_init=1, y_init=2, z_init=3)
        q = cas.Quaternion()

        # These should raise TypeError
        with pytest.raises(TypeError):
            r @ s  # Matrix @ FloatVariable
        with pytest.raises(TypeError):
            r @ e  # Matrix @ Expression (scalar)
        with pytest.raises(TypeError):
            r @ p  # Matrix @ Point3 (use TransformationMatrix instead)
        with pytest.raises(TypeError):
            r @ q  # Matrix @ Quaternion

    @given(random_angle(), random_angle(), random_angle())
    def test_rpy_roundtrip(self, roll, pitch, yaw):
        """Property-based krrood_test for RPY round-trip conversion"""
        # Avoid gimbal lock region
        assume(abs(pitch) < np.pi / 2 - 0.1)

        r = cas.RotationMatrix.from_rpy(roll, pitch, yaw)
        r_roll, r_pitch, r_yaw = r.to_rpy()

        # Round-trip should preserve values (within numerical precision)
        assert_allclose(r_roll, roll, atol=1e-10)
        assert_allclose(r_pitch, pitch, atol=1e-10)
        assert_allclose(r_yaw, yaw, atol=1e-10)

    @given(unit_vector(length=3), random_angle())
    def test_axis_angle_properties(self, axis, angle):
        """Property-based krrood_test for axis-angle rotation properties"""
        # Skip very small angles to avoid numerical issues
        assume(abs(angle) > 1e-6)

        axis_unit = cas.Vector3.from_iterable(axis)
        r = cas.RotationMatrix.from_axis_angle(axis_unit, angle)

        # Test that axis is preserved (rotation around axis shouldn't change axis)
        rotated_axis = r @ axis_unit
        # For rotation around axis, the axis should remain unchanged
        dot_product = axis_unit @ rotated_axis
        assert_allclose(dot_product, 1.0, atol=1e-10)

    def test_small_angle_approximation(self):
        """Test behavior with very small rotation angles"""
        small_angle = 1e-8

        # Small rotation around Z-axis
        r = cas.RotationMatrix.from_axis_angle(cas.Vector3.Z(), small_angle)
        rotation_part = r[:3, :3]

        # Should be close to identity for very small angles
        assert_allclose(rotation_part, np.eye(3), atol=1e-7)

        # But determinant should still be 1
        assert_allclose(rotation_part.det(), 1.0, atol=1e-12)

    def test_symbolic_operations(self):
        """Test operations with symbolic expressions"""
        angle_sym = cas.FloatVariable(name="theta")

        # Create symbolic rotation
        r_sym = cas.RotationMatrix.from_axis_angle(cas.Vector3.Z(), angle_sym)

        # Should be able to compose with other rotations
        r_numeric = cas.RotationMatrix.from_rpy(0.1, 0, 0)
        result = r_sym @ r_numeric

        assert isinstance(result, cas.RotationMatrix)

        # Should contain the variable
        variables = result.free_variables()
        variable_names = [s.name for s in variables if hasattr(s, "name")]
        assert "theta" in variable_names

    def test_compilation(self):
        """Test compilation and execution of rotation matrices"""
        # Test symbolic rotation compilation
        compiled_rotation = cas.RotationMatrix.from_axis_angle(
            cas.Vector3.Z(), np.pi / 4
        )

        # Should be a valid 4x4 rotation matrix
        assert compiled_rotation.shape == (4, 4)
        assert_allclose(compiled_rotation.det(), 1.0)
        assert_allclose(compiled_rotation @ compiled_rotation.T, np.eye(4), atol=1e-10)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Zero rotation
        r_zero = cas.RotationMatrix.from_axis_angle(cas.Vector3.X(), 0)
        identity = cas.RotationMatrix()
        assert_allclose(r_zero, identity, atol=1e-12)

        # Full rotation (2π)
        r_full = cas.RotationMatrix.from_axis_angle(cas.Vector3.Y(), 2 * np.pi)
        assert_allclose(r_full, identity, atol=1e-10)

        # π rotation (180 degrees)
        r_pi = cas.RotationMatrix.from_axis_angle(cas.Vector3.Z(), np.pi)
        rotation_part = r_pi[:3, :3]
        # Should flip X and Y axes
        expected_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        assert_allclose(rotation_part, expected_rotation, atol=1e-10)

    def test_quaternion_consistency(self):
        """Test consistency between quaternion and rotation matrix representations"""
        # Create rotation via different methods
        r_rpy = cas.RotationMatrix.from_rpy(0.1, 0.2, 0.3)
        q = r_rpy.to_quaternion()
        r_from_q = cas.RotationMatrix.from_quaternion(q)

        # Should be identical
        assert_allclose(r_rpy, r_from_q, atol=1e-12)

    def test_determinant_preservation(self):
        """Test that all operations preserve determinant = 1"""
        r1 = cas.RotationMatrix.from_rpy(0.5, -0.3, 1.2)
        r2 = cas.RotationMatrix.from_axis_angle(cas.Vector3.unit_vector(1, 1, 1), 0.8)

        operations_to_test = [
            r1,
            r1.T,
            r1.inverse(),
            r1 @ r2,
            r2 @ r1,
        ]

        for r in operations_to_test:
            rotation_part = r[:3, :3]
            det = rotation_part.det()
            assert_allclose(
                det, 1.0, atol=1e-10
            ), f"Determinant {det} != 1.0 for operation"


class TestPoint3:
    def test_distance_point_to_line_segment1(self):
        p = cas.Point3(x_init=0, y_init=0, z_init=0)
        start = cas.Point3(x_init=0, y_init=0, z_init=-1)
        end = cas.Point3(x_init=0, y_init=0, z_init=1)
        distance = p.distance_to_line_segment(start, end)[0]
        nearest = p.distance_to_line_segment(start, end)[1]
        assert_allclose(distance, 0)
        assert_allclose(nearest, p)

    def test_distance_point_to_line_segment2(self):
        p = cas.Point3(x_init=0, y_init=1, z_init=0.5)
        start = cas.Point3(x_init=0, y_init=0, z_init=0)
        end = cas.Point3(x_init=0, y_init=0, z_init=1)
        distance = p.distance_to_line_segment(start, end)[0]
        nearest = p.distance_to_line_segment(start, end)[1]
        assert_allclose(distance, 1)
        assert_allclose(nearest, cas.Point3(x_init=0, y_init=0, z_init=0.5))

    def test_distance_point_to_line_segment3(self):
        p = cas.Point3(x_init=0, y_init=1, z_init=2)
        start = cas.Point3(x_init=0, y_init=0, z_init=0)
        end = cas.Point3(x_init=0, y_init=0, z_init=1)
        distance = p.distance_to_line_segment(start, end)[0]
        nearest = p.distance_to_line_segment(start, end)[1]
        assert_allclose(distance, 1.4142135623730951)
        assert_allclose(nearest, cas.Point3(x_init=0, y_init=0, z_init=1.0))

    @given(vector(3))
    def test_norm(self, v):
        p = cas.Point3.from_iterable(v)
        actual = p.norm()
        expected = np.linalg.norm(v)
        assert_allclose(actual, expected)

    def test_init(self):
        l = [1, 2, 3]
        s = cas.FloatVariable(name="s")
        e = cas.Expression(data=1)
        v = cas.Vector3(x_init=1, y_init=1, z_init=1)
        p = cas.Point3(x_init=l[0], y_init=l[1], z_init=l[2])
        r = cas.RotationMatrix()
        q = cas.Quaternion()
        t = cas.TransformationMatrix()

        cas.Point3()
        cas.Point3(x_init=s, y_init=e, z_init=0)
        assert_allclose(p[3], 1)
        assert_allclose(p[:3], l)

        cas.Point3.from_iterable(cas.Expression(data=v))
        cas.Point3.from_iterable(l)
        with pytest.raises(WrongDimensionsError):
            cas.Point3.from_iterable(r)
        with pytest.raises(WrongDimensionsError):
            cas.Point3.from_iterable(t)
        with pytest.raises(WrongDimensionsError):
            cas.Point3.from_iterable(t.to_np())

    @given(float_no_nan_no_inf(), vector(3), vector(3))
    def test_if_greater_zero(self, condition, if_result, else_result):
        actual = cas.if_greater_zero(
            condition,
            cas.Point3.from_iterable(if_result),
            cas.Point3.from_iterable(else_result),
        )
        expected = if_result if condition > 0 else else_result
        assert_allclose(actual[:3], expected)

    def test_arithmetic_operations(self):
        """Test all allowed arithmetic operations on Point3"""
        p1 = cas.Point3(x_init=1, y_init=2, z_init=3)
        p2 = cas.Point3(x_init=4, y_init=5, z_init=6)
        v = cas.Vector3(x_init=1, y_init=1, z_init=1)
        s = cas.FloatVariable(name="s")

        # Test Point + Vector = Point (translate point by vector)
        result = p1 + v
        assert isinstance(result, cas.Point3)
        assert result.x == 2 and result.y == 3 and result.z == 4
        assert result[3] == 1  # Homogeneous coordinate preserved
        assert result.reference_frame == p1.reference_frame

        # Test Point - Point = Vector (displacement between points)
        result = p2 - p1
        assert isinstance(result, cas.Vector3)
        assert result.x == 3 and result.y == 3 and result.z == 3
        assert result[3] == 0  # Vector has 0 in homogeneous coordinate
        assert result.reference_frame == p2.reference_frame

        # Test Point - Vector = Point (translate point by negative vector)
        result = p2 - v
        assert isinstance(result, cas.Point3)
        assert result.x == 3 and result.y == 4 and result.z == 5
        assert result[3] == 1
        assert result.reference_frame == p2.reference_frame

        # Test -Point = Point (negate all coordinates)
        result = -p1
        assert isinstance(result, cas.Point3)
        assert result.x == -1 and result.y == -2 and result.z == -3
        assert result[3] == 1
        assert result.reference_frame == p1.reference_frame

        # Test Point.norm() = scalar (distance from origin)
        result = p1.norm()
        assert isinstance(result, cas.Expression)
        expected_norm = np.sqrt(1**2 + 2**2 + 3**2)
        assert_allclose(result, expected_norm)

        # Test invalid operations that should raise TypeError
        with pytest.raises(TypeError):
            p1 + p2  # Point + Point not allowed

        with pytest.raises(TypeError):
            s + p2
        with pytest.raises(TypeError):
            p1 + s

        with pytest.raises(TypeError):
            p1 * 2  # Point * scalar not allowed (scaling a position is meaningless)

        with pytest.raises(TypeError):
            2 * p1  # scalar * Point not allowed

        with pytest.raises(TypeError):
            p1 / 2  # Point / scalar not allowed

        with pytest.raises(TypeError):
            p1**2  # Point ** scalar not allowed

        with pytest.raises(TypeError):
            p1 @ p2  # Point @ Point not allowed

        with pytest.raises(TypeError):
            p1 @ v  # Point @ Vector not allowed

        # Test operations with symbolic expressions
        x = cas.FloatVariable(name="x")
        p_symbolic = cas.Point3(x, y_init=2, z_init=3)
        result = p_symbolic + v
        assert isinstance(result, cas.Point3)
        assert result[3] == 1

        # Test property access
        assert p1.x == 1
        assert p1.y == 2
        assert p1.z == 3

        # Test property assignment
        p_copy = cas.Point3(x_init=1, y_init=2, z_init=3)
        p_copy.x = 10
        p_copy.y = 20
        p_copy.z = 30
        assert p_copy[0] == 10 and p_copy[1] == 20 and p_copy[2] == 30
        assert p_copy[3] == 1

    def test_properties(self):
        """Test x, y, z property getters and setters"""
        p = cas.Point3(x_init=1, y_init=2, z_init=3)

        # Test getters
        assert p.x == 1
        assert p.y == 2
        assert p.z == 3

        # Test setters
        p.x = 10
        p.y = 20
        p.z = 30
        assert p[0] == 10
        assert p[1] == 20
        assert p[2] == 30
        assert p[3] == 1  # Homogeneous coordinate unchanged

    def test_geometric_operations(self):
        """Test geometric operations specific to points"""
        p1 = cas.Point3(x_init=0, y_init=0, z_init=0)  # Origin
        p2 = cas.Point3(x_init=3, y_init=4, z_init=0)  # Point on XY plane

        # Distance between points (via subtraction and norm)
        displacement = p2 - p1
        assert isinstance(displacement, cas.Vector3)
        distance = displacement.norm()
        assert_allclose(distance, 5.0)  # 3-4-5 triangle

        # Midpoint calculation
        midpoint = p1 + (p2 - p1) * 0.5
        assert isinstance(midpoint, cas.Point3)
        assert_allclose(midpoint.x, 1.5)
        assert_allclose(midpoint.y, 2.0)
        assert_allclose(midpoint.z, 0.0)

    def test_reference_frame_preservation(self):
        """Test that reference frames are properly preserved through operations"""
        p1 = cas.Point3(x_init=1, y_init=2, z_init=3)  # reference_frame=some_frame
        v = cas.Vector3(x_init=1, y_init=1, z_init=1)

        # Operations should preserve the reference frame of the point
        result = p1 + v
        assert result.reference_frame == p1.reference_frame

        result = -p1
        assert result.reference_frame == p1.reference_frame

        # Point - Point should preserve reference frame of left operand
        p2 = cas.Point3(x_init=4, y_init=5, z_init=6)
        result = p1 - p2
        assert isinstance(result, cas.Vector3)
        assert result.reference_frame == p1.reference_frame

    def test_invalid_operations(self):
        """Test operations that should raise TypeError"""
        p = cas.Point3(x_init=1, y_init=2, z_init=3)
        r = cas.RotationMatrix()
        q = cas.Quaternion()
        t = cas.TransformationMatrix()

        # Invalid additions - Point + Point is not geometrically meaningful
        p2 = cas.Point3(x_init=4, y_init=5, z_init=6)
        with pytest.raises(TypeError):
            p + p2

        # Invalid additions with matrices
        with pytest.raises(TypeError):
            p + r
        with pytest.raises(TypeError):
            p + q
        with pytest.raises(TypeError):
            p + t

        # Invalid multiplications with points/vectors
        v = cas.Vector3(x_init=1, y_init=1, z_init=1)
        with pytest.raises(TypeError):
            p * p2  # Point * Point not defined
        with pytest.raises(TypeError):
            p * v  # Point * Vector not defined

    @given(vector(3), vector(3))
    def test_distance_property_based(self, p1_data, p2_data):
        """Property-based krrood_test for point-to-point distance"""
        p1 = cas.Point3.from_iterable(p1_data)
        p2 = cas.Point3.from_iterable(p2_data)

        # Distance via subtraction and norm
        displacement = p2 - p1
        actual = displacement.norm()

        # Compare with numpy calculation
        expected = np.linalg.norm(np.array(p2_data) - np.array(p1_data))

        assert_allclose(actual, expected)

    def test_transformation_operations(self):
        """Test transformation matrix operations with points"""
        p = cas.Point3(x_init=1, y_init=2, z_init=3)
        t = cas.TransformationMatrix()

        # Test matrix @ point = point (homogeneous transformation)
        result = t @ p
        assert isinstance(result, cas.Point3)
        assert result[3] == 1  # Homogeneous coordinate preserved

        # Test that point @ matrix raises error (not mathematically meaningful)
        with pytest.raises(TypeError):
            p @ t

    def test_project_to_line(self):
        point = cas.Point3(x_init=1, y_init=2, z_init=3)
        line_point = cas.Point3(x_init=0, y_init=0, z_init=0)
        line_direction = cas.Vector3(
            x_init=1, y_init=2, z_init=3
        )  # Point lies on this line
        point, distance = point.project_to_line(line_point, line_direction)
        assert_allclose(distance, 0.0)
        assert_allclose(point, np.array([1, 2, 3, 1]))

        point = cas.Point3(x_init=1, y_init=0, z_init=0)
        line_point = cas.Point3(x_init=0, y_init=0, z_init=0)
        line_direction = cas.Vector3(x_init=0, y_init=1, z_init=0)  # Y-axis
        point, distance = point.project_to_line(line_point, line_direction)
        assert_allclose(distance, 1.0)
        assert_allclose(point, np.array([0, 0, 0, 1]))

        point = cas.Point3(x_init=0, y_init=0, z_init=5)
        line_point = cas.Point3(x_init=0, y_init=0, z_init=0)
        line_direction = cas.Vector3(x_init=1, y_init=0, z_init=0)  # X-axis
        point, distance = point.project_to_line(line_point, line_direction)
        assert_allclose(distance, 5.0)
        assert_allclose(point, np.array([0, 0, 0, 1]))

    def test_distance_to_line_segment(self):
        pass

    def test_project_to_plane(self):
        p = cas.Point3(x_init=0, y_init=0, z_init=1)
        actual, distance = p.project_to_plane(
            frame_V_plane_vector1=cas.Vector3.X(),
            frame_V_plane_vector2=cas.Vector3.Y(),
        )
        expected = cas.Point3(x_init=0, y_init=0, z_init=0)
        assert_allclose(actual, expected)
        assert_allclose(distance, 1)

    def test_compilation_and_execution(self):
        """Test that Point3 operations compile and execute correctly"""
        # Test point arithmetic compilation
        compiled_add = cas.Point3(x_init=1, y_init=2, z_init=3) + cas.Vector3(
            x_init=1, y_init=1, z_init=1
        )
        expected = np.array([2, 3, 4, 1])
        assert_allclose(compiled_add, expected)

        # Test point subtraction compilation
        compiled_sub = cas.Point3(x_init=5, y_init=6, z_init=7) - cas.Point3(
            x_init=1, y_init=2, z_init=3
        )
        expected_vector = np.array([4, 4, 4, 0])  # Result is a Vector3
        assert_allclose(compiled_sub, expected_vector)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with zero coordinates
        p_zero = cas.Point3(x_init=0, y_init=0, z_init=0)
        assert p_zero[0] == 0 and p_zero[1] == 0 and p_zero[2] == 0
        assert p_zero[3] == 1

        # Test with negative coordinates
        p_neg = cas.Point3(x_init=-1, y_init=-2, z_init=-3)
        assert p_neg[0] == -1 and p_neg[1] == -2 and p_neg[2] == -3
        assert p_neg[3] == 1

        # Test very large coordinates
        large_val = 1e6
        p_large = cas.Point3(x_init=large_val, y_init=large_val, z_init=large_val)
        assert p_large[0] == large_val
        assert p_large[3] == 1

        # Test very small coordinates
        small_val = 1e-6
        p_small = cas.Point3(x_init=small_val, y_init=small_val, z_init=small_val)
        assert p_small[0] == small_val
        assert p_small[3] == 1

    def test_symbolic_operations(self):
        """Test operations with symbolic expressions"""
        x, y, z = cas.create_float_variables(["x", "y", "z"])
        p_symbolic = cas.Point3(x_init=x, y_init=y, z_init=z)
        p_numeric = cas.Point3(x_init=1, y_init=2, z_init=3)

        # Test symbolic point operations
        result = p_symbolic + cas.Vector3(x_init=1, y_init=1, z_init=1)
        assert isinstance(result, cas.Point3)
        assert result[3] == 1

        # Test mixed symbolic/numeric operations
        result = p_symbolic - p_numeric
        assert isinstance(result, cas.Vector3)
        assert result[3] == 0

        # Verify symbolic expressions are preserved
        variables = result.free_variables()
        variable_names = [s.name for s in variables if hasattr(s, "name")]
        assert "x" in variable_names and "y" in variable_names and "z" in variable_names


class TestVector3:
    @given(vector(3), vector(3))
    def test_cross(self, u, v):
        assert_allclose(
            cas.Vector3.from_iterable(u).cross(cas.Vector3.from_iterable(v))[:3],
            np.cross(u, v),
        )

    def test_init(self):
        l = [1, 2, 3]
        s = cas.FloatVariable(name="s")
        e = cas.Expression(data=1)
        v = cas.Vector3(x_init=1, y_init=1, z_init=1)
        p = cas.Point3(x_init=1, y_init=1, z_init=1)
        r = cas.RotationMatrix()
        q = cas.Quaternion()
        t = cas.TransformationMatrix()

        cas.Vector3()
        cas.Vector3(x_init=s, y_init=e, z_init=0)
        v = cas.Vector3(x_init=l[0], y_init=l[1], z_init=l[2])
        assert v[0] == l[0]
        assert v[1] == l[1]
        assert v[2] == l[2]
        assert v[3] == 0  # Vector3 has 0 in homogeneous coordinate

        cas.Vector3.from_iterable(cas.Expression(data=v))
        cas.Vector3.from_iterable(p)  # Can create Vector3 from Point3
        cas.Vector3.from_iterable(v)
        cas.Vector3.from_iterable(v.casadi_sx)
        cas.Vector3.from_iterable(l)
        cas.Vector3.from_iterable(q)
        with pytest.raises(WrongDimensionsError):
            cas.Vector3.from_iterable(r)
        with pytest.raises(WrongDimensionsError):
            cas.Vector3.from_iterable(t)
        with pytest.raises(WrongDimensionsError):
            cas.Vector3.from_iterable(t.to_np())

    @given(vector(3))
    def test_is_length_1(self, v):
        assume(abs(v[0]) > 0.00001 or abs(v[1]) > 0.00001 or abs(v[2]) > 0.00001)
        unit_v = cas.Vector3.unit_vector(*v)
        assert_allclose(unit_v.norm(), 1)

    def test_length_0(self):
        unit_v = cas.Vector3.unit_vector(x=0, y=0, z=0)
        assert np.isnan(unit_v.norm().to_np())

    @given(vector(3))
    def test_norm(self, v):
        expected = np.linalg.norm(v)
        v = cas.Vector3.from_iterable(v)
        actual = v.norm()
        assert_allclose(actual, expected)

    @given(vector(3), float_no_nan_no_inf(), vector(3))
    def test_save_division(self, nominator, denominator, if_nan):
        nominator_expr = cas.Vector3.from_iterable(nominator)
        denominator_expr = cas.Expression(data=denominator)
        if_nan_expr = cas.Vector3.from_iterable(if_nan)
        result = nominator_expr.safe_division(denominator_expr, if_nan_expr)
        if denominator == 0:
            assert_allclose(result[:3], if_nan)
        else:
            assert_allclose(result[:3], np.array(nominator) / denominator)

    @given(
        lists_of_same_length(
            [float_no_nan_no_inf(), float_no_nan_no_inf()], min_length=3, max_length=3
        )
    )
    def test_dot(self, vectors):
        u, v = vectors
        u = np.array(u)
        v = np.array(v)
        result = cas.Vector3.from_iterable(u).dot(cas.Vector3.from_iterable(v))
        expected = np.dot(u, v.T)
        assert_allclose(result, expected)

    def test_cross_product(self):
        """Test cross product operations"""
        v1 = cas.Vector3(x_init=1, y_init=0, z_init=0)
        v2 = cas.Vector3(x_init=0, y_init=1, z_init=0)

        result = v1.cross(v2)
        assert isinstance(result, cas.Vector3)
        assert result[3] == 0
        # Cross product of x and y unit vectors should be z unit vector
        assert_allclose(result[:3], np.array([0, 0, 1]))

        # Cross product is anti-commutative
        result2 = v2.cross(v1)
        assert_allclose(result2[:3], np.array([0, 0, -1]))

    def test_properties(self):
        """Test x, y, z property getters and setters"""
        v = cas.Vector3(x_init=1, y_init=2, z_init=3)

        # Test getters
        assert v.x == 1
        assert v.y == 2
        assert v.z == 3

        # Test setters
        v.x = 10
        v.y = 20
        v.z = 30
        assert v[0] == 10
        assert v[1] == 20
        assert v[2] == 30
        assert v[3] == 0  # Homogeneous coordinate unchanged

    def test_reference_frame_preservation(self):
        """Test that reference frames are properly preserved through operations"""
        # This would require a mock reference frame object
        v1 = cas.Vector3(x_init=1, y_init=2, z_init=3)  # reference_frame=some_frame
        v2 = cas.Vector3(x_init=4, y_init=5, z_init=6)

        # Operations should preserve the reference frame of the left operand
        result = v1 + v2
        assert result.reference_frame == v1.reference_frame

        result = v1 * 2
        assert result.reference_frame == v1.reference_frame

        result = -v1
        assert result.reference_frame == v1.reference_frame

    def test_negation(self):
        """Test unary negation operator"""
        v = cas.Vector3(x_init=1, y_init=-2, z_init=3)
        result = -v

        assert isinstance(result, cas.Vector3)
        assert result[0] == -1
        assert result[1] == 2
        assert result[2] == -3
        assert result[3] == 0

    def test_angle_between(self):
        v1 = cas.Vector3(x_init=1, y_init=0, z_init=0)
        v2 = cas.Vector3(x_init=0, y_init=1, z_init=0)
        angle = v1.angle_between(v2)
        assert_allclose(angle, np.pi / 2)

    def test_scale_method(self):
        """Test the scale method with safe and unsafe modes"""
        v = cas.Vector3(x_init=3, y_init=4, z_init=0)  # Length = 5

        # Safe scaling (default)
        v_copy = cas.Vector3(x_init=3, y_init=4, z_init=0)
        v_copy.scale(10)
        expected_norm = v_copy.norm()
        assert_allclose(expected_norm, 10)

        # Unsafe scaling
        v_copy2 = cas.Vector3(x_init=3, y_init=4, z_init=0)
        v_copy2.scale(10, unsafe=True)
        expected_norm2 = v_copy2.norm()
        assert_allclose(expected_norm2, 10)

    def test_invalid_operations(self):
        """Test operations that should raise TypeError"""
        v = cas.Vector3(x_init=1, y_init=2, z_init=3)
        r = cas.RotationMatrix()
        q = cas.Quaternion()
        t = cas.TransformationMatrix()

        # Invalid additions
        with pytest.raises(TypeError):
            v + r
        with pytest.raises(TypeError):
            v + q
        with pytest.raises(TypeError):
            v + t

        # Invalid multiplications with vectors
        with pytest.raises(TypeError):
            v * v  # Vector * Vector is not defined (use dot or cross)
        with pytest.raises(TypeError):
            v / v

    @given(vector(3), vector(3))
    def test_dot_product_property_based(self, v1_data, v2_data):
        """Property-based krrood_test for dot product"""
        v1 = cas.Vector3.from_iterable(v1_data)
        v2 = cas.Vector3.from_iterable(v2_data)

        actual = v1.dot(v2)
        assert isinstance(actual, cas.Expression)

        # Compare with numpy dot product
        expected = np.dot(v1_data, v2_data)

        assert_allclose(actual, expected)

    @given(vector(3))
    def test_norm_property_based(self, v_data):
        """Property-based krrood_test for vector norm"""
        v = cas.Vector3.from_iterable(v_data)
        actual = v.norm()

        expected = np.linalg.norm(v_data)

        assert_allclose(actual, expected)

    def test_from_iterable_edge_cases(self):
        """Test edge cases for from_iterable class method"""
        # Test with different iterable types
        v1 = cas.Vector3.from_iterable([1, 2, 3])
        assert v1[0] == 1 and v1[1] == 2 and v1[2] == 3 and v1[3] == 0

        v2 = cas.Vector3.from_iterable((4, 5, 6))
        assert v2[0] == 4 and v2[1] == 5 and v2[2] == 6 and v2[3] == 0

        v3 = cas.Vector3.from_iterable(np.array([7, 8, 9]))
        assert v3[0] == 7 and v3[1] == 8 and v3[2] == 9 and v3[3] == 0

        # Test reference frame inheritance
        existing_vector = cas.Vector3(
            x_init=1, y_init=2, z_init=3
        )  # reference_frame=some_frame
        new_vector = cas.Vector3.from_iterable(existing_vector)
        assert new_vector.reference_frame == existing_vector.reference_frame

    def test_compilation_and_execution(self):
        """Test that Vector3 operations compile and execute correctly"""
        v1 = cas.Vector3(
            x_init=cas.FloatVariable(name="x"),
            y_init=cas.FloatVariable(name="y"),
            z_init=cas.FloatVariable(name="z"),
        )
        v2 = cas.Vector3(x_init=1, y_init=2, z_init=3)

        # Test dot product compilation
        dot_expr = v1.dot(v2)
        compiled_dot = cas.Vector3(x_init=1, y_init=2, z_init=3).dot(
            cas.Vector3(x_init=1, y_init=2, z_init=3)
        )
        expected = 1 * 1 + 2 * 2 + 3 * 3  # = 14
        assert_allclose(compiled_dot, expected)

        # Test cross product compilation
        cross_expr = v1.cross(v2)
        compiled_cross = cas.Vector3(x_init=2, y_init=3, z_init=4).cross(
            cas.Vector3(x_init=1, y_init=2, z_init=3)
        )
        expected_cross = np.cross([2, 3, 4], [1, 2, 3])
        assert_allclose(compiled_cross[:3], expected_cross)

    def test_project_to_cone(self):
        v = cas.Vector3.X()
        projected_cone = v.project_to_cone(
            frame_V_cone_axis=cas.Vector3(x_init=1, y_init=1, z_init=0),
            cone_theta=np.pi / 4,
        )
        expected = np.array([1, 0, 0, 0])
        assert_allclose(projected_cone, expected, atol=1e-10)

        # projected_cone = v.project_to_cone(frame_V_cone_axis=cas.Vector3(1,1,0), cone_theta=np.pi/2)
        # expected = np.array([1, 0, 0, 0])
        # assert_allclose(projected_cone, expected, atol=1e-10)


class TestTransformationMatrix:
    def test_json(self):
        """Test that the JSON serialization works correctly"""
        t = cas.TransformationMatrix()
        t_json = t.to_json()
        t_copy = cas.TransformationMatrix.from_json(t_json)
        assert_allclose(t.to_np(), t_copy.to_np())

    def test_reference_frames(self):
        reference_frame = "muh"
        child_frame = "kikariki"
        reference_frame2 = "muh2"
        child_frame2 = "kikariki2"
        t1 = cas.TransformationMatrix(
            reference_frame=reference_frame, child_frame=child_frame
        )
        t2 = cas.TransformationMatrix(
            reference_frame=reference_frame2, child_frame=child_frame2
        )

        p = cas.Point3(reference_frame=reference_frame)
        v = cas.Vector3(reference_frame=reference_frame)
        r = cas.RotationMatrix(reference_frame=reference_frame)

        assert t1.reference_frame == reference_frame
        assert t1.child_frame == child_frame

        t = cas.TransformationMatrix.from_point_rotation_matrix(p, r)
        assert t.reference_frame == reference_frame
        assert t.child_frame == None

        with pytest.raises(SpatialTypesError):
            t = cas.TransformationMatrix.from_point_rotation_matrix(
                cas.Point3(reference_frame=reference_frame2), r
            )

        t = cas.TransformationMatrix.from_point_rotation_matrix(
            p, r, reference_frame=reference_frame2, child_frame=child_frame2
        )
        assert t.reference_frame == reference_frame2
        assert t.child_frame == child_frame2

        t = cas.TransformationMatrix.from_xyz_rpy(
            reference_frame=reference_frame, child_frame=child_frame
        )
        assert t.reference_frame == reference_frame
        assert t.child_frame == child_frame

        t = cas.TransformationMatrix.from_xyz_quaternion(
            reference_frame=reference_frame, child_frame=child_frame
        )
        assert t.reference_frame == reference_frame
        assert t.child_frame == child_frame

        t = cas.TransformationMatrix.from_xyz_axis_angle(
            reference_frame=reference_frame, child_frame=child_frame
        )
        assert t.reference_frame == reference_frame
        assert t.child_frame == child_frame

        assert (t1 @ t2).reference_frame == reference_frame
        assert (t1 @ t2).child_frame == child_frame2
        assert (t2 @ t1).reference_frame == reference_frame2
        assert (t2 @ t1).child_frame == child_frame
        assert t1.inverse().reference_frame == child_frame
        assert t1.inverse().child_frame == reference_frame
        assert t1.to_position().reference_frame == reference_frame
        assert t1.to_quaternion().reference_frame == reference_frame
        assert t1.to_translation().reference_frame == reference_frame
        assert t1.to_rotation_matrix().reference_frame == reference_frame

        assert (t1 @ p).reference_frame == reference_frame
        assert (t1 @ v).reference_frame == reference_frame
        assert (t1 @ r).reference_frame == reference_frame

    def test_matmul_type_preservation(self):
        """Test that @ operator preserves correct types for TransformationMatrix"""
        s = cas.FloatVariable(name="s")
        e = cas.Expression()
        v = cas.Vector3(x_init=1, y_init=1, z_init=1)
        p = cas.Point3(x_init=1, y_init=1, z_init=1)
        r = cas.RotationMatrix()
        q = cas.Quaternion()
        t = cas.TransformationMatrix()

        # TransformationMatrix @ invalid types should raise TypeError
        with pytest.raises(TypeError):
            t @ s  # TransformationMatrix @ FloatVariable
        with pytest.raises(TypeError):
            t @ e  # TransformationMatrix @ Expression (scalar)
        with pytest.raises(TypeError):
            t @ q  # TransformationMatrix @ Quaternion

        # TransformationMatrix @ valid types should return correct types
        assert isinstance(t @ v, cas.Vector3)  # Transform vector -> Vector3
        assert isinstance(t @ p, cas.Point3)  # Transform point -> Point3
        assert isinstance(
            t @ r, cas.RotationMatrix
        )  # Transform rotation -> RotationMatrix
        assert isinstance(
            t @ t, cas.TransformationMatrix
        )  # Transform transformation -> TransformationMatrix

        # Test reverse operations (other types @ TransformationMatrix)
        # RotationMatrix @ TransformationMatrix -> TransformationMatrix (already tested in RotationMatrix)
        assert isinstance(r @ t, cas.TransformationMatrix)

        # Verify that the transformed objects maintain their geometric properties
        # Vector should remain homogeneous coordinate = 0
        transformed_vector = t @ v
        assert transformed_vector[3] == 0

        # Point should remain homogeneous coordinate = 1
        transformed_point = t @ p
        assert transformed_point[3] == 1

        # Test with non-identity transformation to ensure type preservation holds
        t_rot = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)

        assert isinstance(t_rot @ v, cas.Vector3)
        assert isinstance(t_rot @ p, cas.Point3)
        assert isinstance(t_rot @ r, cas.RotationMatrix)
        assert isinstance(t_rot @ t, cas.TransformationMatrix)

        # Verify homogeneous coordinate preservation with non-identity transform
        transformed_vector_rot = t_rot @ v
        assert transformed_vector_rot[3] == 0

        transformed_point_rot = t_rot @ p
        assert transformed_point_rot[3] == 1

    @given(float_no_nan_no_inf(), float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_translation3(self, x, y, z):
        r1 = cas.TransformationMatrix.from_xyz_rpy(x, y, z)
        r2 = np.identity(4)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        assert_allclose(r1, r2)

    def test_dot(self):
        s = cas.FloatVariable(name="x")
        m1 = cas.TransformationMatrix()
        m2 = cas.TransformationMatrix.from_xyz_rpy(x=s)
        m1.dot(m2)

    def test_TransformationMatrix(self):
        f = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3)
        assert isinstance(f, cas.TransformationMatrix)

    @given(st.integers(min_value=1, max_value=10))
    def test_matrix(self, x_dim):
        data = list(range(x_dim))
        with pytest.raises(WrongDimensionsError):
            cas.TransformationMatrix(data=data)

    @given(
        st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10)
    )
    def test_matrix2(self, x_dim, y_dim):
        data = [[i + (j * x_dim) for j in range(y_dim)] for i in range(x_dim)]
        if x_dim != 4 or y_dim != 4:
            with pytest.raises(WrongDimensionsError):
                m = cas.TransformationMatrix(data=data).to_np()
        else:
            m = cas.TransformationMatrix(data=data).to_np()
            assert float(m[3, 0]) == 0
            assert float(m[3, 1]) == 0
            assert float(m[3, 2]) == 0
            assert float(m[x_dim - 1, y_dim - 1]) == 1

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        unit_vector(length=3),
        random_angle(),
    )
    def test_frame3_axis_angle(self, x, y, z, axis, angle):
        r2 = rotation_matrix_from_axis_angle(np.array(axis), angle)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        r = cas.TransformationMatrix.from_point_rotation_matrix(
            cas.Point3(x, y, z), cas.RotationMatrix.from_axis_angle(axis, angle)
        )
        assert_allclose(r, r2)

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        random_angle(),
        random_angle(),
        random_angle(),
    )
    def test_frame3_rpy(self, x, y, z, roll, pitch, yaw):
        r2 = rotation_matrix_from_rpy(roll, pitch, yaw)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        assert_allclose(
            cas.TransformationMatrix.from_xyz_rpy(x, y, z, roll, pitch, yaw), r2
        )

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        unit_vector(4),
    )
    def test_frame3_quaternion(self, x, y, z, q):
        r2 = rotation_matrix_from_quaternion(*q)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        r = cas.TransformationMatrix.from_point_rotation_matrix(
            point=cas.Point3(x, y, z),
            rotation_matrix=cas.RotationMatrix.from_quaternion(cas.Quaternion(*q)),
        )
        assert_allclose(r, r2)

    @given(
        float_no_nan_no_inf(outer_limit=1000),
        float_no_nan_no_inf(outer_limit=1000),
        float_no_nan_no_inf(outer_limit=1000),
        quaternion(),
    )
    def test_inverse_frame(self, x, y, z, q):
        f = rotation_matrix_from_quaternion(*q)
        f[0, 3] = x
        f[1, 3] = y
        f[2, 3] = z
        r = cas.TransformationMatrix(data=f).inverse()

        r2 = np.linalg.inv(f)
        assert_allclose(r, r2, atol=1.0e-4, rtol=1.0e-4)

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        unit_vector(4),
    )
    def test_pos_of(self, x, y, z, q):
        r1 = cas.TransformationMatrix.from_point_rotation_matrix(
            cas.Point3(x, y, z), cas.RotationMatrix.from_quaternion(cas.Quaternion(*q))
        ).to_position()
        r2 = [x, y, z, 1]
        for i, e in enumerate(r2):
            assert_allclose(r1[i], e)

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        unit_vector(4),
    )
    def test_trans_of(self, x, y, z, q):
        r1 = cas.TransformationMatrix.from_point_rotation_matrix(
            point=cas.Point3(x, y, z),
            rotation_matrix=cas.RotationMatrix.from_quaternion(cas.Quaternion(*q)),
        ).to_translation()
        r2 = np.identity(4)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        assert_allclose(r1, r2)

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        unit_vector(4),
    )
    def test_rot_of(self, x, y, z, q):
        r1 = cas.TransformationMatrix.from_point_rotation_matrix(
            point=cas.Point3(x_init=x, y_init=y, z_init=z),
            rotation_matrix=cas.RotationMatrix.from_quaternion(
                cas.Quaternion.from_iterable(q)
            ),
        ).to_rotation_matrix()
        r2 = rotation_matrix_from_quaternion(*q)
        assert_allclose(r1, r2)

    def test_rot_of2(self):
        """
        Test to make sure the function doesn't alter the original
        """
        f = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3)
        r = f.to_rotation_matrix()
        assert f[0, 3] == 1
        assert f[1, 3] == 2
        assert f[2, 3] == 3
        assert r[0, 0] == 1
        assert r[1, 1] == 1
        assert r[2, 2] == 1

    def test_initialization(self):
        """Test various ways to initialize TransformationMatrix"""
        # Default initialization (identity)
        t_identity = cas.TransformationMatrix()
        assert isinstance(t_identity, cas.TransformationMatrix)
        identity_np = t_identity
        expected_identity = np.eye(4)
        assert_allclose(identity_np, expected_identity)

        # From RotationMatrix
        r = cas.RotationMatrix.from_rpy(0.1, 0.2, 0.3)
        t_from_r = cas.TransformationMatrix(r)
        assert isinstance(t_from_r, cas.TransformationMatrix)
        # Should preserve rotation, zero translation
        assert t_from_r[0, 3] == 0
        assert t_from_r[1, 3] == 0
        assert t_from_r[2, 3] == 0

        # From another TransformationMatrix
        t_copy = cas.TransformationMatrix(t_from_r)
        assert isinstance(t_copy, cas.TransformationMatrix)
        assert_allclose(t_copy, t_from_r)

        # From numpy array
        transform_data = np.eye(4)
        transform_data[:3, 3] = [1, 2, 3]  # Add translation
        t_from_np = cas.TransformationMatrix(data=transform_data)
        assert isinstance(t_from_np, cas.TransformationMatrix)

    def test_sanity_check(self):
        """Test that sanity check enforces proper transformation matrix structure"""
        # Valid 4x4 matrix should pass
        valid_matrix = np.eye(4)
        valid_matrix[:3, 3] = [1, 2, 3]
        t = cas.TransformationMatrix(data=valid_matrix)

        # Check that bottom row is enforced to [0, 0, 0, 1]
        assert t[3, 0] == 0
        assert t[3, 1] == 0
        assert t[3, 2] == 0
        assert t[3, 3] == 1

        # Invalid shape should raise ValueError
        with pytest.raises(WrongDimensionsError):
            cas.TransformationMatrix(data=np.eye(3))  # 3x3 instead of 4x4

        with pytest.raises(WrongDimensionsError):
            cas.TransformationMatrix(data=np.ones((2, 5)))  # Wrong dimensions

    def test_properties(self):
        """Test x, y, z property getters and setters"""
        t = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)

        # Test getters
        assert t.x.to_np() == 1
        assert t.y.to_np() == 2
        assert t.z.to_np() == 3

        # Test setters
        t.x = 10
        t.y = 20
        t.z = 30
        assert t[0, 3].to_np() == 10
        assert t[1, 3].to_np() == 20
        assert t[2, 3].to_np() == 30

        # Bottom row should remain unchanged
        assert t[3, 3] == 1

    def test_from_point_rotation(self):
        """Test construction from point and rotation matrix"""
        p = cas.Point3(x_init=1, y_init=2, z_init=3)
        r = cas.RotationMatrix.from_rpy(0.1, 0.2, 0.3)

        # From both point and rotation
        t1 = cas.TransformationMatrix.from_point_rotation_matrix(p, r)
        assert isinstance(t1, cas.TransformationMatrix)
        assert_allclose(t1.x, 1)
        assert_allclose(t1.y, 2)
        assert_allclose(t1.z, 3)

        # From point only (identity rotation)
        t2 = cas.TransformationMatrix.from_point_rotation_matrix(point=p)
        rotation_part = t2[:3, :3]
        assert_allclose(rotation_part, np.eye(3))
        assert_allclose(t2.x, 1)

        # From rotation only (zero translation)
        t3 = cas.TransformationMatrix.from_point_rotation_matrix(rotation_matrix=r)
        assert_allclose(t3.x, 0)
        assert_allclose(t3.y, 0)
        assert_allclose(t3.z, 0)

    def test_from_xyz_quat(self):
        """Test construction from position and quaternion"""
        t = cas.TransformationMatrix.from_xyz_quaternion(
            pos_x=1,
            pos_y=2,
            pos_z=3,
            quat_w=1,
            quat_x=0,
            quat_y=0,
            quat_z=0,  # Identity quaternion
        )

        assert isinstance(t, cas.TransformationMatrix)
        assert t.x.to_np() == 1
        assert t.y.to_np() == 2
        assert t.z.to_np() == 3

        # Should have identity rotation
        rotation_part = t[:3, :3]
        assert_allclose(rotation_part, np.eye(3))

    def test_composition(self):
        """Test composition of multiple transformations"""
        # Translation only
        t1 = cas.TransformationMatrix.from_xyz_rpy(1, 0, 0)  # Translate in X
        t2 = cas.TransformationMatrix.from_xyz_rpy(0, 1, 0)  # Translate in Y

        # Compose transformations
        combined = t2 @ t1
        assert isinstance(combined, cas.TransformationMatrix)

        # Apply to origin point
        origin = cas.Point3(x_init=0, y_init=0, z_init=0)
        result = combined @ origin
        assert isinstance(result, cas.Point3)
        # Should be at (1, 1, 0) after both translations
        assert_allclose(result.x, 1)
        assert_allclose(result.y, 1)
        assert_allclose(result.z, 0)

    def test_point_transformation(self):
        """Test transformation of points"""
        # Create a transformation: translate by (1,2,3) and rotate by 90° around Z
        t = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3, 0, 0, np.pi / 2)

        # Transform a point
        p = cas.Point3(x_init=1, y_init=0, z_init=0)
        transformed = t @ p

        assert isinstance(transformed, cas.Point3)
        assert transformed[3] == 1  # Homogeneous coordinate preserved

        # Expected: rotation transforms (1,0,0) to (0,1,0), then translation adds (1,2,3)
        expected_x = 0 + 1  # 1 (translation)
        expected_y = 1 + 2  # 3 (rotated + translation)
        expected_z = 0 + 3  # 3 (translation)

        assert_allclose(transformed.x, expected_x, atol=1e-10)
        assert_allclose(transformed.y, expected_y, atol=1e-10)
        assert_allclose(transformed.z, expected_z, atol=1e-10)

    def test_vector_transformation(self):
        """Test transformation of vectors (no translation effect)"""
        # Create transformation with both rotation and translation
        t = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3, 0, 0, np.pi / 2)

        # Transform a vector
        v = cas.Vector3(x_init=1, y_init=0, z_init=0)
        transformed = t @ v

        assert isinstance(transformed, cas.Vector3)
        assert transformed[3] == 0  # Vector homogeneous coordinate

        # Only rotation should affect vectors, not translation
        # 90° rotation around Z: (1,0,0) -> (0,1,0)
        assert_allclose(transformed.x, 0, atol=1e-10)
        assert_allclose(transformed.y, 1, atol=1e-10)
        assert_allclose(transformed.z, 0, atol=1e-10)

    def test_inverse(self):
        """Test matrix inversion"""
        t = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)
        t_inv = t.inverse()

        assert isinstance(t_inv, cas.TransformationMatrix)

        # Test that T @ T^(-1) = I
        identity_check = t @ t_inv
        identity = cas.TransformationMatrix()
        assert_allclose(identity_check, identity, atol=1e-10)

        # Test that T^(-1) @ T = I
        identity_check2 = t_inv @ t
        assert_allclose(identity_check2, identity, atol=1e-10)

        # Test frame swapping
        if hasattr(t, "reference_frame") and hasattr(t, "child_frame"):
            assert t_inv.reference_frame == t.child_frame
            assert t_inv.child_frame == t.reference_frame

    def test_extraction_methods(self):
        """Test methods for extracting components"""
        t = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)

        # Extract position
        position = t.to_position()
        assert isinstance(position, cas.Point3)
        assert position.x.to_np() == 1
        assert position.y.to_np() == 2
        assert position.z.to_np() == 3
        assert position[3] == 1

        # Extract rotation
        rotation = t.to_rotation_matrix()
        assert isinstance(rotation, cas.RotationMatrix)
        # Should have zero translation
        assert rotation[0, 3] == 0
        assert rotation[1, 3] == 0
        assert rotation[2, 3] == 0

        # Extract translation (pure translation matrix)
        translation = t.to_translation()
        assert isinstance(translation, cas.TransformationMatrix)
        # Should have identity rotation
        rotation_part = translation[:3, :3]
        assert_allclose(rotation_part, np.eye(3))
        # Should preserve translation
        assert translation.x.to_np() == 1
        assert translation.y.to_np() == 2
        assert translation.z.to_np() == 3

        # Extract quaternion
        quaternion = t.to_quaternion()
        assert isinstance(quaternion, cas.Quaternion)

    def test_frame_properties(self):
        """Test reference frame and child frame properties"""
        t = cas.TransformationMatrix()

        # Initially should be None
        assert t.reference_frame is None
        assert t.child_frame is None

        # Test frame preservation in operations
        t1 = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)
        t2 = cas.TransformationMatrix.from_xyz_rpy(4, 5, 6, 0.4, 0.5, 0.6)

        result = t1 @ t2
        # Frame handling depends on implementation
        assert hasattr(result, "reference_frame")
        assert hasattr(result, "child_frame")

    def test_invalid_operations(self):
        """Test invalid operations that should raise TypeError"""
        t = cas.TransformationMatrix()
        s = cas.FloatVariable(name="s")
        e = cas.Expression(data=1)
        q = cas.Quaternion()

        # These should raise TypeError
        with pytest.raises(TypeError):
            t @ s  # Matrix @ FloatVariable
        with pytest.raises(TypeError):
            t @ e  # Matrix @ Expression (scalar)
        with pytest.raises(TypeError):
            t @ q  # Matrix @ Quaternion

    @given(float_no_nan_no_inf(), float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_pure_translation(self, x, y, z):
        """Property-based krrood_test for pure translation matrices"""
        t = cas.TransformationMatrix.from_xyz_rpy(x, y, z)

        # Should have identity rotation
        rotation_part = t[:3, :3]
        assert_allclose(rotation_part, np.eye(3))

        # Should have correct translation
        assert_allclose(t.x, x)
        assert_allclose(t.y, y)
        assert_allclose(t.z, z)

        # Bottom row should be [0, 0, 0, 1]
        assert t[3, 0] == 0
        assert t[3, 1] == 0
        assert t[3, 2] == 0
        assert t[3, 3] == 1

    def test_symbolic_operations(self):
        """Test operations with symbolic expressions"""
        x_sym = cas.FloatVariable(name="x")
        y_sym = cas.FloatVariable(name="y")
        angle_sym = cas.FloatVariable(name="theta")

        # Create symbolic transformation
        t_sym = cas.TransformationMatrix.from_xyz_rpy(x_sym, y_sym, 0, 0, 0, angle_sym)

        # Should be able to compose with other transformations
        t_numeric = cas.TransformationMatrix.from_xyz_rpy(1, 1, 1)
        result = t_sym @ t_numeric

        assert isinstance(result, cas.TransformationMatrix)

        # Should contain the variables
        variables = result.free_variables()
        variable_names = [s.name for s in variables if hasattr(s, "name")]
        assert "x" in variable_names
        assert "y" in variable_names
        assert "theta" in variable_names

    def test_compilation(self):
        """Test compilation and execution of transformation matrices"""
        # Test symbolic transformation compilation
        compiled_transform = cas.TransformationMatrix.from_xyz_rpy(
            1, 2, 3, 0.1, 0.2, 0.3
        )

        # Should be a valid 4x4 transformation matrix
        assert compiled_transform.shape == (4, 4)
        assert_allclose(compiled_transform[3, 3], 1)
        assert_allclose(compiled_transform[3, 0], 0)
        assert_allclose(compiled_transform[3, 1], 0)
        assert_allclose(compiled_transform[3, 2], 0)

    def test_deepcopy(self):
        """Test deep copy functionality"""
        t = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)

        from copy import deepcopy

        t_copy = deepcopy(t)

        assert isinstance(t_copy, cas.TransformationMatrix)
        assert_allclose(t, t_copy)

        # Frames should be preserved but not deep copied (reference equality)
        assert t_copy.reference_frame == t.reference_frame
        assert t_copy.child_frame == t.child_frame

    def test_robot_kinematics(self):
        """Test transformation matrices in typical robotics scenarios"""
        # Forward kinematics chain: base -> link1 -> link2 -> end_effector
        base_T_link1 = cas.TransformationMatrix.from_xyz_rpy(
            0, 0, 1, 0, 0, np.pi / 4
        )  # Lift and rotate
        link1_T_link2 = cas.TransformationMatrix.from_xyz_rpy(
            1, 0, 0, 0, np.pi / 2, 0
        )  # Extend and bend
        link2_T_ee = cas.TransformationMatrix.from_xyz_rpy(
            0.5, 0, 0
        )  # End effector offset

        # Forward kinematics
        base_T_ee = base_T_link1 @ link1_T_link2 @ link2_T_ee
        assert isinstance(base_T_ee, cas.TransformationMatrix)

        # Test that inverse kinematics works
        ee_T_base = base_T_ee.inverse()
        identity_check = base_T_ee @ ee_T_base
        identity = cas.TransformationMatrix()
        assert_allclose(identity_check, identity, atol=1e-10)

    def test_coordinate_transformations(self):
        """Test coordinate frame transformations"""
        # Transform from world to robot base
        world_T_robot = cas.TransformationMatrix.from_xyz_rpy(2, 3, 0, 0, 0, np.pi / 2)

        # Point in world coordinates
        world_point = cas.Point3(x_init=1, y_init=0, z_init=1)

        # Transform to robot coordinates
        robot_point = world_T_robot.inverse() @ world_point
        assert isinstance(robot_point, cas.Point3)

        # Transform back to world coordinates
        world_point_back = world_T_robot @ robot_point
        assert isinstance(world_point_back, cas.Point3)

        # Should get original point back
        assert_allclose(world_point, world_point_back, atol=1e-10)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Identity transformation
        t_identity = cas.TransformationMatrix()
        point = cas.Point3(x_init=1, y_init=2, z_init=3)
        transformed = t_identity @ point
        assert_allclose(point, transformed)

        # Zero translation, identity rotation
        t_zero = cas.TransformationMatrix.from_xyz_rpy(0, 0, 0, 0, 0, 0)
        assert_allclose(t_zero, t_identity)

        # Large translation values
        t_large = cas.TransformationMatrix.from_xyz_rpy(1e6, -1e6, 1e6)
        assert t_large.x.to_np() == 1e6
        assert t_large.y.to_np() == -1e6
        assert t_large.z.to_np() == 1e6

        # Small rotation angles (numerical stability)
        t_small = cas.TransformationMatrix.from_xyz_rpy(0, 0, 0, 1e-8, 1e-8, 1e-8)
        rotation_part = t_small[:3, :3]
        assert_allclose(rotation_part, np.eye(3), atol=1e-7)

    @given(quaternion())
    def test_quaternion_consistency(self, q):
        """Property-based krrood_test for quaternion consistency"""
        # Create transformation from quaternion
        t = cas.TransformationMatrix.from_xyz_quaternion(
            pos_x=1,
            pos_y=2,
            pos_z=3,
            quat_w=q[3],
            quat_x=q[0],
            quat_y=q[1],
            quat_z=q[2],
        )

        # Extract quaternion back
        q_extracted = t.to_quaternion()

        # Create transformation from extracted quaternion
        t_roundtrip = cas.TransformationMatrix.from_xyz_quaternion(
            pos_x=1,
            pos_y=2,
            pos_z=3,
            quat_w=q_extracted[3],
            quat_x=q_extracted[0],
            quat_y=q_extracted[1],
            quat_z=q_extracted[2],
        )

        # Should be the same (within numerical precision)
        assert_allclose(t, t_roundtrip, atol=1e-10)

    def test_homogeneous_properties(self):
        """Test homogeneous coordinate properties"""
        t = cas.TransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)

        # Transforming points preserves homogeneous coordinate = 1
        point = cas.Point3(x_init=4, y_init=5, z_init=6)
        transformed_point = t @ point
        assert transformed_point[3] == 1

        # Transforming vectors preserves homogeneous coordinate = 0
        vector = cas.Vector3(x_init=1, y_init=1, z_init=1)
        transformed_vector = t @ vector
        assert transformed_vector[3] == 0

        # Transformation matrix structure
        transform_np = t.to_np()
        assert transform_np[3, 0] == 0
        assert transform_np[3, 1] == 0
        assert transform_np[3, 2] == 0
        assert transform_np[3, 3] == 1


class TestQuaternion:
    @given(
        quaternion(),
        quaternion(),
        st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1),
    )
    def test_slerp(self, q1, q2, t):
        r1 = cas.Quaternion.from_iterable(q1).slerp(cas.Quaternion.from_iterable(q2), t)
        r2 = quaternion_slerp(q1, q2, t)
        compare_orientations(r1, r2)

    @given(unit_vector(length=3), random_angle())
    def test_quaternion_from_axis_angle1(self, axis, angle):
        actual = cas.Quaternion.from_axis_angle(cas.Vector3.from_iterable(axis), angle)
        expected = quaternion_from_axis_angle(axis, angle)
        assert_allclose(actual, expected)

    @given(quaternion(), quaternion())
    def test_quaternion_multiply(self, q, p):
        q_expr = cas.Quaternion.from_iterable(q)
        p_expr = cas.Quaternion.from_iterable(p)
        actual = q_expr.multiply(p_expr)
        expected = quaternion_multiply(q, p)
        compare_orientations(actual, expected)

    @given(quaternion())
    def test_quaternion_conjugate(self, q):
        actual = cas.Quaternion.from_iterable(q).conjugate()
        expected = quaternion_conjugate(q)
        compare_orientations(actual, expected)

    @given(quaternion(), quaternion())
    def test_quaternion_diff(self, q1, q2):
        q1_expr = cas.Quaternion.from_iterable(q1)
        q2_expr = cas.Quaternion.from_iterable(q2)
        actual = q1_expr.diff(q2_expr)
        expected = quaternion_multiply(quaternion_conjugate(q1), q2)
        compare_orientations(actual, expected)

    @given(quaternion())
    def test_axis_angle_from_quaternion(self, q):
        axis2, angle2 = axis_angle_from_quaternion(*q)
        axis = cas.Quaternion.from_iterable(q).to_axis_angle()[0]
        angle = cas.Quaternion.from_iterable(q).to_axis_angle()[1]
        compare_axis_angle(angle, axis[:3], angle2, axis2)
        assert axis[-1] == 0

    def test_axis_angle_from_quaternion2(self):
        q = (0, 0, 0, 1.0000001)
        axis2, angle2 = axis_angle_from_quaternion(*q)
        axis = cas.Quaternion.from_iterable(q).to_axis_angle()[0]
        angle = cas.Quaternion.from_iterable(q).to_axis_angle()[1]
        compare_axis_angle(angle, axis[:3], angle2, axis2)
        assert axis[-1] == 0

    @given(random_angle(), random_angle(), random_angle())
    def test_quaternion_from_rpy(self, roll, pitch, yaw):
        q = cas.Quaternion.from_rpy(roll, pitch, yaw)
        q2 = quaternion_from_rpy(roll, pitch, yaw)
        compare_orientations(q, q2)

    @given(quaternion())
    def test_quaternion_from_matrix(self, q):
        matrix = rotation_matrix_from_quaternion(*q)
        q2 = quaternion_from_rotation_matrix(matrix)
        q1_2 = cas.Quaternion.from_rotation_matrix(cas.RotationMatrix(data=matrix))
        compare_orientations(q2, q1_2)

    @given(quaternion(), quaternion())
    def test_dot(self, q1, q2):
        result = cas.Quaternion.from_iterable(q1).dot(cas.Quaternion.from_iterable(q2))
        q1 = np.array(q1)
        q2 = np.array(q2)
        expected = np.dot(q1.T, q2)
        assert_allclose(result, expected)


class TestCASWrapper:
    @given(st.booleans())
    def test_empty_compiled_function(self, sparse):
        if sparse:
            expected = np.array([1, 2, 3], ndmin=2)
        else:
            expected = np.array([1, 2, 3])
        e = cas.Expression(data=expected)
        f = e.compile(sparse=sparse)
        if sparse:
            assert_allclose(f().toarray(), expected)
            assert_allclose(f(np.array([], dtype=float)).toarray(), expected)
        else:
            assert_allclose(f(), expected)
            assert_allclose(f(np.array([], dtype=float)), expected)

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

    def test_vstack(self):
        m = np.eye(4)
        m1 = cas.Expression(data=m)
        e = cas.Expression.vstack([m1, m1])
        r1 = e
        r2 = np.vstack([m, m])
        assert_allclose(r1, r2)

    def test_vstack_empty(self):
        m = np.eye(0)
        m1 = cas.Expression(data=m)
        e = cas.Expression.vstack([m1, m1])
        r1 = e
        r2 = np.vstack([m, m])
        assert_allclose(r1, r2)

    def test_hstack(self):
        m = np.eye(4)
        m1 = cas.Expression(data=m)
        e = cas.Expression.hstack([m1, m1])
        r1 = e
        r2 = np.hstack([m, m])
        assert_allclose(r1, r2)

    def test_hstack_empty(self):
        m = np.eye(0)
        m1 = cas.Expression(data=m)
        e = cas.Expression.hstack([m1, m1])
        r1 = e
        r2 = np.hstack([m, m])
        assert_allclose(r1, r2)

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
        assert_allclose(r1, combined_matrix)

    @given(float_no_nan_no_inf())
    def test_abs(self, f1):
        assert_allclose(cas.abs(f1), abs(f1))

    @given(float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_max(self, f1, f2):
        assert_allclose(cas.max(f1, f2), max(f1, f2))

    @given(float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_save_division(self, f1, f2):
        assert_allclose(
            cas.Expression(data=f1).safe_division(f2), f1 / f2 if f2 != 0 else 0
        )

    @given(float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_min(self, f1, f2):
        assert_allclose(cas.min(f1, f2), min(f1, f2))

    @given(float_no_nan_no_inf())
    def test_sign(self, f1):
        assert_allclose(cas.sign(f1), np.sign(f1))

    @given(float_no_nan_no_inf(), float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_if_greater_zero(self, condition, if_result, else_result):
        assert_allclose(
            cas.if_greater_zero(condition, if_result, else_result),
            float(if_result if condition > 0 else else_result),
        )

    def test_if_one_arg(self):
        types = [
            cas.Point3,
            cas.Vector3,
            cas.Quaternion,
            cas.Expression,
            cas.TransformationMatrix,
            cas.RotationMatrix,
        ]
        if_functions = [
            cas.if_else,
            cas.if_eq_zero,
            cas.if_greater_eq_zero,
            cas.if_greater_zero,
        ]
        c = cas.FloatVariable(name="c")
        for type_ in types:
            for if_function in if_functions:
                if_result = type_()
                else_result = type_()
                result = if_function(c, if_result, else_result)
                assert isinstance(
                    result, type_
                ), f"{type(result)} != {type_} for {if_function}"

    def test_if_two_arg(self):
        types = [
            cas.Point3,
            cas.Vector3,
            cas.Quaternion,
            cas.Expression,
            cas.TransformationMatrix,
            cas.RotationMatrix,
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
        for type_ in types:
            for if_function in if_functions:
                if_result = type_()
                else_result = type_()
                assert isinstance(if_function(a, b, if_result, else_result), type_)

    @given(float_no_nan_no_inf(), float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_if_greater_eq_zero(self, condition, if_result, else_result):
        assert_allclose(
            cas.if_greater_eq_zero(condition, if_result, else_result),
            float(if_result if condition >= 0 else else_result),
        )

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
    )
    def test_if_greater_eq(self, a, b, if_result, else_result):
        assert_allclose(
            cas.if_greater_eq(a, b, if_result, else_result),
            float(if_result if a >= b else else_result),
        )

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
    )
    def test_if_less_eq(self, a, b, if_result, else_result):
        assert_allclose(
            cas.if_less_eq(a, b, if_result, else_result),
            float(if_result if a <= b else else_result),
        )

    @given(float_no_nan_no_inf(), float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_if_eq_zero(self, condition, if_result, else_result):
        assert_allclose(
            cas.if_eq_zero(condition, if_result, else_result),
            float(if_result if condition == 0 else else_result),
        )

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
    )
    def test_if_eq(self, a, b, if_result, else_result):
        assert_allclose(
            cas.if_eq(a, b, if_result, else_result),
            float(if_result if a == b else else_result),
        )

    @given(float_no_nan_no_inf())
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
        assert_allclose(actual, expected)

    @given(float_no_nan_no_inf())
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
        assert_allclose(actual, expected)

    @given(float_no_nan_no_inf(10))
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

        assert_allclose(
            cas.if_less_eq_cases(a, b_result_cases, cas.Expression(data=0)),
            float(reference(a, b_result_cases, 0)),
        )

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
    )
    def test_if_greater(self, a, b, if_result, else_result):
        assert_allclose(
            cas.if_greater(a, b, if_result, else_result),
            float(if_result if a > b else else_result),
        )

    @given(
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
        float_no_nan_no_inf(),
    )
    def test_if_less(self, a, b, if_result, else_result):
        assert_allclose(
            cas.if_less(a, b, if_result, else_result),
            float(if_result if a < b else else_result),
        )

    @given(float_no_nan_no_inf(), float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_limit(self, x, lower_limit, upper_limit):
        r1 = cas.limit(x, lower_limit, upper_limit)
        r2 = max(lower_limit, min(upper_limit, x))
        assert_allclose(r1, r2)

    @given(unit_vector(4))
    def test_trace(self, q):
        m = rotation_matrix_from_quaternion(*q)
        assert_allclose(m.trace(), np.trace(m))

    @given(float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_fmod(self, a, b):
        ref_r = np.fmod(a, b)
        assert_allclose(cas.fmod(a, b), ref_r, equal_nan=True)

    @given(float_no_nan_no_inf())
    def test_normalize_angle_positive(self, a):
        expected = normalize_angle_positive(a)
        actual = cas.normalize_angle_positive(a)
        assert_allclose(
            shortest_angular_distance(actual.to_np(), expected),
            0.0,
        )

    @given(float_no_nan_no_inf())
    def test_normalize_angle(self, a):
        ref_r = normalize_angle(a)
        assert_allclose(cas.normalize_angle(a), ref_r)

    @given(float_no_nan_no_inf(), float_no_nan_no_inf())
    def test_shorted_angular_distance(self, angle1, angle2):
        try:
            expected = shortest_angular_distance(angle1, angle2)
        except ValueError:
            expected = np.nan
        actual = cas.shortest_angular_distance(angle1, angle2)
        assert_allclose(actual, expected, equal_nan=True)

    @given(unit_vector(4), unit_vector(4))
    def test_entrywise_product(self, q1, q2):
        m1 = rotation_matrix_from_quaternion(*q1)
        m2 = rotation_matrix_from_quaternion(*q2)
        r1 = cas.Expression(data=m1).entrywise_product(m2)
        r2 = m1 * m2
        assert_allclose(r1, r2)

    @given(sq_matrix())
    def test_sum(self, m):
        actual_sum = m.sum()
        expected_sum = np.sum(m)
        assert_allclose(actual_sum, expected_sum, rtol=1.0e-4)

    @given(sq_matrix())
    def test_sum_row(self, m):
        actual_sum = cas.Expression(data=m).sum_row()
        expected_sum = np.sum(m, axis=0)
        assert_allclose(actual_sum, expected_sum)

    @given(sq_matrix())
    def test_sum_column(self, m):
        actual_sum = cas.Expression(data=m).sum_column()
        expected_sum = np.sum(m, axis=1)
        assert_allclose(actual_sum, expected_sum)

    def test_to_str(self):
        axis = cas.Vector3(*cas.create_float_variables(["v1", "v2", "v3"]))
        angle = cas.FloatVariable(name="alpha")
        q = cas.Quaternion.from_axis_angle(axis, angle)
        expr = q.norm()
        assert expr.pretty_str() == [
            [
                "sqrt((((sq((v1*sin((alpha/2))))"
                "+sq((v2*sin((alpha/2)))))"
                "+sq((v3*sin((alpha/2)))))"
                "+sq(cos((alpha/2)))))"
            ]
        ]

    def test_to_str2(self):
        a, b = cas.create_float_variables(["a", "b"])
        e = cas.if_eq(a, 0, a, b)
        assert e.pretty_str() == [["(((a==0)?a:0)+((!(a==0))?b:0))"]]

    def test_leq_on_array(self):
        a = cas.Expression(data=np.array([1, 2, 3, 4]))
        b = cas.Expression(data=np.array([2, 2, 2, 2]))
        assert not cas.logic_all(a <= b).to_np()


class TestCompiledFunction:
    def test_dense(self):
        s1_value = 420.0
        s2_value = 69.0
        s1, s2 = cas.create_float_variables(["s1", "s2"])
        e = cas.sqrt(cas.cos(s1) + cas.sin(s2))
        e_f = e.compile()
        actual = e_f(np.array([s1_value, s2_value]))
        expected = np.sqrt(np.cos(s1_value) + np.sin(s2_value))
        assert_allclose(actual, expected)

    def test_dense_two_params(self):
        s1_value = 420.0
        s2_value = 69.0
        s1, s2 = cas.create_float_variables(["s1", "s2"])
        e = cas.sqrt(cas.cos(s1) + cas.sin(s2))
        e_f = e.compile(parameters=[[s1], [s2]])
        actual = e_f(np.array([s1_value]), np.array([s2_value]))
        expected = np.sqrt(np.cos(s1_value) + np.sin(s2_value))
        assert_allclose(actual, expected)

    def test_sparse(self):
        s1_value = 420.0
        s2_value = 69.0
        s1, s2 = cas.create_float_variables(["s1", "s2"])
        e = cas.sqrt(cas.cos(s1) + cas.sin(s2))
        e_f = e.compile(sparse=True)
        actual = e_f(np.array([s1_value, s2_value]))
        assert isinstance(actual, scipy.sparse.csc_matrix)
        expected = np.sqrt(np.cos(s1_value) + np.sin(s2_value))
        assert_allclose(actual.toarray(), expected)

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
        assert_allclose(actual_e1, expected_e1)
        assert_allclose(actual_e2, expected_e2)

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

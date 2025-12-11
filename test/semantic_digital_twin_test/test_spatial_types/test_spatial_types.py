import numpy as np
import pytest

import semantic_digital_twin.spatial_types.spatial_types as cas
from krrood.symbolic_math.exceptions import UnsupportedOperationError
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    WrongDimensionsError,
    SpatialTypesError,
)
from semantic_digital_twin.world_description.world_entity import Body
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
)
from .utils_for_tests import (
    compare_axis_angle,
    compare_orientations,
)

bool_values = [True, False]
numbers = [-69, 23]
unit_vectors3 = [np.array([1.0, 0, 0]), np.array([0.0, 1, 0])]
points = [np.array([1.0, 2.3, 0.00001]), np.array([0.0, -1, 0])]
quaternions = [np.array([1.0, 0, 0, 0]), np.array([0.0, 1, 0, 0])]


class TestRotationMatrix:
    @pytest.mark.parametrize("q1", quaternions)
    @pytest.mark.parametrize("q2", quaternions)
    def test_rotation_distance(self, q1, q2):
        m1 = rotation_matrix_from_quaternion(*q1)
        m2 = rotation_matrix_from_quaternion(*q2)
        cas.RotationMatrix()
        actual_angle = cas.RotationMatrix(data=m1).rotational_error(
            cas.RotationMatrix(data=m2)
        )
        _, expected_angle = axis_angle_from_rotation_matrix(m1.T.dot(m2))
        expected_angle = expected_angle
        try:
            assert np.allclose(
                shortest_angular_distance(actual_angle.to_np()[0], expected_angle),
                0,
            )
        except AssertionError:
            assert np.allclose(
                shortest_angular_distance(actual_angle.to_np()[0], -expected_angle),
                0,
            )

    def test_reference_frames(self):
        reference_frame = Body(name=PrefixedName("muh"))
        reference_frame2 = Body(name=PrefixedName("muh2"))
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
        rotation_matrix = cas.RotationMatrix()
        works = [
            cas.Vector3(x_init=1, y_init=1, z_init=1),
            rotation_matrix,
            cas.HomogeneousTransformationMatrix(),
            cas.Pose(),
        ]
        does_not_work = [
            cas.FloatVariable(name="s"),
            cas.Expression(data=1),
            cas.Point3(x_init=1, y_init=1, z_init=1),
            cas.Quaternion(),
        ]

        for other in works:
            assert isinstance(rotation_matrix @ other, type(other))
        for other in does_not_work:
            with pytest.raises(UnsupportedOperationError):
                # noinspection PyTypeChecker
                rotation_matrix @ other

    def test_x_y_z_vector(self):
        v = np.array([1, 1, 1])
        v = v / np.linalg.norm(v)
        R_ref = rotation_matrix_from_axis_angle(v, 1)
        R = cas.RotationMatrix().from_axis_angle(cas.Vector3.unit_vector(1, 1, 1), 1)
        assert np.allclose(R.x_vector(), R_ref[:, 0])
        assert np.allclose(R.y_vector(), R_ref[:, 1])
        assert np.allclose(R.z_vector(), R_ref[:, 2])

    def test_create_RotationMatrix(self):
        s = cas.FloatVariable(name="s")
        r = cas.RotationMatrix.from_rpy(1, 2, s)
        r = cas.RotationMatrix.from_rpy(1, 2, 3)
        assert isinstance(r, cas.RotationMatrix)
        t = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3)
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
        assert np.allclose(cas.RotationMatrix.from_vectors(x=x_unit, y=y_unit), R_ref)
        assert np.allclose(cas.RotationMatrix.from_vectors(x=x_unit, z=z_unit), R_ref)
        assert np.allclose(cas.RotationMatrix.from_vectors(y=y_unit, z=z_unit), R_ref)
        assert np.allclose(
            cas.RotationMatrix.from_vectors(x=x_unit, y=y_unit, z=z_unit), R_ref
        )

    @pytest.mark.parametrize("q", quaternions)
    def test_from_quaternion(self, q):
        actual = cas.RotationMatrix.from_quaternion(cas.Quaternion.from_iterable(q))
        expected = rotation_matrix_from_quaternion(*q)
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("roll", numbers)
    @pytest.mark.parametrize("pitch", numbers)
    @pytest.mark.parametrize("yaw", numbers)
    def test_from_rpy(self, roll, pitch, yaw):
        m1 = cas.RotationMatrix.from_rpy(roll, pitch, yaw)
        m2 = rotation_matrix_from_rpy(roll, pitch, yaw)
        assert np.allclose(m1, m2)

    @pytest.mark.parametrize("axis", [np.array([1, 0, 0])])
    @pytest.mark.parametrize("angle", numbers)
    def test_rotation3_axis_angle(self, axis, angle):
        assert np.allclose(
            cas.RotationMatrix.from_axis_angle(axis, angle),
            rotation_matrix_from_axis_angle(np.array(axis), angle),
        )

    @pytest.mark.parametrize("q", quaternions)
    def test_axis_angle_from_matrix(self, q):
        m = rotation_matrix_from_quaternion(*q)

        actual_axis = cas.RotationMatrix(data=m).to_axis_angle()[0]
        actual_angle = cas.RotationMatrix(data=m).to_axis_angle()[1]

        expected_axis, expected_angle = axis_angle_from_rotation_matrix(m)
        compare_axis_angle(actual_angle, actual_axis[:3], expected_angle, expected_axis)

        assert actual_axis[-1].to_np() == 0

    @pytest.mark.parametrize("expected_axis", [np.array([1, 0, 0])])
    @pytest.mark.parametrize("expected_angle", numbers)
    def test_axis_angle_from_matrix2(self, expected_axis, expected_angle):
        m = rotation_matrix_from_axis_angle(expected_axis, expected_angle)
        actual_axis = cas.RotationMatrix(data=m).to_axis_angle()[0]
        actual_angle = cas.RotationMatrix(data=m).to_axis_angle()[1]
        compare_axis_angle(actual_angle, actual_axis[:3], expected_angle, expected_axis)
        assert actual_axis[-1] == 0

    @pytest.mark.parametrize("q", quaternions)
    def test_rpy_from_matrix(self, q):
        expected = rotation_matrix_from_quaternion(*q)

        roll = float(cas.RotationMatrix(data=expected).to_rpy()[0].to_np()[0])
        pitch = float(cas.RotationMatrix(data=expected).to_rpy()[1].to_np()[0])
        yaw = float(cas.RotationMatrix(data=expected).to_rpy()[2].to_np()[0])
        actual = rotation_matrix_from_rpy(roll, pitch, yaw)

        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("q", quaternions)
    def test_rpy_from_matrix2(self, q):
        matrix = rotation_matrix_from_quaternion(*q)
        roll = cas.RotationMatrix(data=matrix).to_rpy()[0]
        pitch = cas.RotationMatrix(data=matrix).to_rpy()[1]
        yaw = cas.RotationMatrix(data=matrix).to_rpy()[2]
        r1 = cas.RotationMatrix.from_rpy(roll, pitch, yaw)
        assert np.allclose(r1, matrix, atol=1.0e-4)

    def test_initialization(self):
        """Test various ways to initialize RotationMatrix"""
        # Default initialization (identity)
        r_identity = cas.RotationMatrix()
        assert isinstance(r_identity, cas.RotationMatrix)
        identity_np = r_identity
        expected_identity = np.eye(4)
        assert np.allclose(identity_np, expected_identity)

        # From another RotationMatrix
        r_copy = cas.RotationMatrix(data=r_identity)
        assert isinstance(r_copy, cas.RotationMatrix)
        assert np.allclose(r_copy, identity_np)

        # From numpy array
        rotation_data = np.eye(4)
        rotation_data[0, 1] = 0.5  # Add some rotation
        r_from_np = cas.RotationMatrix(data=rotation_data)
        assert isinstance(r_from_np, cas.RotationMatrix)

        # From TransformationMatrix
        t = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)
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
        assert np.allclose(should_be_identity, np.eye(4), atol=1e-10)

        # Test that determinant is 1 (proper rotation, not reflection)
        det = r.det()
        assert np.allclose(det, 1.0, atol=1e-10)

    def test_transpose(self):
        """Test transpose operation and its properties"""
        r = cas.RotationMatrix.from_rpy(0.1, 0.2, 0.3)
        r_t = r.T

        assert isinstance(r_t, cas.RotationMatrix)

        # For rotation matrices: R.T = R^(-1)
        product = r @ r_t
        identity = cas.RotationMatrix()
        assert np.allclose(product, identity, atol=1e-10)

        # Double transpose should give original
        r_tt = r_t.T
        assert np.allclose(r, r_tt)

    def test_inverse(self):
        """Test matrix inversion for rotation matrices"""
        r = cas.RotationMatrix.from_rpy(0.5, -0.3, 1.2)

        # For rotation matrices, inverse should equal transpose
        r_inv = r.inverse()
        r_t = r.T
        assert isinstance(r_inv, cas.RotationMatrix)
        assert np.allclose(r_inv, r_t, atol=1e-10)

        # R @ R^(-1) = I
        identity_check = r @ r_inv
        identity = cas.RotationMatrix()
        assert np.allclose(identity_check, identity, atol=1e-10)

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
        assert np.allclose(combined.det(), 1.0)
        assert np.allclose(combined @ combined.T, np.eye(4), atol=1e-10)

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
        assert np.allclose(rotated, expected, atol=1e-10)

        # Test with regular Vector3
        v = cas.Vector3(x_init=1, y_init=0, z_init=0)
        rotated_v = r_z90 @ v
        assert isinstance(rotated_v, cas.Vector3)
        assert np.allclose(rotated_v[:3], np.array([0, 1, 0]), atol=1e-10)

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
        assert np.allclose(roll, 0.1, atol=1e-10)
        assert np.allclose(pitch, 0.2, atol=1e-10)
        assert np.allclose(yaw, 0.3, atol=1e-10)

        # Test conversion to quaternion
        q = r.to_quaternion()
        assert isinstance(q, cas.Quaternion)

        # Round-trip krrood_test: R -> Q -> R should preserve rotation
        r_roundtrip = cas.RotationMatrix.from_quaternion(q)
        assert np.allclose(r, r_roundtrip, atol=1e-10)

    @pytest.mark.parametrize("roll", [np.pi / 2, 0, -np.pi / 23])
    @pytest.mark.parametrize("pitch", [np.pi / 3, 0, -np.pi / 23])
    @pytest.mark.parametrize("yaw", [np.pi / 2, 0, -np.pi / 23])
    def test_rpy_roundtrip(self, roll, pitch, yaw):
        r = cas.RotationMatrix.from_rpy(roll, pitch, yaw)
        r_roll, r_pitch, r_yaw = r.to_rpy()

        assert np.allclose(r_roll, roll, atol=1e-10)
        assert np.allclose(r_pitch, pitch, atol=1e-10)
        assert np.allclose(r_yaw, yaw, atol=1e-10)

    @pytest.mark.parametrize("axis", unit_vectors3)
    @pytest.mark.parametrize("angle", numbers)
    def test_axis_angle_properties(self, axis, angle):

        axis_unit = cas.Vector3.from_iterable(axis)
        r = cas.RotationMatrix.from_axis_angle(axis_unit, angle)

        # Test that axis is preserved (rotation around axis shouldn't change axis)
        rotated_axis = r @ axis_unit
        # For rotation around axis, the axis should remain unchanged
        dot_product = axis_unit @ rotated_axis
        assert np.allclose(dot_product, 1.0, atol=1e-10)

    def test_small_angle_approximation(self):
        """Test behavior with very small rotation angles"""
        small_angle = 1e-8

        # Small rotation around Z-axis
        r = cas.RotationMatrix.from_axis_angle(cas.Vector3.Z(), small_angle)
        rotation_part = r[:3, :3]

        # Should be close to identity for very small angles
        assert np.allclose(rotation_part, np.eye(3), atol=1e-7)

        # But determinant should still be 1
        assert np.allclose(rotation_part.det(), 1.0, atol=1e-12)

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
        assert np.allclose(compiled_rotation.det(), 1.0)
        assert np.allclose(
            compiled_rotation @ compiled_rotation.T, np.eye(4), atol=1e-10
        )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Zero rotation
        r_zero = cas.RotationMatrix.from_axis_angle(cas.Vector3.X(), 0)
        identity = cas.RotationMatrix()
        assert np.allclose(r_zero, identity, atol=1e-12)

        # Full rotation (2π)
        r_full = cas.RotationMatrix.from_axis_angle(cas.Vector3.Y(), 2 * np.pi)
        assert np.allclose(r_full, identity, atol=1e-10)

        # π rotation (180 degrees)
        r_pi = cas.RotationMatrix.from_axis_angle(cas.Vector3.Z(), np.pi)
        rotation_part = r_pi[:3, :3]
        # Should flip X and Y axes
        expected_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        assert np.allclose(rotation_part, expected_rotation, atol=1e-10)

    def test_quaternion_consistency(self):
        """Test consistency between quaternion and rotation matrix representations"""
        # Create rotation via different methods
        r_rpy = cas.RotationMatrix.from_rpy(0.1, 0.2, 0.3)
        q = r_rpy.to_quaternion()
        r_from_q = cas.RotationMatrix.from_quaternion(q)

        # Should be identical
        assert np.allclose(r_rpy, r_from_q, atol=1e-12)

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
            assert np.allclose(
                det, 1.0, atol=1e-10
            ), f"Determinant {det} != 1.0 for operation"


class TestPoint3:
    def test_distance_point_to_line_segment1(self):
        p = cas.Point3(x_init=0, y_init=0, z_init=0)
        start = cas.Point3(x_init=0, y_init=0, z_init=-1)
        end = cas.Point3(x_init=0, y_init=0, z_init=1)
        distance = p.distance_to_line_segment(start, end)[0]
        nearest = p.distance_to_line_segment(start, end)[1]
        assert np.allclose(distance, 0)
        assert np.allclose(nearest, p)

    def test_distance_point_to_line_segment2(self):
        p = cas.Point3(x_init=0, y_init=1, z_init=0.5)
        start = cas.Point3(x_init=0, y_init=0, z_init=0)
        end = cas.Point3(x_init=0, y_init=0, z_init=1)
        distance = p.distance_to_line_segment(start, end)[0]
        nearest = p.distance_to_line_segment(start, end)[1]
        assert np.allclose(distance, 1)
        assert np.allclose(nearest, cas.Point3(x_init=0, y_init=0, z_init=0.5))

    def test_distance_point_to_line_segment3(self):
        p = cas.Point3(x_init=0, y_init=1, z_init=2)
        start = cas.Point3(x_init=0, y_init=0, z_init=0)
        end = cas.Point3(x_init=0, y_init=0, z_init=1)
        distance = p.distance_to_line_segment(start, end)[0]
        nearest = p.distance_to_line_segment(start, end)[1]
        assert np.allclose(distance, 1.4142135623730951)
        assert np.allclose(nearest, cas.Point3(x_init=0, y_init=0, z_init=1.0))

    @pytest.mark.parametrize("v", [np.array([1, 2, 3.0]), [0, 0, 0.0], (-1, 0, 0.0)])
    def test_norm(self, v):
        p = cas.Point3.from_iterable(v)
        actual = p.norm()
        expected = np.linalg.norm(v)
        assert np.allclose(actual, expected)

    def test_init(self):
        l = [1, 2, 3]
        s = cas.FloatVariable(name="s")
        e = cas.Expression(data=1)
        v = cas.Vector3(x_init=1, y_init=1, z_init=1)
        p = cas.Point3(x_init=l[0], y_init=l[1], z_init=l[2])
        r = cas.RotationMatrix()
        q = cas.Quaternion()
        t = cas.HomogeneousTransformationMatrix()

        cas.Point3()
        cas.Point3(x_init=s, y_init=e, z_init=0)
        assert np.allclose(p[3], 1)
        assert np.allclose(p[:3], l)

        cas.Point3.from_iterable(cas.Expression(data=v))
        cas.Point3.from_iterable(l)
        with pytest.raises(WrongDimensionsError):
            cas.Point3.from_iterable(r.to_np())
        with pytest.raises(WrongDimensionsError):
            cas.Point3.from_iterable(t.to_np())
        with pytest.raises(WrongDimensionsError):
            cas.Point3.from_iterable(t.to_np())

    @pytest.mark.parametrize("condition", bool_values)
    def test_if_greater_zero(self, condition):
        if_result, else_result = cas.Point3(1, 2, 3), cas.Point3(4, 5, 6)
        actual = cas.if_greater_zero(condition, if_result, else_result)
        expected = if_result if condition > 0 else else_result
        assert np.allclose(actual, expected)

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
        result2 = p2 - p1
        assert isinstance(result2, cas.Vector3)
        assert result2.x == 3 and result2.y == 3 and result2.z == 3
        assert result2[3] == 0  # Vector has 0 in homogeneous coordinate
        assert result2.reference_frame == p2.reference_frame

        # Test Point - Vector = Point (translate point by negative vector)
        result3 = p2 - v
        assert isinstance(result3, cas.Point3)
        assert result3.x == 3 and result3.y == 4 and result3.z == 5
        assert result3[3] == 1
        assert result3.reference_frame == p2.reference_frame

        # Test -Point = Point (negate all coordinates)
        result4 = -p1
        assert isinstance(result4, cas.Point3)
        assert result4.x == -1 and result4.y == -2 and result4.z == -3
        assert result4[3] == 1
        assert result4.reference_frame == p1.reference_frame

        # Test Point.norm() = scalar (distance from origin)
        result5 = p1.norm()
        assert isinstance(result5, cas.Expression)
        expected_norm = np.sqrt(1**2 + 2**2 + 3**2)
        assert np.allclose(result5, expected_norm)

        # Test operations with symbolic expressions
        x = cas.FloatVariable(name="x")
        p_symbolic = cas.Point3(x, y_init=2, z_init=3)
        result6 = p_symbolic + v
        assert isinstance(result6, cas.Point3)
        assert result6[3] == 1

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
        assert np.allclose(distance, 5.0)  # 3-4-5 triangle

        # Midpoint calculation
        midpoint = p1 + (p2 - p1) * 0.5
        assert isinstance(midpoint, cas.Point3)
        assert np.allclose(midpoint.x, 1.5)
        assert np.allclose(midpoint.y, 2.0)
        assert np.allclose(midpoint.z, 0.0)

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

    def test_invalid_add(self):
        point = cas.Point3()
        does_not_work = [
            1.0,
            point,
            cas.RotationMatrix(),
            cas.Quaternion(),
            cas.HomogeneousTransformationMatrix(),
            cas.FloatVariable(name="s"),
            cas.Pose(),
        ]
        for other in does_not_work:
            with pytest.raises(TypeError):
                # noinspection PyTypeChecker
                point + other

    def test_invalid_sub(self):
        point = cas.Point3()
        does_not_work = [
            1.0,
            cas.RotationMatrix(),
            cas.Quaternion(),
            cas.HomogeneousTransformationMatrix(),
            cas.FloatVariable(name="s"),
            cas.Pose(),
        ]
        for other in does_not_work:
            with pytest.raises(TypeError):
                # noinspection PyTypeChecker
                point - other

    def test_invalid_multiplication(self):
        point = cas.Point3()
        does_not_work = [
            1.0,
            cas.RotationMatrix(),
            cas.Quaternion(),
            cas.HomogeneousTransformationMatrix(),
            cas.FloatVariable(name="s"),
            cas.Pose(),
        ]
        for other in does_not_work:
            with pytest.raises(TypeError):
                # noinspection PyTypeChecker
                point * other

    def test_invalid_division(self):
        point = cas.Point3()
        does_not_work = [
            1.0,
            cas.RotationMatrix(),
            cas.Quaternion(),
            cas.HomogeneousTransformationMatrix(),
            cas.FloatVariable(name="s"),
            cas.Pose(),
        ]
        for other in does_not_work:
            with pytest.raises(TypeError):
                # noinspection PyTypeChecker
                point / other

    def test_invalid_mat_mul(self):
        point = cas.Point3()
        does_not_work = [
            1.0,
            cas.RotationMatrix(),
            cas.Quaternion(),
            cas.HomogeneousTransformationMatrix(),
            cas.FloatVariable(name="s"),
            cas.Pose(),
        ]
        for other in does_not_work:
            with pytest.raises(TypeError):
                # noinspection PyUnresolvedReferences
                point @ other

    @pytest.mark.parametrize("p1_data", points)
    @pytest.mark.parametrize("p2_data", points)
    def test_distance_property_based(self, p1_data, p2_data):
        p1 = cas.Point3.from_iterable(p1_data)
        p2 = cas.Point3.from_iterable(p2_data)

        # Distance via subtraction and norm
        displacement = p2 - p1
        actual = displacement.norm()

        # Compare with numpy calculation
        expected = np.linalg.norm(np.array(p2_data) - np.array(p1_data))

        assert np.allclose(actual, expected)

    def test_transformation_operations(self):
        """Test transformation matrix operations with points"""
        p = cas.Point3(x_init=1, y_init=2, z_init=3)
        t = cas.HomogeneousTransformationMatrix()

        # Test matrix @ point = point (homogeneous transformation)
        result = t @ p
        assert isinstance(result, cas.Point3)
        assert result[3] == 1  # Homogeneous coordinate preserved

    def test_project_to_line(self):
        point = cas.Point3(x_init=1, y_init=2, z_init=3)
        line_point = cas.Point3(x_init=0, y_init=0, z_init=0)
        line_direction = cas.Vector3(
            x_init=1, y_init=2, z_init=3
        )  # Point lies on this line
        point, distance = point.project_to_line(line_point, line_direction)
        assert np.allclose(distance, 0.0)
        assert np.allclose(point, np.array([1, 2, 3, 1]))

        point = cas.Point3(x_init=1, y_init=0, z_init=0)
        line_point = cas.Point3(x_init=0, y_init=0, z_init=0)
        line_direction = cas.Vector3(x_init=0, y_init=1, z_init=0)  # Y-axis
        point, distance = point.project_to_line(line_point, line_direction)
        assert np.allclose(distance, 1.0)
        assert np.allclose(point, np.array([0, 0, 0, 1]))

        point = cas.Point3(x_init=0, y_init=0, z_init=5)
        line_point = cas.Point3(x_init=0, y_init=0, z_init=0)
        line_direction = cas.Vector3(x_init=1, y_init=0, z_init=0)  # X-axis
        point, distance = point.project_to_line(line_point, line_direction)
        assert np.allclose(distance, 5.0)
        assert np.allclose(point, np.array([0, 0, 0, 1]))

    def test_distance_to_line_segment(self):
        pass

    def test_project_to_plane(self):
        p = cas.Point3(x_init=0, y_init=0, z_init=1)
        actual, distance = p.project_to_plane(
            frame_V_plane_vector1=cas.Vector3.X(),
            frame_V_plane_vector2=cas.Vector3.Y(),
        )
        expected = cas.Point3(x_init=0, y_init=0, z_init=0)
        assert np.allclose(actual, expected)
        assert np.allclose(distance, 1)

    def test_compilation_and_execution(self):
        """Test that Point3 operations compile and execute correctly"""
        # Test point arithmetic compilation
        compiled_add = cas.Point3(x_init=1, y_init=2, z_init=3) + cas.Vector3(
            x_init=1, y_init=1, z_init=1
        )
        expected = np.array([2, 3, 4, 1])
        assert np.allclose(compiled_add, expected)

        # Test point subtraction compilation
        compiled_sub = cas.Point3(x_init=5, y_init=6, z_init=7) - cas.Point3(
            x_init=1, y_init=2, z_init=3
        )
        expected_vector = np.array([4, 4, 4, 0])  # Result is a Vector3
        assert np.allclose(compiled_sub, expected_vector)

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
    @pytest.mark.parametrize("u", unit_vectors3)
    @pytest.mark.parametrize("v", unit_vectors3)
    def test_cross(self, u, v):
        assert np.allclose(
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
        t = cas.HomogeneousTransformationMatrix()

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
            cas.Vector3.from_iterable(r.to_np())
        with pytest.raises(WrongDimensionsError):
            cas.Vector3.from_iterable(t.to_np())
        with pytest.raises(WrongDimensionsError):
            cas.Vector3.from_iterable(t.to_np())

    @pytest.mark.parametrize("v", points)
    def test_is_length_1(self, v):
        unit_v = cas.Vector3.unit_vector(*v)
        assert np.allclose(unit_v.norm(), 1)

    def test_length_0(self):
        unit_v = cas.Vector3.unit_vector(x=0, y=0, z=0)
        assert np.isnan(unit_v.norm().to_np())

    @pytest.mark.parametrize("v", points)
    def test_norm(self, v):
        expected = np.linalg.norm(v)
        v = cas.Vector3.from_iterable(v)
        actual = v.norm()
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("nominator", unit_vectors3)
    @pytest.mark.parametrize("denominator", numbers)
    @pytest.mark.parametrize("if_nan", unit_vectors3)
    def test_save_division(self, nominator, denominator, if_nan):
        nominator_expr = cas.Vector3.from_iterable(nominator)
        denominator_expr = cas.Expression(data=denominator)
        if_nan_expr = cas.Vector3.from_iterable(if_nan)
        result = nominator_expr.safe_division(denominator_expr, if_nan_expr)
        if denominator == 0:
            assert np.allclose(result[:3], if_nan)
        else:
            assert np.allclose(result[:3], np.array(nominator) / denominator)

    @pytest.mark.parametrize("u", points)
    @pytest.mark.parametrize("v", points)
    def test_dot(self, u, v):
        result = cas.Vector3.from_iterable(u) @ cas.Vector3.from_iterable(v)
        expected = np.dot(u, v.T)
        assert np.allclose(result, expected)

    def test_cross_product(self):
        """Test cross product operations"""
        v1 = cas.Vector3(x_init=1, y_init=0, z_init=0)
        v2 = cas.Vector3(x_init=0, y_init=1, z_init=0)

        result = v1.cross(v2)
        assert isinstance(result, cas.Vector3)
        assert result[3] == 0
        # Cross product of x and y unit vectors should be z unit vector
        assert np.allclose(result[:3], np.array([0, 0, 1]))

        # Cross product is anti-commutative
        result2 = v2.cross(v1)
        assert np.allclose(result2[:3], np.array([0, 0, -1]))

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
        assert np.allclose(angle, np.pi / 2)

    def test_scale_method(self):
        """Test the scale method with safe and unsafe modes"""
        v = cas.Vector3(x_init=3, y_init=4, z_init=0)  # Length = 5

        # Safe scaling (default)
        v_copy = cas.Vector3(x_init=3, y_init=4, z_init=0)
        v_copy.scale(10)
        expected_norm = v_copy.norm()
        assert np.allclose(expected_norm, 10)

        # Unsafe scaling
        v_copy2 = cas.Vector3(x_init=3, y_init=4, z_init=0)
        v_copy2.scale(10, unsafe=True)
        expected_norm2 = v_copy2.norm()
        assert np.allclose(expected_norm2, 10)

    def test_invalid_add(self):
        vector = cas.Vector3()
        does_not_work = [
            1.0,
            cas.Point3(),
            cas.RotationMatrix(),
            cas.Quaternion(),
            cas.HomogeneousTransformationMatrix(),
            cas.FloatVariable(name="s"),
            cas.Pose(),
        ]
        for other in does_not_work:
            with pytest.raises(TypeError):
                # noinspection PyTypeChecker
                vector + other

    def test_invalid_sub(self):
        vector = cas.Vector3()
        does_not_work = [
            1.0,
            cas.Point3(),
            cas.RotationMatrix(),
            cas.Quaternion(),
            cas.HomogeneousTransformationMatrix(),
            cas.FloatVariable(name="s"),
            cas.Pose(),
        ]
        for other in does_not_work:
            with pytest.raises(TypeError):
                # noinspection PyTypeChecker
                vector - other

    def test_invalid_multiplication(self):
        vector = cas.Vector3()
        does_not_work = [
            vector,
            cas.Point3(),
            cas.RotationMatrix(),
            cas.Quaternion(),
            cas.HomogeneousTransformationMatrix(),
            cas.Pose(),
        ]
        for other in does_not_work:
            with pytest.raises(TypeError):
                # noinspection PyTypeChecker
                vector * other

    def test_invalid_division(self):
        vector = cas.Vector3()
        does_not_work = [
            vector,
            cas.Point3(),
            cas.RotationMatrix(),
            cas.Quaternion(),
            cas.HomogeneousTransformationMatrix(),
            cas.Pose(),
        ]
        for other in does_not_work:
            with pytest.raises(TypeError):
                # noinspection PyTypeChecker
                vector / other

    def test_invalid_mat_mul(self):
        vector = cas.Vector3()
        does_not_work = [
            1.0,
            cas.Point3(),
            cas.RotationMatrix(),
            cas.Quaternion(),
            cas.HomogeneousTransformationMatrix(),
            cas.FloatVariable(name="s"),
            cas.Pose(),
        ]
        for other in does_not_work:
            with pytest.raises(TypeError):
                # noinspection PyTypeChecker
                vector @ other

    @pytest.mark.parametrize("v_data", points)
    def test_norm_property_based(self, v_data):
        v = cas.Vector3.from_iterable(v_data)
        actual = v.norm()

        expected = np.linalg.norm(v_data)

        assert np.allclose(actual, expected)

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
        assert np.allclose(compiled_dot, expected)

        # Test cross product compilation
        cross_expr = v1.cross(v2)
        compiled_cross = cas.Vector3(x_init=2, y_init=3, z_init=4).cross(
            cas.Vector3(x_init=1, y_init=2, z_init=3)
        )
        expected_cross = np.cross([2, 3, 4], [1, 2, 3])
        assert np.allclose(compiled_cross[:3], expected_cross)

    def test_project_to_cone(self):
        v = cas.Vector3.X()
        projected_cone = v.project_to_cone(
            frame_V_cone_axis=cas.Vector3(x_init=1, y_init=1, z_init=0),
            cone_theta=np.pi / 4,
        )
        expected = np.array([1, 0, 0, 0])
        assert np.allclose(projected_cone, expected, atol=1e-10)

        # projected_cone = v.project_to_cone(frame_V_cone_axis=cas.Vector3(1,1,0), cone_theta=np.pi/2)
        # expected = np.array([1, 0, 0, 0])
        # assert np.allclose()(projected_cone, expected, atol=1e-10)


class TestTransformationMatrix:
    def test_json(self):
        """Test that the JSON serialization works correctly"""
        t = cas.HomogeneousTransformationMatrix()
        t_json = t.to_json()
        t_copy = cas.HomogeneousTransformationMatrix.from_json(t_json)
        assert np.allclose(t.to_np(), t_copy.to_np())

    def test_reference_frames(self):
        reference_frame = Body(name=PrefixedName("muh"))
        child_frame = Body(name=PrefixedName("kikariki"))
        reference_frame2 = Body(name=PrefixedName("muh2"))
        child_frame2 = Body(name=PrefixedName("kikariki2"))
        t1 = cas.HomogeneousTransformationMatrix(
            reference_frame=reference_frame, child_frame=child_frame
        )
        t2 = cas.HomogeneousTransformationMatrix(
            reference_frame=reference_frame2, child_frame=child_frame2
        )

        p = cas.Point3(reference_frame=reference_frame)
        v = cas.Vector3(reference_frame=reference_frame)
        r = cas.RotationMatrix(reference_frame=reference_frame)

        assert t1.reference_frame == reference_frame
        assert t1.child_frame == child_frame

        t = cas.HomogeneousTransformationMatrix.from_point_rotation_matrix(p, r)
        assert t.reference_frame == reference_frame
        assert t.child_frame == None

        with pytest.raises(SpatialTypesError):
            t = cas.HomogeneousTransformationMatrix.from_point_rotation_matrix(
                cas.Point3(reference_frame=reference_frame2), r
            )

        t = cas.HomogeneousTransformationMatrix.from_point_rotation_matrix(
            p, r, reference_frame=reference_frame2, child_frame=child_frame2
        )
        assert t.reference_frame == reference_frame2
        assert t.child_frame == child_frame2

        t = cas.HomogeneousTransformationMatrix.from_xyz_rpy(
            reference_frame=reference_frame, child_frame=child_frame
        )
        assert t.reference_frame == reference_frame
        assert t.child_frame == child_frame

        t = cas.HomogeneousTransformationMatrix.from_xyz_quaternion(
            reference_frame=reference_frame, child_frame=child_frame
        )
        assert t.reference_frame == reference_frame
        assert t.child_frame == child_frame

        t = cas.HomogeneousTransformationMatrix.from_xyz_axis_angle(
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
        assert t1.to_translation_matrix().reference_frame == reference_frame
        assert t1.to_rotation_matrix().reference_frame == reference_frame

        assert (t1 @ p).reference_frame == reference_frame
        assert (t1 @ v).reference_frame == reference_frame
        assert (t1 @ r).reference_frame == reference_frame

    def test_matmul_type_preservation(self):
        transform = cas.HomogeneousTransformationMatrix()
        works = [
            cas.Vector3(x_init=1, y_init=1, z_init=1),
            transform,
            cas.RotationMatrix(),
            cas.Pose(),
            cas.Point3(x_init=1, y_init=1, z_init=1),
        ]
        does_not_work = [
            cas.FloatVariable(name="s"),
            cas.Expression(data=1),
            cas.Quaternion(),
        ]

        for other in works:
            assert isinstance(transform @ other, type(other))
        for other in does_not_work:
            with pytest.raises(UnsupportedOperationError):
                # noinspection PyTypeChecker
                transform @ other

    @pytest.mark.parametrize("x", numbers)
    @pytest.mark.parametrize("y", numbers)
    @pytest.mark.parametrize("z", numbers)
    def test_translation3(self, x, y, z):
        r1 = cas.HomogeneousTransformationMatrix.from_xyz_rpy(x, y, z)
        r2 = np.identity(4)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        assert np.allclose(r1, r2)

    def test_dot(self):
        s = cas.FloatVariable(name="x")
        m1 = cas.HomogeneousTransformationMatrix()
        m2 = cas.HomogeneousTransformationMatrix.from_xyz_rpy(x=s)
        m1.dot(m2)

    def test_TransformationMatrix(self):
        f = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3)
        assert isinstance(f, cas.HomogeneousTransformationMatrix)

    def test_wrong_dimensions(self):
        data = np.eye(2)
        with pytest.raises(WrongDimensionsError):
            cas.HomogeneousTransformationMatrix(data=data)

    def test_frame3_axis_angle(self):
        x, y, z, angle = 1, 2, 3, 0.1
        axis = np.array([0, 1, 0])
        r2 = rotation_matrix_from_axis_angle(np.array(axis), angle)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        r = cas.HomogeneousTransformationMatrix.from_point_rotation_matrix(
            cas.Point3(x, y, z), cas.RotationMatrix.from_axis_angle(axis, angle)
        )
        assert np.allclose(r, r2)

    def test_frame3_rpy(self):
        x, y, z, roll, pitch, yaw = 1, 2, 3, 0.1, 0.2, 0.3
        r2 = rotation_matrix_from_rpy(roll, pitch, yaw)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        assert np.allclose(
            cas.HomogeneousTransformationMatrix.from_xyz_rpy(x, y, z, roll, pitch, yaw),
            r2,
        )

    def test_frame3_quaternion(self):
        x, y, z, qx, qy, qz, qw = 1, 2, 3, 1, 0, 0, 0
        r2 = rotation_matrix_from_quaternion(qx, qy, qz, qw)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        r = cas.HomogeneousTransformationMatrix.from_xyz_quaternion(
            x, y, z, qx, qy, qz, qw
        )
        assert np.allclose(r, r2)

    def test_inverse_frame(self):
        t = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)
        actual = cas.HomogeneousTransformationMatrix(data=t).inverse()
        expected = np.linalg.inv(t.to_np())
        assert np.allclose(actual, expected, atol=1.0e-4, rtol=1.0e-4)

    def test_pos_of(self):
        x, y, z, qx, qy, qz, qw = 1, 2, 3, 1, 0, 0, 0
        r1 = cas.HomogeneousTransformationMatrix.from_point_rotation_matrix(
            cas.Point3(x, y, z),
            cas.RotationMatrix.from_quaternion(cas.Quaternion(qx, qy, qz, qw)),
        ).to_position()
        r2 = [x, y, z, 1]
        for i, e in enumerate(r2):
            assert np.allclose(r1[i], e)

    def test_trans_of(self):
        x, y, z, qx, qy, qz, qw = 1, 2, 3, 1, 0, 0, 0
        r1 = cas.HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=cas.Point3(x, y, z),
            rotation_matrix=cas.RotationMatrix.from_quaternion(
                cas.Quaternion(qx, qy, qz, qw)
            ),
        ).to_translation_matrix()
        r2 = np.identity(4)
        r2[0, 3] = x
        r2[1, 3] = y
        r2[2, 3] = z
        assert np.allclose(r1, r2)

    def test_rot_of(self):
        x, y, z, qx, qy, qz, qw = 1, 2, 3, 1, 0, 0, 0
        r1 = cas.HomogeneousTransformationMatrix.from_xyz_quaternion(
            x, y, z, qx, qy, qz, qw
        ).to_rotation_matrix()
        r2 = rotation_matrix_from_quaternion(qx, qy, qz, qw)
        assert np.allclose(r1, r2)

    def test_rot_of2(self):
        """
        Test to make sure the function doesn't alter the original
        """
        f = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3)
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
        t_identity = cas.HomogeneousTransformationMatrix()
        assert isinstance(t_identity, cas.HomogeneousTransformationMatrix)
        identity_np = t_identity
        expected_identity = np.eye(4)
        assert np.allclose(identity_np, expected_identity)

        # From RotationMatrix
        r = cas.RotationMatrix.from_rpy(0.1, 0.2, 0.3)
        t_from_r = cas.HomogeneousTransformationMatrix(r)
        assert isinstance(t_from_r, cas.HomogeneousTransformationMatrix)
        # Should preserve rotation, zero translation
        assert t_from_r[0, 3] == 0
        assert t_from_r[1, 3] == 0
        assert t_from_r[2, 3] == 0

        # From another TransformationMatrix
        t_copy = cas.HomogeneousTransformationMatrix(t_from_r)
        assert isinstance(t_copy, cas.HomogeneousTransformationMatrix)
        assert np.allclose(t_copy, t_from_r)

        # From numpy array
        transform_data = np.eye(4)
        transform_data[:3, 3] = [1, 2, 3]  # Add translation
        t_from_np = cas.HomogeneousTransformationMatrix(data=transform_data)
        assert isinstance(t_from_np, cas.HomogeneousTransformationMatrix)

    def test_sanity_check(self):
        """Test that sanity check enforces proper transformation matrix structure"""
        # Valid 4x4 matrix should pass
        valid_matrix = np.eye(4)
        valid_matrix[:3, 3] = [1, 2, 3]
        t = cas.HomogeneousTransformationMatrix(data=valid_matrix)

        # Check that bottom row is enforced to [0, 0, 0, 1]
        assert t[3, 0] == 0
        assert t[3, 1] == 0
        assert t[3, 2] == 0
        assert t[3, 3] == 1

        # Invalid shape should raise ValueError
        with pytest.raises(WrongDimensionsError):
            cas.HomogeneousTransformationMatrix(data=np.eye(3))  # 3x3 instead of 4x4

        with pytest.raises(WrongDimensionsError):
            cas.HomogeneousTransformationMatrix(
                data=np.ones((2, 5))
            )  # Wrong dimensions

    def test_properties(self):
        """Test x, y, z property getters and setters"""
        t = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)

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
        t1 = cas.HomogeneousTransformationMatrix.from_point_rotation_matrix(p, r)
        assert isinstance(t1, cas.HomogeneousTransformationMatrix)
        assert np.allclose(t1.x, 1)
        assert np.allclose(t1.y, 2)
        assert np.allclose(t1.z, 3)

        # From point only (identity rotation)
        t2 = cas.HomogeneousTransformationMatrix.from_point_rotation_matrix(point=p)
        rotation_part = t2[:3, :3]
        assert np.allclose(rotation_part, np.eye(3))
        assert np.allclose(t2.x, 1)

        # From rotation only (zero translation)
        t3 = cas.HomogeneousTransformationMatrix.from_point_rotation_matrix(
            rotation_matrix=r
        )
        assert np.allclose(t3.x, 0)
        assert np.allclose(t3.y, 0)
        assert np.allclose(t3.z, 0)

    def test_from_xyz_quat(self):
        """Test construction from position and quaternion"""
        t = cas.HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=1,
            pos_y=2,
            pos_z=3,
            quat_w=1,
            quat_x=0,
            quat_y=0,
            quat_z=0,  # Identity quaternion
        )

        assert isinstance(t, cas.HomogeneousTransformationMatrix)
        assert t.x.to_np() == 1
        assert t.y.to_np() == 2
        assert t.z.to_np() == 3

        # Should have identity rotation
        rotation_part = t[:3, :3]
        assert np.allclose(rotation_part, np.eye(3))

    def test_composition(self):
        """Test composition of multiple transformations"""
        # Translation only
        t1 = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 0, 0)  # Translate in X
        t2 = cas.HomogeneousTransformationMatrix.from_xyz_rpy(0, 1, 0)  # Translate in Y

        # Compose transformations
        combined = t2 @ t1
        assert isinstance(combined, cas.HomogeneousTransformationMatrix)

        # Apply to origin point
        origin = cas.Point3(x_init=0, y_init=0, z_init=0)
        result = combined @ origin
        assert isinstance(result, cas.Point3)
        # Should be at (1, 1, 0) after both translations
        assert np.allclose(result.x, 1)
        assert np.allclose(result.y, 1)
        assert np.allclose(result.z, 0)

    def test_point_transformation(self):
        """Test transformation of points"""
        # Create a transformation: translate by (1,2,3) and rotate by 90° around Z
        t = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3, 0, 0, np.pi / 2)

        # Transform a point
        p = cas.Point3(x_init=1, y_init=0, z_init=0)
        transformed = t @ p

        assert isinstance(transformed, cas.Point3)
        assert transformed[3] == 1  # Homogeneous coordinate preserved

        # Expected: rotation transforms (1,0,0) to (0,1,0), then translation adds (1,2,3)
        expected_x = 0 + 1  # 1 (translation)
        expected_y = 1 + 2  # 3 (rotated + translation)
        expected_z = 0 + 3  # 3 (translation)

        assert np.allclose(transformed.x, expected_x, atol=1e-10)
        assert np.allclose(transformed.y, expected_y, atol=1e-10)
        assert np.allclose(transformed.z, expected_z, atol=1e-10)

    def test_vector_transformation(self):
        """Test transformation of vectors (no translation effect)"""
        # Create transformation with both rotation and translation
        t = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3, 0, 0, np.pi / 2)

        # Transform a vector
        v = cas.Vector3(x_init=1, y_init=0, z_init=0)
        transformed = t @ v

        assert isinstance(transformed, cas.Vector3)
        assert transformed[3] == 0  # Vector homogeneous coordinate

        # Only rotation should affect vectors, not translation
        # 90° rotation around Z: (1,0,0) -> (0,1,0)
        assert np.allclose(transformed.x, 0, atol=1e-10)
        assert np.allclose(transformed.y, 1, atol=1e-10)
        assert np.allclose(transformed.z, 0, atol=1e-10)

    def test_inverse(self):
        """Test matrix inversion"""
        t = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)
        t_inv = t.inverse()

        assert isinstance(t_inv, cas.HomogeneousTransformationMatrix)

        # Test that T @ T^(-1) = I
        identity_check = t @ t_inv
        identity = cas.HomogeneousTransformationMatrix()
        assert np.allclose(identity_check, identity, atol=1e-10)

        # Test that T^(-1) @ T = I
        identity_check2 = t_inv @ t
        assert np.allclose(identity_check2, identity, atol=1e-10)

        # Test frame swapping
        if hasattr(t, "reference_frame") and hasattr(t, "child_frame"):
            assert t_inv.reference_frame == t.child_frame
            assert t_inv.child_frame == t.reference_frame

    def test_extraction_methods(self):
        """Test methods for extracting components"""
        t = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)

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
        translation = t.to_translation_matrix()
        assert isinstance(translation, cas.HomogeneousTransformationMatrix)
        # Should have identity rotation
        rotation_part = translation[:3, :3]
        assert np.allclose(rotation_part, np.eye(3))
        # Should preserve translation
        assert translation.x.to_np() == 1
        assert translation.y.to_np() == 2
        assert translation.z.to_np() == 3

        # Extract quaternion
        quaternion = t.to_quaternion()
        assert isinstance(quaternion, cas.Quaternion)

    def test_frame_properties(self):
        """Test reference frame and child frame properties"""
        t = cas.HomogeneousTransformationMatrix()

        # Initially should be None
        assert t.reference_frame is None
        assert t.child_frame is None

        # Test frame preservation in operations
        t1 = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)
        t2 = cas.HomogeneousTransformationMatrix.from_xyz_rpy(4, 5, 6, 0.4, 0.5, 0.6)

        result = t1 @ t2
        # Frame handling depends on implementation
        assert hasattr(result, "reference_frame")
        assert hasattr(result, "child_frame")

    def test_pure_translation(self):
        x, y, z = 1, 2, 3
        t = cas.HomogeneousTransformationMatrix.from_xyz_rpy(x, y, z)

        # Should have identity rotation
        rotation_part = t[:3, :3]
        assert np.allclose(rotation_part, np.eye(3))

        # Should have correct translation
        assert np.allclose(t.x, x)
        assert np.allclose(t.y, y)
        assert np.allclose(t.z, z)

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
        t_sym = cas.HomogeneousTransformationMatrix.from_xyz_rpy(
            x_sym, y_sym, 0, 0, 0, angle_sym
        )

        # Should be able to compose with other transformations
        t_numeric = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 1, 1)
        result = t_sym @ t_numeric

        assert isinstance(result, cas.HomogeneousTransformationMatrix)

        # Should contain the variables
        variables = result.free_variables()
        variable_names = [s.name for s in variables if hasattr(s, "name")]
        assert "x" in variable_names
        assert "y" in variable_names
        assert "theta" in variable_names

    def test_compilation(self):
        """Test compilation and execution of transformation matrices"""
        # Test symbolic transformation compilation
        compiled_transform = cas.HomogeneousTransformationMatrix.from_xyz_rpy(
            1, 2, 3, 0.1, 0.2, 0.3
        )

        # Should be a valid 4x4 transformation matrix
        assert compiled_transform.shape == (4, 4)
        assert np.allclose(compiled_transform[3, 3], 1)
        assert np.allclose(compiled_transform[3, 0], 0)
        assert np.allclose(compiled_transform[3, 1], 0)
        assert np.allclose(compiled_transform[3, 2], 0)

    def test_deepcopy(self):
        """Test deep copy functionality"""
        t = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)

        from copy import deepcopy

        t_copy = deepcopy(t)

        assert isinstance(t_copy, cas.HomogeneousTransformationMatrix)
        assert np.allclose(t, t_copy)

        # Frames should be preserved but not deep copied (reference equality)
        assert t_copy.reference_frame == t.reference_frame
        assert t_copy.child_frame == t.child_frame

    def test_robot_kinematics(self):
        """Test transformation matrices in typical robotics scenarios"""
        # Forward kinematics chain: base -> link1 -> link2 -> end_effector
        base_T_link1 = cas.HomogeneousTransformationMatrix.from_xyz_rpy(
            0, 0, 1, 0, 0, np.pi / 4
        )  # Lift and rotate
        link1_T_link2 = cas.HomogeneousTransformationMatrix.from_xyz_rpy(
            1, 0, 0, 0, np.pi / 2, 0
        )  # Extend and bend
        link2_T_ee = cas.HomogeneousTransformationMatrix.from_xyz_rpy(
            0.5, 0, 0
        )  # End effector offset

        # Forward kinematics
        base_T_ee = base_T_link1 @ link1_T_link2 @ link2_T_ee
        assert isinstance(base_T_ee, cas.HomogeneousTransformationMatrix)

        # Test that inverse kinematics works
        ee_T_base = base_T_ee.inverse()
        identity_check = base_T_ee @ ee_T_base
        identity = cas.HomogeneousTransformationMatrix()
        assert np.allclose(identity_check, identity, atol=1e-10)

    def test_coordinate_transformations(self):
        """Test coordinate frame transformations"""
        # Transform from world to robot base
        world_T_robot = cas.HomogeneousTransformationMatrix.from_xyz_rpy(
            2, 3, 0, 0, 0, np.pi / 2
        )

        # Point in world coordinates
        world_point = cas.Point3(x_init=1, y_init=0, z_init=1)

        # Transform to robot coordinates
        robot_point = world_T_robot.inverse() @ world_point
        assert isinstance(robot_point, cas.Point3)

        # Transform back to world coordinates
        world_point_back = world_T_robot @ robot_point
        assert isinstance(world_point_back, cas.Point3)

        # Should get original point back
        assert np.allclose(world_point, world_point_back, atol=1e-10)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Identity transformation
        t_identity = cas.HomogeneousTransformationMatrix()
        point = cas.Point3(x_init=1, y_init=2, z_init=3)
        transformed = t_identity @ point
        assert np.allclose(point, transformed)

        # Zero translation, identity rotation
        t_zero = cas.HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0, 0, 0, 0)
        assert np.allclose(t_zero, t_identity)

        # Large translation values
        t_large = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1e6, -1e6, 1e6)
        assert t_large.x.to_np() == 1e6
        assert t_large.y.to_np() == -1e6
        assert t_large.z.to_np() == 1e6

        # Small rotation angles (numerical stability)
        t_small = cas.HomogeneousTransformationMatrix.from_xyz_rpy(
            0, 0, 0, 1e-8, 1e-8, 1e-8
        )
        rotation_part = t_small[:3, :3]
        assert np.allclose(rotation_part, np.eye(3), atol=1e-7)

    @pytest.mark.parametrize("q", quaternions)
    def test_quaternion_consistency(self, q):
        t = cas.HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=1,
            pos_y=2,
            pos_z=3,
            quat_w=q[3],
            quat_x=q[0],
            quat_y=q[1],
            quat_z=q[2],
        )

        q_extracted = t.to_quaternion()

        t_roundtrip = cas.HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=1,
            pos_y=2,
            pos_z=3,
            quat_w=q_extracted[3],
            quat_x=q_extracted[0],
            quat_y=q_extracted[1],
            quat_z=q_extracted[2],
        )

        assert np.allclose(t, t_roundtrip, atol=1e-10)

    def test_homogeneous_properties(self):
        t = cas.HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3, 0.1, 0.2, 0.3)

        point = cas.Point3(x_init=4, y_init=5, z_init=6)
        transformed_point = t @ point
        assert transformed_point[3] == 1

        vector = cas.Vector3(x_init=1, y_init=1, z_init=1)
        transformed_vector = t @ vector
        assert transformed_vector[3] == 0

        transform_np = t.to_np()
        assert transform_np[3, 0] == 0
        assert transform_np[3, 1] == 0
        assert transform_np[3, 2] == 0
        assert transform_np[3, 3] == 1


class TestQuaternion:

    @pytest.mark.parametrize("q1", quaternions)
    @pytest.mark.parametrize("q2", quaternions)
    @pytest.mark.parametrize("t", numbers)
    def test_slerp(self, q1, q2, t):
        r1 = cas.Quaternion.from_iterable(q1).slerp(cas.Quaternion.from_iterable(q2), t)
        r2 = quaternion_slerp(q1, q2, t)
        compare_orientations(r1, r2)

    @pytest.mark.parametrize("axis", unit_vectors3)
    @pytest.mark.parametrize("angle", numbers)
    def test_quaternion_from_axis_angle1(self, axis, angle):
        actual = cas.Quaternion.from_axis_angle(cas.Vector3.from_iterable(axis), angle)
        expected = quaternion_from_axis_angle(axis, angle)
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("q", quaternions)
    @pytest.mark.parametrize("p", quaternions)
    def test_quaternion_multiply(self, q, p):
        q_expr = cas.Quaternion.from_iterable(q)
        p_expr = cas.Quaternion.from_iterable(p)
        actual = q_expr.multiply(p_expr)
        expected = quaternion_multiply(q, p)
        compare_orientations(actual, expected)

    @pytest.mark.parametrize("q", quaternions)
    def test_quaternion_conjugate(self, q):
        actual = cas.Quaternion.from_iterable(q).conjugate()
        expected = quaternion_conjugate(q)
        compare_orientations(actual, expected)

    @pytest.mark.parametrize("q1", quaternions)
    @pytest.mark.parametrize("q2", quaternions)
    def test_quaternion_diff(self, q1, q2):
        q1_expr = cas.Quaternion.from_iterable(q1)
        q2_expr = cas.Quaternion.from_iterable(q2)
        actual = q1_expr.diff(q2_expr)
        expected = quaternion_multiply(quaternion_conjugate(q1), q2)
        compare_orientations(actual, expected)

    @pytest.mark.parametrize("q", quaternions)
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

    @pytest.mark.parametrize("roll", numbers)
    @pytest.mark.parametrize("pitch", numbers)
    @pytest.mark.parametrize("yaw", numbers)
    def test_quaternion_from_rpy(self, roll, pitch, yaw):
        q = cas.Quaternion.from_rpy(roll, pitch, yaw)
        q2 = quaternion_from_rpy(roll, pitch, yaw)
        compare_orientations(q, q2)

    @pytest.mark.parametrize("q", quaternions)
    def test_quaternion_from_matrix(self, q):
        matrix = rotation_matrix_from_quaternion(*q)
        q2 = quaternion_from_rotation_matrix(matrix)
        q1_2 = cas.Quaternion.from_rotation_matrix(cas.RotationMatrix(data=matrix))
        compare_orientations(q2, q1_2)

    @pytest.mark.parametrize("q1", quaternions)
    @pytest.mark.parametrize("q2", quaternions)
    def test_dot(self, q1, q2):
        result = cas.Quaternion.from_iterable(q1).dot(cas.Quaternion.from_iterable(q2))
        q1 = np.array(q1)
        q2 = np.array(q2)
        expected = np.dot(q1.T, q2)
        assert np.allclose(result, expected)

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation as R

from semantic_digital_twin.spatial_types import RotationMatrix, Vector3
from semantic_digital_twin.world_description.inertial_properties import (
    InertiaTensor,
    PrincipalMoments,
    PrincipalAxes,
)

moments_and_axes_values = [
    (
        np.array([5.83719208, 3.95164937, 9.27281864]),
        R.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix(),
    ),
    (
        np.array([0.38493155, 7.82294011, 6.19327648]),
        R.from_quat([0.1826, 0.5477, 0.7303, 0.3651]).as_matrix(),
    ),
    (
        np.array([8.10165073, 2.31478824, 9.57593136]),
        R.from_euler("zyx", [45, 30, 60], degrees=True).as_matrix(),
    ),
    (
        np.array([0.47867564, 9.35508812, 2.56453092]),
        R.from_rotvec([0.5, 1.0, 1.5]).as_matrix(),
    ),
    (
        np.array([6.97907492, 1.87208357, 6.81324033]),
        R.from_euler("yxz", [15, 25, 35], degrees=True).as_matrix(),
    ),
    (
        np.array([9.01041187, 0.33042861, 8.40792188]),
        R.from_quat([0.5, 0.5, 0.5, 0.5]).as_matrix(),
    ),
    (
        np.array([4.57660058, 6.88937794, 3.12482013]),
        R.from_euler("zxy", [5, 15, 25], degrees=True).as_matrix(),
    ),
    (
        np.array([3.20547751, 7.77882492, 4.09043471]),
        R.from_rotvec([1.0, 0.5, 0.0]).as_matrix(),
    ),
    (
        np.array([5.2887385, 6.49982344, 1.62593433]),
        R.from_euler("xyz", [90, 45, 30], degrees=True).as_matrix(),
    ),
    (
        np.array([8.37652253, 3.48921285, 7.14783691]),
        R.from_quat([0.7071, 0.0, 0.7071, 0.0]).as_matrix(),
    ),
]


class TestComponentsAndAssembly:
    def test_principal_moments_properties(self):
        for moments_values, _ in moments_and_axes_values:
            m = PrincipalMoments.from_values(*moments_values)
            assert_allclose(m.data, moments_values, atol=1e-12)

    def test_principal_axes_properties(self):
        for _, axes_values in moments_and_axes_values:
            rotation_matrix = RotationMatrix(axes_values)
            axes_1 = PrincipalAxes.from_rotation_matrix(rotation_matrix)
            axes_2 = PrincipalAxes(data=axes_values)
            axes_3 = axes_2.to_rotation_matrix().to_np()[:3, :3]
            assert_allclose(axes_1.data, axes_values, atol=1e-12)
            assert_allclose(axes_2.data, axes_values, atol=1e-12)
            assert_allclose(axes_3, axes_values, atol=1e-12)

    def test_inertia_tensor_properties(self):
        for moments_values, axes_values in moments_and_axes_values:
            expected_tensor = axes_values @ np.diag(moments_values) @ axes_values.T
            ixx = expected_tensor[0, 0]
            iyy = expected_tensor[1, 1]
            izz = expected_tensor[2, 2]
            ixy = expected_tensor[0, 1]
            ixz = expected_tensor[0, 2]
            iyz = expected_tensor[1, 2]

            moments = PrincipalMoments.from_values(*moments_values)
            axes = PrincipalAxes(data=axes_values)
            tensor_from_principal = InertiaTensor.from_principal_moments_and_axes(
                moments=moments, axes=axes
            )
            tensor_from_values = InertiaTensor.from_values(
                ixx=ixx, iyy=iyy, izz=izz, ixy=ixy, ixz=ixz, iyz=iyz
            )

            for each_tensor in [tensor_from_principal, tensor_from_values]:
                assert_allclose(each_tensor.data, expected_tensor, atol=1e-12)

                ixx_out, iyy_out, izz_out, ixy_out, ixz_out, iyz_out = (
                    each_tensor.to_values()
                )
                assert_allclose(
                    [ixx_out, iyy_out, izz_out, ixy_out, ixz_out, iyz_out],
                    [ixx, iyy, izz, ixy, ixz, iyz],
                    atol=1e-12,
                )
                moments_out, axes_out = each_tensor.to_principal_moments_and_axes(
                    sorted_array=moments_values
                )
                assert_allclose(moments_out.data, moments_values, atol=1e-12)
                sorted_tensor = (
                    axes_out.data @ np.diag(moments_out.data) @ axes_out.data.T
                )
                assert_allclose(sorted_tensor, expected_tensor, atol=1e-12)

import numpy as np


def assert_quat_equal_up_to_sign(actual_quat, expected_quat):
    actual = np.array(actual_quat)
    expected = np.array(expected_quat)
    assert np.allclose(actual, expected) or np.allclose(actual, -expected)

import numpy as np
from semantic_digital_twin.datastructures.types import AnyMatrix4x4, NpMatrix4x4


def inverse_frame(f1_T_f2: NpMatrix4x4) -> NpMatrix4x4:
    """
    :param f1_T_f2: 4x4 Matrix
    :return: f2_T_f1
    """
    R = f1_T_f2[:3, :3]
    t = f1_T_f2[:3, 3]
    Rt = R.T
    f2_T_f1 = np.empty((4, 4), dtype=f1_T_f2.dtype)
    f2_T_f1[:3, :3] = Rt
    f2_T_f1[:3, 3] = -Rt @ t
    f2_T_f1[3] = [0, 0, 0, 1]
    return f2_T_f1

"""
Serialisation of the CasADi boundary for spatial types.
"""

from __future__ import annotations

import threading
from copy import deepcopy

from krrood.symbolic_math.symbolic_math import CasadiLock

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix

BLOCKED_WAIT_IN_SECONDS = 0.5
"""
How long a copy that must be blocked is given to prove that it is.
"""

COMPLETION_WAIT_IN_SECONDS = 5.0
"""
How long a copy that must succeed is given to finish once it is unblocked.
"""


def test_copying_a_spatial_type_waits_for_the_casadi_lock():
    """
    Copying a spatial type reaches into CasADi, so it must not run while another thread
    holds the CasADi lock.
    """
    matrix = HomogeneousTransformationMatrix()
    copy_finished = threading.Event()

    def copy_matrix() -> None:
        deepcopy(matrix)
        copy_finished.set()

    copying_thread = threading.Thread(target=copy_matrix, daemon=True)
    with CasadiLock():
        copying_thread.start()
        assert not copy_finished.wait(timeout=BLOCKED_WAIT_IN_SECONDS)

    copying_thread.join(timeout=COMPLETION_WAIT_IN_SECONDS)
    assert copy_finished.is_set()

# %% imports

from __future__ import annotations

from dataclasses import dataclass

import pytest

from semantic_digital_twin.exceptions import RobotAlreadyInWorldError
from semantic_digital_twin.robots.pr2 import PR2

# %% mimic robot type


@dataclass(eq=False)
class _RobotOfDifferentTypeAtSameRoot(PR2):
    """
    A robot annotation type distinct from PR2 whose root body coincides with PR2's.
    """


# %% root-body uniqueness guard


class TestRobotAnnotationRootUniqueness:
    """
    Validates that a root body carries at most one robot annotation of any type.
    """

    def test_different_robot_type_at_same_root_is_rejected(self, pr2_world_copy):
        """
        Creating a robot annotation of another type at an already-annotated root raises,
        instead of stacking a second robot onto the same body.
        """
        with pytest.raises(RobotAlreadyInWorldError):
            _RobotOfDifferentTypeAtSameRoot.from_world(pr2_world_copy)

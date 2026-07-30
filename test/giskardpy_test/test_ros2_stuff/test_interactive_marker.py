import pytest

from giskardpy.data_types.exceptions import IncompleteKinematicChainParametersError
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    InteractiveMarkerNode,
)

# %% root/tip link pairing validation


def test_only_root_links_raises():
    with pytest.raises(IncompleteKinematicChainParametersError):
        InteractiveMarkerNode(root_links=["map"])


def test_only_tip_links_raises():
    with pytest.raises(IncompleteKinematicChainParametersError):
        InteractiveMarkerNode(tip_links=["hand"])

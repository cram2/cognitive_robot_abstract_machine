import pytest

from giskardpy.data_types.exceptions import UnpairedKinematicChainParametersError
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import RootTipPair

# %% pairing root links with tip links

ROOT_LINKS = ["map", "arm_mount_left_link"]
"""
Root links of two chains, used to pair against a matching and a mismatched tip list.
"""

TIP_LINKS = ["right_hand_tcp", "left_hand_tcp"]
"""
Tip links belonging to :data:`ROOT_LINKS`, in the same order.
"""


def test_endpoints_are_paired_by_index():
    chains = RootTipPair.pair_up(ROOT_LINKS, TIP_LINKS)

    assert chains == [
        RootTipPair(root_link=ROOT_LINKS[0], tip_link=TIP_LINKS[0]),
        RootTipPair(root_link=ROOT_LINKS[1], tip_link=TIP_LINKS[1]),
    ]


def test_more_root_links_than_tip_links_raises():
    with pytest.raises(UnpairedKinematicChainParametersError):
        RootTipPair.pair_up(ROOT_LINKS, TIP_LINKS[:1])


def test_more_tip_links_than_root_links_raises():
    with pytest.raises(UnpairedKinematicChainParametersError):
        RootTipPair.pair_up(ROOT_LINKS[:1], TIP_LINKS)

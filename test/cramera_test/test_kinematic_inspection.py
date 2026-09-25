"""
Read standard URDF descriptions independently of XML attribute ordering.
"""

from pathlib import Path

from coraplex.datastructures.enums import JointType
from cramera.knowledge.scene_bundle import ParsedUrdf, UrdfJoint


# %% standard robot descriptions
def test_kinematics_accepts_reordered_joint_attributes(fixture_scene: Path) -> None:
    """
    Preserve the complete kinematic tree of a valid reordered URDF.

    :param fixture_scene: Existing recorded scene fixture and its robot asset.
    """
    description = Path(__file__).parent / "dataset" / "reordered_attributes.urdf"
    (fixture_scene / "scenes" / "fixture" / "robot.urdf").write_text(
        description.read_text()
    )

    parsed = ParsedUrdf.of_scene("fixture")

    assert parsed.links == ["base", "tool"]
    assert parsed.joints == [UrdfJoint("tool_mount", JointType.FIXED, "base", "tool")]

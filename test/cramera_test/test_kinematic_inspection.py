"""
Inspect standard robot descriptions and their kinematic graph views.
"""

from dataclasses import replace
from pathlib import Path

from coraplex.datastructures.enums import JointType
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.knowledge.scene_bundle import ParsedUrdf, UrdfJoint
from cramera.knowledge.views.kinematics import UrdfViewPayload


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


# %% link identifiers
def test_link_identifiers_use_the_view_prefix(fixture_scene: Path) -> None:
    """
    A view owns its link namespace independently of other views.

    :param fixture_scene: Existing recorded scene fixture and its robot asset.
    """
    knowledge_base = EpisodeKnowledgeBase.of_scene(None)
    view = UrdfViewPayload.of_tab(knowledge_base)
    prefix = knowledge_base.robot.name + ":"
    other_view = replace(view, link_prefix=prefix)
    link = ParsedUrdf.of_scene().links[0]

    assert other_view.link_id(link) == prefix + link
    assert view.link_id(link) == view.nodes[0].id
    assert other_view.link_id(link) != view.link_id(link)


def test_kinematic_graph_uses_its_view_link_identifiers(fixture_scene: Path) -> None:
    """
    Nodes, edges and details address the same links as scene highlights.

    :param fixture_scene: Existing recorded scene fixture and its robot asset.
    """
    view = UrdfViewPayload.of_tab(EpisodeKnowledgeBase.of_scene(None))
    parsed = ParsedUrdf.of_scene()
    identifiers = {view.link_id(link) for link in parsed.links}

    assert {node.id for node in view.nodes} == identifiers
    assert set(view.details) == identifiers
    assert {(edge.source, edge.target) for edge in view.edges} == {
        (view.link_id(joint.parent), view.link_id(joint.child))
        for joint in parsed.joints
    }


def test_missing_description_returns_an_empty_kinematic_view(
    fixture_scene: Path,
) -> None:
    """
    A missing robot asset yields an empty graph without a part legend.

    :param fixture_scene: Existing recorded scene fixture and its robot asset.
    """
    knowledge_base = EpisodeKnowledgeBase.of_scene(None)
    (fixture_scene / "scenes" / "fixture" / "robot.urdf").unlink()

    view = UrdfViewPayload.of_tab(knowledge_base)

    assert view.nodes == []
    assert view.edges == []
    assert view.details == {}
    assert view.legend is None

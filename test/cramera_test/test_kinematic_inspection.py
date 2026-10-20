"""
Inspect standard robot descriptions and their kinematic graph views.
"""

from pathlib import Path

import pytest

from coraplex.datastructures.enums import JointType
from cramera.knowledge.enums import SceneEntityPrefix
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
def test_kinematic_graph_uses_shared_link_identifiers(fixture_scene: Path) -> None:
    """
    Nodes, edges and details address the same links as scene highlights.

    :param fixture_scene: Existing recorded scene fixture and its robot asset.
    """
    view = UrdfViewPayload.of_tab(EpisodeKnowledgeBase.of_scene(None))
    parsed = ParsedUrdf.of_scene()
    identifiers = {SceneEntityPrefix.URDF_LINK + link for link in parsed.links}

    assert {node.id for node in view.nodes} == identifiers
    assert set(view.details) == identifiers
    assert {(edge.source, edge.target) for edge in view.edges} == {
        (
            SceneEntityPrefix.URDF_LINK + joint.parent,
            SceneEntityPrefix.URDF_LINK + joint.child,
        )
        for joint in parsed.joints
    }


def test_kinematic_edges_require_both_link_endpoints(
    fixture_scene: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Joints referencing an absent parent or child never create dangling graph edges.

    :param fixture_scene: Existing recorded scene fixture and its robot asset.
    :param monkeypatch: The active monkeypatch fixture.
    """
    knowledge_base = EpisodeKnowledgeBase.of_scene(None)
    expected = UrdfViewPayload.of_tab(knowledge_base)
    parsed = ParsedUrdf.of_scene()
    parsed.joints.extend(
        [
            UrdfJoint("missing_parent", JointType.FIXED, "absent", parsed.links[0]),
            UrdfJoint("missing_child", JointType.FIXED, parsed.links[0], "absent"),
        ]
    )
    monkeypatch.setattr(ParsedUrdf, "of_scene", lambda scene_name: parsed)

    view = UrdfViewPayload.of_tab(knowledge_base)

    assert view.edges == expected.edges
    assert {node.id for node in view.nodes} == {node.id for node in expected.nodes}


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

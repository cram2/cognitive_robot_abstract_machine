"""
Tests for the scene-driven knowledge base and its graph-panel payloads.
"""

import json
import os

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from coraplex.datastructures.enums import Arms  # noqa: E402

from semantic_digital_twin.datastructures.prefixed_name import (
    PrefixedName,
)  # noqa: E402
from semantic_digital_twin.spatial_types import Point3  # noqa: E402
from semantic_digital_twin.world_description.world_entity import Body  # noqa: E402

from cramera.knowledge.entities import BenchObject  # noqa: E402
from cramera.knowledge.eql_session import EqlSession, RowRenderer  # noqa: E402
from cramera.knowledge.graph_payload import KnowledgeGraphPayload  # noqa: E402
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase  # noqa: E402
from cramera.knowledge.presets import Preset  # noqa: E402
from cramera.knowledge.views.architecture import SubgraphViewPayload  # noqa: E402
from cramera.knowledge.views.dispatcher import GraphPanelViews  # noqa: E402
from cramera.knowledge.views.plan_tree import PlanViewPayload  # noqa: E402
from cramera.knowledge import knowledge_base  # noqa: E402
from cramera.knowledge.architecture_entities import (  # noqa: E402
    Package,
    PythonClass,
)
from cramera.knowledge.architecture_scan import ArchitectureScanner  # noqa: E402
from cramera.knowledge.enums import (  # noqa: E402
    EdgeKind,
    KinematicChainGroup,
    NodeGroup,
    PlanNodeGroup,
)
from cramera.knowledge.scene_bundle import SceneBundle  # noqa: E402
from cramera.knowledge.subgraph import DetailEntry, GraphEdge  # noqa: E402
from cramera.knowledge.views import plan_tree as plan_view  # noqa: E402
from cramera.robot_parts import (  # noqa: E402
    ArmSide,
    RobotPartAnnotation,
    RobotPartRole,
)


@pytest.fixture()
def fresh_knowledge_base(fixture_scene):
    EpisodeKnowledgeBase.reset()
    return EpisodeKnowledgeBase.of_active_scene()


class TestKnowledgeBaseFreshness:
    """
    The per-scene cache must serve a rebuilt bundle fresh: the live scene's
    ``scene.json`` is rewritten by the bridge on every attach, and a knowledge base
    built from the old bundle would keep answering for a world that no longer exists.
    """

    def test_an_unchanged_bundle_stays_cached(self, fixture_scene):
        EpisodeKnowledgeBase.reset()
        first = EpisodeKnowledgeBase.of_scene("fixture")

        assert EpisodeKnowledgeBase.of_scene("fixture") is first

    def test_a_rewritten_bundle_rebuilds_the_knowledge_base(self, fixture_scene):
        EpisodeKnowledgeBase.reset()
        first = EpisodeKnowledgeBase.of_scene("fixture")
        scene_path = fixture_scene / "scenes" / "fixture" / "scene.json"
        scene = json.loads(scene_path.read_text())
        scene["robot"]["name"] = "renamed_robot"
        scene_path.write_text(json.dumps(scene))
        stat = scene_path.stat()
        os.utime(scene_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

        rebuilt = EpisodeKnowledgeBase.of_scene("fixture")

        assert rebuilt is not first
        assert rebuilt.robot.name == "renamed_robot"


class TestEpisodeKnowledgeBase:
    def test_scene_entities(self, fresh_knowledge_base):
        assert [o.name for o in fresh_knowledge_base.objects] == ["milk", "place_area"]
        assert fresh_knowledge_base.robot.name == "pr2"
        assert [a.side for a in fresh_knowledge_base.arms] == [Arms.LEFT]
        assert fresh_knowledge_base.arms[0].gripper.name == "left_gripper"

    def test_episodes_link_objects(self, fresh_knowledge_base):
        transport = next(
            e for e in fresh_knowledge_base.episodes if e.name == "transport_milk"
        )
        assert transport.picks is fresh_knowledge_base.objects[0]
        assert transport.places_at.name == "place_area"
        assert transport.performed_by.side == Arms.LEFT

    def test_joint_motion_ranges(self, fresh_knowledge_base):
        torso = next(
            j for j in fresh_knowledge_base.joints if j.name == "torso_lift_joint"
        )
        assert torso.minimum_radians == 0.0 and torso.maximum_radians == 0.3

    def test_architecture_scan(self, fresh_knowledge_base):
        names = {p.name for p in fresh_knowledge_base.packages}
        assert {"coraplex", "krrood"} <= names
        assert any(c.name == "Plan" for c in fresh_knowledge_base.classes)


class TestArchitectureScanner:
    def test_scan_returns_real_entities_without_an_intermediate_dict(
        self, fixture_scene
    ):
        """
        ``scan()`` must hand back typed ``Package``/``PythonClass`` instances directly,
        not the raw dicts the on-disk cache stores.
        """
        result = ArchitectureScanner.of_configured_root().scan()
        assert result.packages and all(
            isinstance(package, Package) for package in result.packages
        )
        assert result.classes and all(
            isinstance(python_class, PythonClass) for python_class in result.classes
        )
        plan_class = next(c for c in result.classes if c.name == "Plan")
        assert plan_class.subpackage == "coraplex.plans"

    def test_the_scanner_reads_the_root_it_was_given(self, fixture_scene, tmp_path):
        """
        The root is a field, not something each method reads from the environment, so a
        scanner can be pointed at another checkout.
        """
        other_root = tmp_path / "other_checkout"
        (other_root / "solo_package").mkdir(parents=True)
        (other_root / "solo_package" / "module.py").write_text(
            "class Alone:\n    pass\n"
        )

        result = ArchitectureScanner(root=str(other_root)).scan()

        # "root" is the synthetic entry for loose top-level scripts
        assert [package.name for package in result.packages] == ["root", "solo_package"]
        assert [python_class.name for python_class in result.classes] == ["Alone"]

    def test_load_caches_the_scan_on_disk(self, fixture_scene):
        """
        ``load()`` must return the same entities as ``scan()``, from the cache on a
        second call.
        """
        scanner = ArchitectureScanner.of_configured_root()
        scanned = scanner.scan()
        loaded_once = scanner.load()
        loaded_again = scanner.load()
        assert {p.name for p in loaded_once.packages} == {
            p.name for p in scanned.packages
        }
        assert loaded_once.classes == loaded_again.classes


class TestArmsFromRecordedAnnotations:
    def test_the_recorded_annotations_decide_the_arms_and_their_sides(
        self, fixture_scene, monkeypatch
    ):
        """
        A bundle carrying sem_dt robot-part annotations is read straight off them, so an
        arm whose name spells no side still gets the side its robot annotated it with.
        """
        bundle = SceneBundle.of_active_scene()
        scene, trajectory = bundle.scene, bundle.trajectory
        scene["robot"]["partAnnotations"] = [
            RobotPartAnnotation(
                name="ManipulatorOne",
                role=RobotPartRole.ARM,
                side=ArmSide.RIGHT,
                links=["upper_link"],
            ).to_payload(),
            RobotPartAnnotation(
                name="HandOne",
                role=RobotPartRole.END_EFFECTOR,
                side=ArmSide.RIGHT,
                links=["hand_link"],
                attached_to="ManipulatorOne",
            ).to_payload(),
        ]
        monkeypatch.setattr(
            SceneBundle,
            "of_scene",
            lambda scene_name=None: SceneBundle(scene, trajectory),
        )
        EpisodeKnowledgeBase.reset()
        knowledge_base_instance = EpisodeKnowledgeBase.of_active_scene()

        [arm] = knowledge_base_instance.arms
        assert arm.name == "ManipulatorOne"
        assert arm.side == Arms.RIGHT
        assert arm.gripper.name == "HandOne"
        assert arm.gripper.side == Arms.RIGHT
        assert knowledge_base_instance.grippers == [arm.gripper]


class TestArmSideInference:
    def test_an_arm_whose_name_encodes_no_side_is_unknown(
        self, fixture_scene, monkeypatch
    ):
        """
        An arm part name that names neither `left` nor `right` cannot be assigned a side
        by name inspection, and must not silently masquerade as one.
        """
        bundle = SceneBundle.of_active_scene()
        scene, trajectory = bundle.scene, bundle.trajectory
        scene["robot"]["parts"]["center_arm"] = ["center_link"]
        monkeypatch.setattr(
            SceneBundle,
            "of_scene",
            lambda scene_name=None: SceneBundle(scene, trajectory),
        )
        EpisodeKnowledgeBase.reset()
        center_arm = next(
            arm
            for arm in EpisodeKnowledgeBase.of_active_scene().arms
            if arm.name == "center_arm"
        )
        assert center_arm.side is None


class TestQueries:
    def test_entity_query(self, fixture_scene):
        result = EqlSession.of_active_scene().run(
            "the(entity(scene_object).where(scene_object.name == 'milk'))"
        )
        assert result.ok and result.count == 1
        assert result.rows[0]["__entity__"] == "milk"
        assert "milk" in result.highlight

    def test_a_set_of_query_returns_its_unification_rows(self, fixture_scene):
        """
        ``set_of`` answers with a mapping per row rather than an entity, and that path
        is only reached at query time — an import missing from it survives collection.
        """
        result = EqlSession.of_active_scene().run(
            "set_of(scene_object.name, scene_object.kind)"
        )

        assert result.ok
        assert result.rows == [
            {"BenchObject.name": "milk", "BenchObject.kind": "object"},
            {"BenchObject.name": "place_area", "BenchObject.kind": "location"},
        ]

    def test_only_a_real_entity_is_treated_as_one(self):
        """
        A result value is an entity because of its type, not because it happens to carry
        a ``name``: semantic_digital_twin's ``Body`` is a dataclass with one and must
        not be reported as an entity to highlight.
        """
        milk = BenchObject(
            name="milk",
            kind="object",
            label="Milk",
            height_metres=None,
            position=Point3(0.0, 0.0, 0.0),
        )
        body = Body(name=PrefixedName("milk"))

        assert RowRenderer._entity_name(milk) == "milk"
        assert RowRenderer._entity_name(body) is None

    def test_an_unknown_name_raises(self, fixture_scene):
        """
        A query naming something the namespace does not define must raise.

        The server turns this into a JSON error payload; the knowledge base itself does
        not swallow it.
        """
        with pytest.raises(NameError):
            EqlSession.of_active_scene().run("this is not python")

    def test_a_syntactically_invalid_query_raises(self, fixture_scene):
        with pytest.raises(SyntaxError):
            EqlSession.of_active_scene().run("definitely not python (((")


class TestRecordedMeasurements:
    def test_an_unrecorded_height_stays_unknown(self, fresh_knowledge_base):
        """
        The fixture bundle records no object height, so none may be invented.
        """
        milk = next(
            entry for entry in fresh_knowledge_base.objects if entry.name == "milk"
        )
        assert milk.height_metres is None

    def test_an_unrecorded_gripper_opening_stays_unknown(self, fresh_knowledge_base):
        assert fresh_knowledge_base.arms[0].gripper.opening_metres is None

    def test_a_recorded_height_is_used(self, fixture_scene, monkeypatch):
        """
        A bundle that reports a height must be taken at its word.
        """
        bundle = SceneBundle.of_active_scene()
        scene, trajectory = bundle.scene, bundle.trajectory
        scene["objects"][0]["height"] = 0.23
        monkeypatch.setattr(
            SceneBundle,
            "of_scene",
            lambda scene_name=None: SceneBundle(scene, trajectory),
        )
        EpisodeKnowledgeBase.reset()
        milk = next(
            entry
            for entry in EpisodeKnowledgeBase.of_active_scene().objects
            if entry.name == "milk"
        )
        assert milk.height_metres == 0.23

    def test_unknown_measurements_are_left_out_of_the_graph(self, fixture_scene):
        """
        A tooltip must not show a height the bundle never recorded.
        """
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")
        milk = payload.details["milk"]
        assert not any(line.startswith("height:") for line in milk.lines)


class TestActionLabelShortening:
    def test_action_suffix_is_dropped(self):
        assert PlanViewPayload._shorten_action_label("TransportAction") == "Transport"

    def test_the_word_action_inside_a_label_is_kept(self):
        assert PlanViewPayload._shorten_action_label("ActionNode") == "ActionNode"

    def test_only_the_trailing_occurrence_is_dropped(self):
        assert (
            PlanViewPayload._shorten_action_label("ActionSequenceAction")
            == "ActionSequence"
        )

    def test_a_label_that_is_only_the_suffix_is_kept(self):
        assert PlanViewPayload._shorten_action_label("Action") == "Action"


class TestViewPayloads:
    def test_knowledge_view(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")
        assert payload.ok
        ids = {n.id for n in payload.nodes}
        assert {"pr2", "milk", "transport_milk", "plan"} <= ids
        assert payload.presets

    def test_kinematics_view(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("kinematics")
        assert payload.ok
        ids = {n.id for n in payload.nodes}
        assert "urdf:base_link" in ids and "urdf:l_gripper_link" in ids
        # fixed joints render dashed ('type'), movable solid ('prop')
        kinds = {e.label.split(" ")[0]: e.kind for e in payload.edges}
        assert kinds["torso_lift_joint"] == EdgeKind.PROPERTY
        assert kinds["l_gripper_joint"] == EdgeKind.TYPE

    def test_kinematics_links_carry_their_own_colour_groups(self, fixture_scene):
        """
        The URDF tree groups links by robot part, not by the knowledge graph's
        ontological categories — a right arm is not an "event".
        """
        payload = GraphPanelViews.of_active_scene().for_tab("kinematics")
        by_id = {node.id: node.group for node in payload.nodes}

        assert by_id["urdf:torso_link"] == KinematicChainGroup.BASE
        assert by_id["urdf:l_shoulder_link"] == KinematicChainGroup.LEFT_ARM
        assert by_id["urdf:l_gripper_link"] == KinematicChainGroup.GRIPPER
        assert not {group for group in by_id.values()} & set(NodeGroup)

    def test_the_kinematics_legend_names_every_chain_group(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("kinematics")

        assert [row.group for row in payload.legend] == [
            KinematicChainGroup.BASE,
            KinematicChainGroup.LEFT_ARM,
            KinematicChainGroup.RIGHT_ARM,
            KinematicChainGroup.GRIPPER,
            KinematicChainGroup.SENSOR,
        ]

    def test_kinematics_edge_label_shows_the_urdf_joint_type(self, fixture_scene):
        """
        ``UrdfJoint.type`` is a :class:`~coraplex.datastructures.enums.JointType` now,
        but the tooltip must still read the plain URDF word (``prismatic``), not the
        enum member's own text (``JointType.PRISMATIC``).
        """
        payload = GraphPanelViews.of_active_scene().for_tab("kinematics")
        torso_edge = next(
            e for e in payload.edges if e.label.startswith("torso_lift_joint")
        )
        assert torso_edge.label == "torso_lift_joint (prismatic)"

    def test_kinematics_counts_every_movable_joint(self, fixture_scene):
        """
        The movable-joint tally must match the joints drawn as movable.

        The fixture's ``torso_lift_joint`` is prismatic: movable, but not revolute.
        """
        payload = GraphPanelViews.of_active_scene().for_tab("kinematics")
        movable_edges = [
            edge for edge in payload.edges if edge.kind == EdgeKind.PROPERTY
        ]
        root_lines = payload.details["urdf:base_link"].lines
        summary = next(line for line in root_lines if "movable" in line)
        assert summary.endswith("(%d movable)" % len(movable_edges))

    def test_plan_view_carries_status(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("plan")
        rendered = payload.to_payload()
        assert payload.ok and rendered["layout"] == "hier"
        assert rendered["live"] == "plan" and rendered["statusLegend"]
        by_label = {n.label: n for n in payload.nodes}
        assert by_label["SequentialNode"].status == "SUCCEEDED"
        # recorded inner nodes stay CREATED (only the root is performed)
        assert by_label["Transport"].status == "CREATED"
        assert len(payload.edges) == len(payload.nodes) - 1

    def test_plan_view_legend(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("plan")
        expected = [
            {"group": entry.group, "label": entry.label}
            for entry in plan_view.PLAN_LEGEND
        ]
        assert payload.to_payload()["legend"] == expected

    def test_chart_view_is_live_only(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("chart")
        rendered = payload.to_payload()
        assert payload.ok and rendered["nodes"] == []
        assert rendered["live"] == "chart" and rendered["empty"]

    def test_unknown_view(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("bogus")
        assert not payload.ok


# %% BUG-1 -- attach/detach plan-node grouping
class TestPlanGroups:
    def test_attach_node_renders_in_the_attachment_group(
        self, fixture_scene, monkeypatch
    ):
        """
        Coraplex's real class is ``AttachNode``, not ``AttachmentNode``.
        """
        bundle = SceneBundle.of_active_scene()
        scene, trajectory = bundle.scene, bundle.trajectory
        scene["planTrees"][0]["children"].append(
            {
                "kind": "AttachNode",
                "label": "AttachNode",
                "status": "CREATED",
                "children": [],
            }
        )
        monkeypatch.setattr(
            SceneBundle,
            "of_scene",
            lambda scene_name=None: SceneBundle(scene, trajectory),
        )
        EpisodeKnowledgeBase.reset()
        node = next(
            n
            for n in GraphPanelViews.of_active_scene().for_tab("plan").nodes
            if n.label == "AttachNode"
        )
        assert node.group == PlanNodeGroup.ATTACHMENT

    def test_detach_node_renders_in_the_attachment_group(
        self, fixture_scene, monkeypatch
    ):
        """
        Coraplex's real class is ``DetachNode``, not ``DetachmentNode``.
        """
        bundle = SceneBundle.of_active_scene()
        scene, trajectory = bundle.scene, bundle.trajectory
        scene["planTrees"][0]["children"].append(
            {
                "kind": "DetachNode",
                "label": "DetachNode",
                "status": "CREATED",
                "children": [],
            }
        )
        monkeypatch.setattr(
            SceneBundle,
            "of_scene",
            lambda scene_name=None: SceneBundle(scene, trajectory),
        )
        EpisodeKnowledgeBase.reset()
        node = next(
            n
            for n in GraphPanelViews.of_active_scene().for_tab("plan").nodes
            if n.label == "DetachNode"
        )
        assert node.group == PlanNodeGroup.ATTACHMENT


# %% BUG-2 -- EQL preset splicing
class TestPresetSafety:
    def test_an_apostrophe_in_an_object_name_does_not_break_its_preset(
        self, fixture_scene, monkeypatch
    ):
        """
        ``Preset.of_scene()`` must escape object names, not splice them raw into EQL
        source.
        """
        bundle = SceneBundle.of_active_scene()
        scene, trajectory = bundle.scene, bundle.trajectory
        scene["objects"][0]["id"] = "o'brien"
        scene["segments"][1]["picks"] = "o'brien"
        monkeypatch.setattr(
            SceneBundle,
            "of_scene",
            lambda scene_name=None: SceneBundle(scene, trajectory),
        )
        EpisodeKnowledgeBase.reset()
        preset = next(p for p in Preset.of_scene() if "scene_object.name" in p.code)
        result = EqlSession.of_active_scene().run(preset.code)
        assert result.ok and result.rows[0]["__entity__"] == "o'brien"

    def test_an_apostrophe_in_an_episode_name_does_not_break_its_presets(
        self, fixture_scene, monkeypatch
    ):
        """
        Covers both the ``places_at`` and ``performed_by`` presets, which splice the
        same episode name.
        """
        bundle = SceneBundle.of_active_scene()
        scene, trajectory = bundle.scene, bundle.trajectory
        scene["segments"][1]["step"] = "transport_o'brien"
        monkeypatch.setattr(
            SceneBundle,
            "of_scene",
            lambda scene_name=None: SceneBundle(scene, trajectory),
        )
        EpisodeKnowledgeBase.reset()
        for preset in Preset.of_scene():
            assert EqlSession.of_active_scene().run(preset.code).ok


# %% characterization: GraphPanelViews.of_active_scene().for_tab("knowledge") structure
class TestGraphPayloadStructure:
    def test_robot_arm_gripper_chain(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")
        by_id = {n.id: n for n in payload.nodes}
        assert by_id["pr2"].label == "pr2" and by_id["pr2"].group == NodeGroup.ROBOT
        assert by_id["left_arm"].label == "left arm"
        assert by_id["left_gripper"].label == "left gripper"
        chain_edges = [e for e in payload.edges if e.label == "has part"]
        assert chain_edges == [
            GraphEdge("pr2", "left_arm", EdgeKind.PROPERTY, "has part"),
            GraphEdge("left_arm", "left_gripper", EdgeKind.PROPERTY, "has part"),
        ]
        assert payload.details["pr2"] == DetailEntry(
            "pr2",
            NodeGroup.ROBOT,
            ["a Robot", "1 arm", "double-click: full URDF tree"],
        )

    def test_episode_chain(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")
        episode_edges = [
            e
            for e in payload.edges
            if e.label in ("precedes", "performed by", "picks", "places at")
        ]
        assert episode_edges == [
            GraphEdge("prepare", "transport_milk", EdgeKind.TYPE, "precedes"),
            GraphEdge("transport_milk", "pr2", EdgeKind.PROPERTY, "performed by"),
            GraphEdge("transport_milk", "milk", EdgeKind.PROPERTY, "picks"),
            GraphEdge("transport_milk", "place_area", EdgeKind.PROPERTY, "places at"),
        ]

    def test_object_detail_lines(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")
        assert payload.details["milk"] == DetailEntry(
            "Milk",
            NodeGroup.OBJECT,
            [
                "a BenchObject",
                "kind: object",
                "position: (2.37, 2.00, 1.05)",
            ],
        )
        # place_area's height (0.0) is recorded, unlike milk's, so its measurement
        # line is present
        assert payload.details["place_area"] == DetailEntry(
            "Place area",
            NodeGroup.OBJECT,
            [
                "a BenchObject",
                "kind: location",
                "position: (4.90, 3.30, 0.72)",
                "height: 0.00 m",
            ],
        )

    def test_architecture_cluster(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")
        ids = {n.id for n in payload.nodes}
        assert {"cram", "root", "coraplex", "krrood", "coraplex.plans"} <= ids
        assert payload.details["cram"] == DetailEntry(
            "CRAM architecture",
            NodeGroup.ROOT,
            [
                "~/cognitive_robot_abstract_machine",
                "3 packages · 4 Python classes",
            ],
        )
        assert payload.details["coraplex"] == DetailEntry(
            "coraplex",
            NodeGroup.PACKAGE,
            [
                "a Package",
                "the plan executive: designators, plans, locations",
                "2 modules · 2 classes",
                "double-click to open",
            ],
        )
        assert payload.details["coraplex.plans"] == DetailEntry(
            "plans",
            NodeGroup.SUBPACKAGE,
            [
                "a SubPackage of coraplex",
                "2 modules · 2 classes",
                "double-click to open",
            ],
        )
        contains_edges = [e for e in payload.edges if e.label == "contains"]
        assert contains_edges == [
            GraphEdge("cram", "root", EdgeKind.PROPERTY, "contains"),
            GraphEdge("cram", "coraplex", EdgeKind.PROPERTY, "contains"),
            GraphEdge("cram", "krrood", EdgeKind.PROPERTY, "contains"),
            GraphEdge("coraplex", "coraplex.plans", EdgeKind.PROPERTY, "contains"),
        ]
        import_edges = [e for e in payload.edges if e.label == "imports"]
        assert import_edges == [
            GraphEdge("coraplex", "krrood", EdgeKind.TYPE, "imports")
        ]

    def test_link_grounding_edge_present_branch(self, fixture_scene):
        """
        ``link()`` wires the anchor episode to ``coraplex.plans``, which exists as a
        node in the fixture architecture.
        """
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")
        assert (
            GraphEdge("transport_milk", "coraplex.plans", EdgeKind.TYPE, "planned by")
            in payload.edges
        )

    def test_link_grounding_edge_absent_branch(self, fixture_scene):
        """
        ``link()`` silently drops edges whose target isn't a node in this view — neither
        ``giskardpy.motion_statechart`` nor ``semantic_digital_twin`` exists in the
        fixture architecture, so no edge may target them.
        """
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")
        targets = {e.target for e in payload.edges}
        assert "giskardpy.motion_statechart" not in targets
        assert "semantic_digital_twin" not in targets

    def test_plan_tree_cluster(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")
        assert payload.details["plan"] == DetailEntry(
            "executed plan",
            NodeGroup.PLAN,
            [
                "the plan tree the demo actually executed",
                "4 nodes",
                "double-click to open",
            ],
        )
        plan_edges = [e for e in payload.edges if e.source == "plan"]
        assert plan_edges == [
            GraphEdge("plan", "pr2", EdgeKind.PROPERTY, "executed by"),
            GraphEdge("plan", "prepare", EdgeKind.TYPE, "spans"),
            GraphEdge("plan", "transport_milk", EdgeKind.TYPE, "spans"),
        ]

    def test_status_string_reports_derived_counts(self, fixture_scene):
        """
        The status line's numbers must track the live payload/knowledge base, not a
        second hardcoded copy of them.
        """
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")
        knowledge_base = EpisodeKnowledgeBase.of_active_scene()
        assert payload.status == (
            "EQL ready · %d graph nodes · %d joints · %d CRAM classes"
            % (
                len(payload.nodes),
                len(knowledge_base.joints),
                len(knowledge_base.classes),
            )
        )


# %% characterization: GraphPanelViews.of_active_scene().for_node() dispatch
class TestExpandNode:
    def test_robot_dispatches_to_urdf_view(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_node("pr2")
        assert payload.breadcrumb == "pr2 · URDF"
        ids = {n.id for n in payload.nodes}
        assert "urdf:base_link" in ids

    def test_plan_dispatches_to_plan_view(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_node("plan")
        assert payload.to_payload()["breadcrumb"] == "executed plan"
        assert len(payload.nodes) == 4
        assert len(payload.edges) == 3

    def test_package_dispatches_to_package_view(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_node("coraplex")
        assert {n.id for n in payload.nodes} == {"coraplex", "coraplex.plans"}
        assert payload.edges == [
            GraphEdge("coraplex", "coraplex.plans", EdgeKind.PROPERTY, "contains")
        ]

    def test_subpackage_dispatches_to_subpackage_view(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_node("coraplex.plans")
        assert {n.id for n in payload.nodes} == {
            "coraplex.plans",
            "coraplex.src.coraplex.plans.plan.Plan",
            "coraplex.src.coraplex.plans.typed_plan.TypedPlan",
        }

    def test_class_dispatches_to_class_view(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_node(
            "coraplex.src.coraplex.plans.plan.Plan"
        )
        assert payload.breadcrumb == "Plan"
        assert {n.id for n in payload.nodes} == {
            "coraplex.src.coraplex.plans.plan.Plan",
            "coraplex.src.coraplex.plans.typed_plan.TypedPlan",
        }

    def test_unknown_node_is_not_drillable(self, fixture_scene):
        assert GraphPanelViews.of_active_scene().for_node("does-not-exist") is None

    def test_class_view_resolves_an_internal_base(self, fixture_scene):
        """
        ``TypedPlan``'s base ``Plan`` is scanned from the same fixture repository, so it
        resolves to the real class node rather than an external stub.
        """
        payload = GraphPanelViews.of_active_scene().for_node(
            "coraplex.src.coraplex.plans.typed_plan.TypedPlan"
        )
        assert (
            GraphEdge(
                "coraplex.src.coraplex.plans.typed_plan.TypedPlan",
                "coraplex.src.coraplex.plans.plan.Plan",
                EdgeKind.TYPE,
                "inherits",
            )
            in payload.edges
        )
        assert (
            payload.details["coraplex.src.coraplex.plans.plan.Plan"].group
            == NodeGroup.PYTHON_CLASS
        )

    def test_class_view_falls_back_to_an_external_base(self, fixture_scene):
        """
        ``EqlError``'s base ``Exception`` is not defined anywhere in the scanned
        repository, so it renders as an external stub instead of a real class node.
        """
        payload = GraphPanelViews.of_active_scene().for_node(
            "krrood.src.krrood.errors.EqlError"
        )
        assert payload.details["external:Exception"] == DetailEntry(
            "Exception",
            NodeGroup.EXTERNAL_CLASS,
            ["external base class (outside the repo)"],
        )
        assert (
            GraphEdge(
                "krrood.src.krrood.errors.EqlError",
                "external:Exception",
                EdgeKind.TYPE,
                "inherits",
            )
            in payload.edges
        )

    def test_class_view_lists_repository_subclasses(self, fixture_scene):
        """
        ``Plan`` has no declared bases, but ``TypedPlan`` names it as a base — so
        ``Plan``'s inheritance view must list ``TypedPlan`` as a subclass.
        """
        payload = GraphPanelViews.of_active_scene().for_node(
            "coraplex.src.coraplex.plans.plan.Plan"
        )
        assert (
            GraphEdge(
                "coraplex.src.coraplex.plans.typed_plan.TypedPlan",
                "coraplex.src.coraplex.plans.plan.Plan",
                EdgeKind.TYPE,
                "inherits",
            )
            in payload.edges
        )

    def test_package_view_truncates_to_the_maximum_classes_shown(self, fixture_scene):
        knowledge_base = EpisodeKnowledgeBase.of_active_scene()
        synthetic_classes = [
            PythonClass(
                name="Synthetic%d" % index,
                package="synthetic_pkg",
                subpackage="synthetic_pkg",
                module="synthetic_pkg.synthetic%d" % index,
                bases=(),
                methods=index,
                docstring_summary="",
            )
            for index in range(SubgraphViewPayload.MAXIMUM_CLASSES_SHOWN + 1)
        ]
        knowledge_base.packages = knowledge_base.packages + [
            Package(name="synthetic_pkg", description="", module_count=0, class_count=0)
        ]
        knowledge_base.classes = knowledge_base.classes + synthetic_classes
        payload = GraphPanelViews.of_active_scene().for_node("synthetic_pkg")
        assert payload.details["synthetic_pkg"].lines[-1] == (
            "showing the %d largest of %d classes (by method count)"
            % (
                SubgraphViewPayload.MAXIMUM_CLASSES_SHOWN,
                SubgraphViewPayload.MAXIMUM_CLASSES_SHOWN + 1,
            )
        )

    def test_class_view_truncates_to_the_maximum_subclasses_shown(self, fixture_scene):
        knowledge_base = EpisodeKnowledgeBase.of_active_scene()
        base_class = PythonClass(
            name="SyntheticBase",
            package="synthetic_pkg",
            subpackage="synthetic_pkg",
            module="synthetic_pkg.base",
            bases=(),
            methods=0,
            docstring_summary="",
        )
        synthetic_subclasses = [
            PythonClass(
                name="SyntheticSubclass%d" % index,
                package="synthetic_pkg",
                subpackage="synthetic_pkg",
                module="synthetic_pkg.sub%d" % index,
                bases=("SyntheticBase",),
                methods=0,
                docstring_summary="",
            )
            for index in range(SubgraphViewPayload.MAXIMUM_SUBCLASSES_SHOWN + 1)
        ]
        knowledge_base.classes = (
            knowledge_base.classes + [base_class] + synthetic_subclasses
        )
        payload = GraphPanelViews.of_active_scene().for_node(
            "synthetic_pkg.base.SyntheticBase"
        )
        assert payload.details["synthetic_pkg.base.SyntheticBase"].lines[-1] == (
            "showing %d of %d subclasses"
            % (
                SubgraphViewPayload.MAXIMUM_SUBCLASSES_SHOWN,
                SubgraphViewPayload.MAXIMUM_SUBCLASSES_SHOWN + 1,
            )
        )


# %% smoke test: every generated preset must run without raising
class TestSceneSelection:
    """
    The viewer keeps several onboarded bundles open at once by naming one per request,
    so a named scene must build its own knowledge base rather than the active one's.
    """

    def _second_scene(self, data_directory) -> str:
        """
        Write a second bundle next to the fixture, with a differently named robot.

        :param data_directory: The fixture's data directory.
        """
        source = data_directory / "scenes" / "fixture"
        other = data_directory / "scenes" / "second"
        other.mkdir()
        scene = json.loads((source / "scene.json").read_text())
        scene["robot"]["name"] = "second_robot"
        (other / "scene.json").write_text(json.dumps(scene))
        (other / "trajectory.json").write_text((source / "trajectory.json").read_text())
        EpisodeKnowledgeBase.reset()
        return "second"

    def test_a_named_scene_builds_its_own_knowledge_base(self, fixture_scene):
        name = self._second_scene(fixture_scene)

        assert EpisodeKnowledgeBase.of_scene(name).robot.name == "second_robot"
        assert EpisodeKnowledgeBase.of_active_scene().robot.name == "pr2"

    def test_each_scene_is_built_once_and_kept(self, fixture_scene):
        name = self._second_scene(fixture_scene)

        assert EpisodeKnowledgeBase.of_scene(name) is EpisodeKnowledgeBase.of_scene(
            name
        )
        assert EpisodeKnowledgeBase.of_scene(name) is not (
            EpisodeKnowledgeBase.of_active_scene()
        )

    def test_the_views_of_a_named_scene_describe_that_scene(self, fixture_scene):
        name = self._second_scene(fixture_scene)

        payload = GraphPanelViews.of_scene(name).for_tab("knowledge").to_payload()

        assert any(node["id"] == "second_robot" for node in payload["nodes"])


class TestPresetSmoke:
    def test_every_preset_runs_and_returns_rows(self, fixture_scene):
        """
        Every preset ``Preset.of_scene()`` hands to the EQL panel must actually run.

        Replaces the module's former ``if __name__ == "__main__":`` smoke script, which
        logged OK/FAIL per preset instead of asserting anything.
        """
        for preset in Preset.of_scene():
            result = EqlSession.of_active_scene().run(preset.code)
            assert result.ok, "%s: %s" % (preset.text, result)
            assert result.count == len(result.rows)

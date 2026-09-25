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
from cramera.knowledge.eql_session import EqlSession  # noqa: E402
from cramera.knowledge.query_runner import RowRenderer  # noqa: E402
from cramera.knowledge.graph_payload import KnowledgeGraphPayload  # noqa: E402
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase  # noqa: E402
from cramera.knowledge.presets import (  # noqa: E402
    Preset,
    SCENE_PRESETS,
    PresetsPerType,
)
from cramera.knowledge.queryable_knowledge import QueryScope  # noqa: E402
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
from cramera.knowledge.recorded_statecharts import (  # noqa: E402
    RecordedStatecharts,
    STATECHART_FILE,
)
from cramera.knowledge.scene_bundle import SceneBundle  # noqa: E402

from cramera.generated_json import write_json_atomically  # noqa: E402

from .conftest import reset_knowledge_base_cache  # noqa: E402
from .test_recorded_statecharts import snapshot as chart_snapshot  # noqa: E402
from cramera.knowledge.subgraph import (  # noqa: E402
    DetailEntry,
    DuplicateNodeId,
    GraphEdge,
    SubgraphAccumulator,
)
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
            {"name": "milk", "kind": "object"},
            {"name": "place_area", "kind": "location"},
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

    def test_an_unrecorded_measurement_stays_unknown(self, fixture_scene):
        """
        A measurement the bundle never recorded must read as absent, not as zero.
        """
        milk = next(
            entry
            for entry in EpisodeKnowledgeBase.of_active_scene().objects
            if entry.name == "milk"
        )
        assert milk.height_metres is None


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


def recorded_statecharts() -> RecordedStatecharts:
    """
    The statecharts of a two-tick run: one motion, ticked to completion.
    """
    return RecordedStatecharts.of_snapshots(
        [chart_snapshot(), chart_snapshot(life_cycles=("DONE", "DONE"))]
    )


class TestViewPayloads:
    def test_knowledge_view(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")
        assert payload.ok
        ids = {n.id for n in payload.nodes}
        assert {"cram", "coraplex", "coraplex.plans"} <= ids
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

    def test_plan_view_reports_no_status(self, fixture_scene):
        """
        A replay's plan tab shows structure only.

        Which node was running when is a live statement, and a recording is scrubbed
        back and forth; a status shown at every played moment would be a claim about one
        of them. The bridge streams the statuses while a demo performs the plan.
        """
        payload = GraphPanelViews.of_active_scene().for_tab("plan")
        rendered = payload.to_payload()

        assert payload.ok and rendered["layout"] == "hier"
        assert rendered["statusLegend"] is False
        assert {node.status for node in payload.nodes} == {None}
        assert "status" not in rendered["nodes"][0]
        assert not [
            line
            for entry in payload.details.values()
            for line in entry.lines
            if line.startswith("status")
        ]
        assert len(payload.edges) == len(payload.nodes) - 1

    def test_plan_view_nodes_carry_tree_structure(self, fixture_scene):
        """
        The step list nests by ``parent`` and filters by ``kind``, so a recorded plan
        node must carry both — without them every node reads as a top-level step.
        """
        payload = GraphPanelViews.of_active_scene().for_tab("plan")
        by_label = {node["label"]: node for node in payload.to_payload()["nodes"]}
        root, action = by_label["SequentialNode"], by_label["Transport"]
        assert root["kind"] == "SequentialNode" and root["parent"] is None
        assert action["kind"] == "ActionNode" and action["parent"] == root["id"]
        assert by_label["ConditionNode"]["parent"] == action["id"]

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

    def test_the_chart_view_serves_what_a_recording_captured(self, fixture_scene):
        """
        A replay's statechart tab is filled from the bundle rather than from a bridge:

        the recording kept every statechart the run ticked (see
        :mod:`cramera.knowledge.recorded_statecharts`).
        """
        recorded = recorded_statecharts()
        write_json_atomically(
            fixture_scene / "scenes" / "fixture" / STATECHART_FILE,
            recorded.to_payload(),
        )
        EpisodeKnowledgeBase.reset()

        rendered = GraphPanelViews.of_active_scene().for_tab("chart").to_payload()

        assert rendered["recorded"] == recorded.to_payload()

    def test_the_chart_view_of_a_scene_without_a_recording_has_nothing(
        self, fixture_scene
    ):
        rendered = GraphPanelViews.of_active_scene().for_tab("chart").to_payload()

        assert "recorded" not in rendered

    def test_transform_view_is_live_only(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("transforms")
        rendered = payload.to_payload()
        assert payload.ok and rendered["nodes"] == []
        assert rendered["live"] == "transforms" and rendered["empty"]

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


# %% characterization: GraphPanelViews.of_active_scene().for_tab("knowledge") structure
# %% one node per id
class TestSubgraphNodeIds:
    """
    A node id addresses one node: the frontend builds its graph from a keyed data set,
    so a repeated id throws there and the whole panel renders nothing at all.
    """

    def test_a_repeated_id_is_refused(self):
        view = SubgraphAccumulator()
        view.add("plan", "executed plan", NodeGroup.PLAN, [])

        with pytest.raises(DuplicateNodeId):
            view.add("plan", "plan", NodeGroup.EVENT, [])

    def test_the_refusal_names_the_id(self):
        view = SubgraphAccumulator()
        view.add("plan", "executed plan", NodeGroup.PLAN, [])

        with pytest.raises(DuplicateNodeId, match="plan"):
            view.add("plan", "plan", NodeGroup.EVENT, [])


class TestGraphPayloadStructure:
    def test_the_overview_holds_only_the_architecture(self, fixture_scene):
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")

        assert {node.group for node in payload.nodes} == {
            NodeGroup.ROOT,
            NodeGroup.PACKAGE,
            NodeGroup.SUBPACKAGE,
        }

    def test_nothing_the_recording_itself_holds_is_drawn(self, fixture_scene):
        """
        The robot, its arms, the objects and the episodes each have a tab or a query of
        their own; in the architecture graph they were unconnected strays.
        """
        knowledge_base = EpisodeKnowledgeBase.of_active_scene()
        recorded = (
            {knowledge_base.robot.name}
            | {arm.name for arm in knowledge_base.arms}
            | {bench_object.name for bench_object in knowledge_base.objects}
            | {episode.name for episode in knowledge_base.episodes}
        )
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")

        assert recorded
        assert not recorded & {node.id for node in payload.nodes}

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

    def test_the_status_names_what_the_recording_is(self, fixture_scene):
        """
        The line above the questions says where an answer comes from -- the live bridge
        names the demo, a bundle names what it recorded: one robot, in one environment,
        doing one task.

        The scene's own name is in the picker beside it.
        """
        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")

        assert payload.status == "recorded · pr2"

    def test_the_status_names_the_task_a_recording_states(self, fixture_scene):
        scene_path = fixture_scene / "scenes" / "fixture" / "scene.json"
        scene = json.loads(scene_path.read_text())
        scene["task"] = "make breakfast"
        scene_path.write_text(json.dumps(scene))
        reset_knowledge_base_cache()

        payload = GraphPanelViews.of_active_scene().for_tab("knowledge")

        assert payload.status == "recorded · pr2 · make breakfast"


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

        assert EpisodeKnowledgeBase.of_scene(name).robot.name in payload["status"]


class TestScenePresets:
    """
    The EQL panel offers one fixed pair of questions, whatever a scene holds and
    whatever a bundle declares.
    """

    def test_exactly_the_offered_pair_is_handed_to_the_panel(self, fixture_scene):
        runner = EqlSession.of_active_scene().runner()

        assert Preset.of_scene() == [preset.worded(runner) for preset in SCENE_PRESETS]

    def test_a_bundle_cannot_declare_its_own(self, fixture_scene):
        """
        A demo's own questions range over variables only that demo offers; they reach
        the panel from the live bridge, not from the recorded bundle.
        """
        (fixture_scene / "scenes" / "fixture" / "presets.json").write_text(
            json.dumps(
                {
                    "presets": [
                        {
                            "text": "which shapes are inserted?",
                            "code": "an(entity(shape))",
                        }
                    ]
                }
            )
        )
        EpisodeKnowledgeBase.reset()

        assert [preset.text for preset in Preset.of_scene()] == [
            preset.text for preset in SCENE_PRESETS
        ]


class TestPresetWording:
    """
    Every preset carries its question read back as English, so the panel can show what
    is asked instead of EQL source.
    """

    def test_every_scene_preset_is_worded_by_the_scenes_own_runner(self, fixture_scene):
        runner = EqlSession.of_active_scene().runner()
        for preset in Preset.of_scene():
            assert preset.verbalization == runner.verbalize(preset.code), preset.text

    def test_an_offered_preset_carries_both_renderings(self, fixture_scene):
        preset = Preset.of_scene()[0]

        assert preset.verbalization is not None
        assert preset.verbalization.text
        assert "<span" in preset.verbalization.html

    def test_wording_returns_a_copy_and_leaves_the_original_untouched(
        self, fixture_scene
    ):
        preset = Preset("which robot is this?", "the(entity(robot))")
        worded = preset.worded(EqlSession.of_active_scene().runner())

        assert preset.verbalization is None
        assert worded.verbalization is not None
        assert (worded.text, worded.code, worded.requires_live, worded.scope) == (
            preset.text,
            preset.code,
            preset.requires_live,
            preset.scope,
        )


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


# %% one question per type a record can be
class TestPresetsPerType:
    """
    A question that names one type out of many is worth recognizing for every type,
    which is what :class:`PresetsPerType` writes out.
    """

    def test_a_camel_case_class_name_is_asked_for_in_plain_words(self):
        questions = PresetsPerType(
            class_suffix="Event",
            class_names=("PickUpEvent", "LossOfContainmentEvent"),
            code="an(entity(event).where(event.event_type == '%s'))",
        ).questions()

        assert [question.text for question in questions] == [
            "give me all pick up events",
            "give me all loss of containment events",
        ]

    def test_each_question_names_its_own_type_in_the_query(self):
        [question] = PresetsPerType(
            class_suffix="Action",
            class_names=("PickUpAction",),
            code="an(entity(action).where(action.action_type == '%s'))",
        ).questions()

        assert (
            question.code
            == "an(entity(action).where(action.action_type == 'PickUpAction'))"
        )

    def test_every_question_is_about_the_scope_the_family_declares(self):
        questions = PresetsPerType(
            class_suffix="Event",
            class_names=("PickUpEvent", "InsertionEvent"),
            code="an(entity(event).where(event.event_type == '%s'))",
            scope=QueryScope.DETECTED_EVENTS,
        ).questions()

        assert [question.scope for question in questions] == [
            QueryScope.DETECTED_EVENTS,
            QueryScope.DETECTED_EVENTS,
        ]

"""
Tests for the scene-driven knowledge base and its graph-panel payloads.
"""

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from cram_viz import knowledge  # noqa: E402  (importable once krrood is present)
from cram_viz.knowledge import knowledge_base  # noqa: E402
from cram_viz.knowledge.architecture_entities import (  # noqa: E402
    Package,
    PythonClass,
)
from cram_viz.knowledge.architecture_scan import ArchitectureScanner  # noqa: E402
from cram_viz.knowledge.enums import ArmSide, EdgeKind, NodeGroup  # noqa: E402
from cram_viz.knowledge.scene_bundle import SceneBundle  # noqa: E402
from cram_viz.knowledge.subgraph import DetailEntry, GraphEdge  # noqa: E402
from cram_viz.knowledge.views import plan_tree as plan_view  # noqa: E402


@pytest.fixture()
def fresh_knowledge_base(fixture_scene):
    knowledge.reset_knowledge_base()
    return knowledge.get_knowledge_base()


class TestEpisodeKnowledgeBase:
    def test_scene_entities(self, fresh_knowledge_base):
        assert [o.name for o in fresh_knowledge_base.objects] == ["milk", "place_area"]
        assert fresh_knowledge_base.robot.name == "pr2"
        assert [a.side for a in fresh_knowledge_base.arms] == ["left"]
        assert fresh_knowledge_base.arms[0].gripper.name == "left_gripper"

    def test_episodes_link_objects(self, fresh_knowledge_base):
        transport = next(
            e for e in fresh_knowledge_base.episodes if e.name == "transport_milk"
        )
        assert transport.picks is fresh_knowledge_base.objects[0]
        assert transport.places_at.name == "place_area"
        assert transport.performed_by.side == "left"

    def test_joint_motion_ranges(self, fresh_knowledge_base):
        torso = next(
            j for j in fresh_knowledge_base.joints if j.name == "torso_lift_joint"
        )
        assert torso.min_rad == 0.0 and torso.max_rad == 0.3

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
        result = ArchitectureScanner().scan()
        assert result.packages and all(
            isinstance(package, Package) for package in result.packages
        )
        assert result.classes and all(
            isinstance(python_class, PythonClass) for python_class in result.classes
        )
        plan_class = next(c for c in result.classes if c.name == "Plan")
        assert plan_class.subpackage == "coraplex.plans"

    def test_load_caches_the_scan_on_disk(self, fixture_scene):
        """
        ``load()`` must return the same entities as ``scan()``, from the cache on a
        second call.
        """
        scanner = ArchitectureScanner()
        scanned = scanner.scan()
        loaded_once = scanner.load()
        loaded_again = scanner.load()
        assert {p.name for p in loaded_once.packages} == {
            p.name for p in scanned.packages
        }
        assert loaded_once.classes == loaded_again.classes


class TestArmSideInference:
    def test_an_arm_whose_name_encodes_no_side_is_unknown(
        self, fixture_scene, monkeypatch
    ):
        """
        An arm part name that names neither `left` nor `right` cannot be assigned a side
        by name inspection, and must not silently masquerade as one.
        """
        bundle = knowledge.load_scene()
        scene, trajectory = bundle.scene, bundle.trajectory
        scene["robot"]["parts"]["center_arm"] = ["center_link"]
        monkeypatch.setattr(
            knowledge_base, "load_scene", lambda: SceneBundle(scene, trajectory)
        )
        knowledge.reset_knowledge_base()
        center_arm = next(
            arm
            for arm in knowledge.get_knowledge_base().arms
            if arm.name == "center_arm"
        )
        assert center_arm.side == ArmSide.UNKNOWN


class TestQueries:
    def test_entity_query(self, fixture_scene):
        result = knowledge.run_query("the(entity(obj).where(obj.name == 'milk'))")
        assert result.ok and result.count == 1
        assert result.rows[0]["__entity__"] == "milk"
        assert "milk" in result.highlight

    def test_an_unknown_name_raises(self, fixture_scene):
        """
        A query naming something the namespace does not define must raise.

        The server turns this into a JSON error payload; the knowledge base itself does
        not swallow it.
        """
        with pytest.raises(NameError):
            knowledge.run_query("this is not python")

    def test_a_syntactically_invalid_query_raises(self, fixture_scene):
        with pytest.raises(SyntaxError):
            knowledge.run_query("definitely not python (((")


class TestRecordedMeasurements:
    def test_an_unrecorded_height_stays_unknown(self, fresh_knowledge_base):
        """
        The fixture bundle records no object height, so none may be invented.
        """
        milk = next(
            entry for entry in fresh_knowledge_base.objects if entry.name == "milk"
        )
        assert milk.height_m is None

    def test_an_unrecorded_gripper_opening_stays_unknown(self, fresh_knowledge_base):
        assert fresh_knowledge_base.arms[0].gripper.opening_m is None

    def test_a_recorded_height_is_used(self, fixture_scene, monkeypatch):
        """
        A bundle that reports a height must be taken at its word.
        """
        bundle = knowledge.load_scene()
        scene, trajectory = bundle.scene, bundle.trajectory
        scene["objects"][0]["height"] = 0.23
        monkeypatch.setattr(
            knowledge_base, "load_scene", lambda: SceneBundle(scene, trajectory)
        )
        knowledge.reset_knowledge_base()
        milk = next(
            entry
            for entry in knowledge.get_knowledge_base().objects
            if entry.name == "milk"
        )
        assert milk.height_m == 0.23

    def test_unknown_measurements_are_left_out_of_the_graph(self, fixture_scene):
        """
        A tooltip must not show a height the bundle never recorded.
        """
        payload = knowledge.view_payload("knowledge")
        milk = payload.details["milk"]
        assert not any(line.startswith("height:") for line in milk.lines)


class TestActionLabelShortening:
    def test_action_suffix_is_dropped(self):
        assert knowledge.shorten_action_label("TransportAction") == "Transport"

    def test_the_word_action_inside_a_label_is_kept(self):
        assert knowledge.shorten_action_label("ActionNode") == "ActionNode"

    def test_only_the_trailing_occurrence_is_dropped(self):
        assert (
            knowledge.shorten_action_label("ActionSequenceAction") == "ActionSequence"
        )

    def test_a_label_that_is_only_the_suffix_is_kept(self):
        assert knowledge.shorten_action_label("Action") == "Action"


class TestViewPayloads:
    def test_knowledge_view(self, fixture_scene):
        payload = knowledge.view_payload("knowledge")
        assert payload.ok
        ids = {n.id for n in payload.nodes}
        assert {"pr2", "milk", "transport_milk", "plan"} <= ids
        assert payload.presets

    def test_kinematics_view(self, fixture_scene):
        payload = knowledge.view_payload("kinematics")
        assert payload.ok
        ids = {n.id for n in payload.nodes}
        assert "urdf:base_link" in ids and "urdf:l_gripper_link" in ids
        # fixed joints render dashed ('type'), movable solid ('prop')
        kinds = {e.label.split(" ")[0]: e.kind for e in payload.edges}
        assert kinds["torso_lift_joint"] == EdgeKind.PROP
        assert kinds["l_gripper_joint"] == EdgeKind.TYPE

    def test_kinematics_counts_every_movable_joint(self, fixture_scene):
        """
        The movable-joint tally must match the joints drawn as movable.

        The fixture's ``torso_lift_joint`` is prismatic: movable, but not revolute.
        """
        payload = knowledge.view_payload("kinematics")
        movable_edges = [edge for edge in payload.edges if edge.kind == EdgeKind.PROP]
        root_lines = payload.details["urdf:base_link"].lines
        summary = next(line for line in root_lines if "movable" in line)
        assert summary.endswith("(%d movable)" % len(movable_edges))

    def test_plan_view_carries_status(self, fixture_scene):
        payload = knowledge.view_payload("plan")
        rendered = payload.to_payload()
        assert payload.ok and rendered["layout"] == "hier"
        assert rendered["live"] == "plan" and rendered["statusLegend"]
        by_label = {n.label: n for n in payload.nodes}
        assert by_label["SequentialNode"].status == "SUCCEEDED"
        # recorded inner nodes stay CREATED (only the root is performed)
        assert by_label["Transport"].status == "CREATED"
        assert len(payload.edges) == len(payload.nodes) - 1

    def test_plan_view_legend(self, fixture_scene):
        payload = knowledge.view_payload("plan")
        expected = [
            {"group": entry.group, "label": entry.label}
            for entry in plan_view.PLAN_LEGEND
        ]
        assert payload.to_payload()["legend"] == expected

    def test_chart_view_is_live_only(self, fixture_scene):
        payload = knowledge.view_payload("chart")
        rendered = payload.to_payload()
        assert payload.ok and rendered["nodes"] == []
        assert rendered["live"] == "chart" and rendered["empty"]

    def test_unknown_view(self, fixture_scene):
        payload = knowledge.view_payload("bogus")
        assert not payload.ok


# %% BUG-1 -- attach/detach plan-node grouping
class TestPlanGroups:
    def test_attach_node_renders_in_the_object_group(self, fixture_scene, monkeypatch):
        """
        Coraplex's real class is ``AttachNode``, not ``AttachmentNode``.
        """
        bundle = knowledge.load_scene()
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
            plan_view, "load_scene", lambda: SceneBundle(scene, trajectory)
        )
        knowledge.reset_knowledge_base()
        node = next(
            n for n in knowledge.view_payload("plan").nodes if n.label == "AttachNode"
        )
        assert node.group == NodeGroup.OBJECT

    def test_detach_node_renders_in_the_object_group(self, fixture_scene, monkeypatch):
        """
        Coraplex's real class is ``DetachNode``, not ``DetachmentNode``.
        """
        bundle = knowledge.load_scene()
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
            plan_view, "load_scene", lambda: SceneBundle(scene, trajectory)
        )
        knowledge.reset_knowledge_base()
        node = next(
            n for n in knowledge.view_payload("plan").nodes if n.label == "DetachNode"
        )
        assert node.group == NodeGroup.OBJECT


# %% BUG-2 -- EQL preset splicing
class TestPresetSafety:
    def test_an_apostrophe_in_an_object_name_does_not_break_its_preset(
        self, fixture_scene, monkeypatch
    ):
        """
        ``get_presets()`` must escape object names, not splice them raw into EQL source.
        """
        bundle = knowledge.load_scene()
        scene, trajectory = bundle.scene, bundle.trajectory
        scene["objects"][0]["id"] = "o'brien"
        scene["segments"][1]["picks"] = "o'brien"
        monkeypatch.setattr(
            knowledge_base, "load_scene", lambda: SceneBundle(scene, trajectory)
        )
        knowledge.reset_knowledge_base()
        preset = next(p for p in knowledge.get_presets() if "obj.name" in p.code)
        result = knowledge.run_query(preset.code)
        assert result.ok and result.rows[0]["__entity__"] == "o'brien"

    def test_an_apostrophe_in_an_episode_name_does_not_break_its_presets(
        self, fixture_scene, monkeypatch
    ):
        """
        Covers both the ``places_at`` and ``performed_by`` presets, which splice the
        same episode name.
        """
        bundle = knowledge.load_scene()
        scene, trajectory = bundle.scene, bundle.trajectory
        scene["segments"][1]["step"] = "transport_o'brien"
        monkeypatch.setattr(
            knowledge_base, "load_scene", lambda: SceneBundle(scene, trajectory)
        )
        knowledge.reset_knowledge_base()
        for preset in knowledge.get_presets():
            assert knowledge.run_query(preset.code).ok


# %% characterization: graph_payload() structure
class TestGraphPayloadStructure:
    def test_robot_arm_gripper_chain(self, fixture_scene):
        payload = knowledge.graph_payload()
        by_id = {n.id: n for n in payload.nodes}
        assert by_id["pr2"].label == "pr2" and by_id["pr2"].group == NodeGroup.ROBOT
        assert by_id["left_arm"].label == "left arm"
        assert by_id["left_gripper"].label == "left gripper"
        chain_edges = [e for e in payload.edges if e.label == "has part"]
        assert chain_edges == [
            GraphEdge("pr2", "left_arm", EdgeKind.PROP, "has part"),
            GraphEdge("left_arm", "left_gripper", EdgeKind.PROP, "has part"),
        ]
        assert payload.details["pr2"] == DetailEntry(
            "pr2",
            NodeGroup.ROBOT,
            ["a Robot", "1 arm", "double-click: full URDF tree"],
        )

    def test_episode_chain(self, fixture_scene):
        payload = knowledge.graph_payload()
        episode_edges = [
            e
            for e in payload.edges
            if e.label in ("precedes", "performed by", "picks", "places at")
        ]
        assert episode_edges == [
            GraphEdge("prepare", "transport_milk", EdgeKind.TYPE, "precedes"),
            GraphEdge("transport_milk", "pr2", EdgeKind.PROP, "performed by"),
            GraphEdge("transport_milk", "milk", EdgeKind.PROP, "picks"),
            GraphEdge("transport_milk", "place_area", EdgeKind.PROP, "places at"),
        ]

    def test_object_detail_lines(self, fixture_scene):
        payload = knowledge.graph_payload()
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
        payload = knowledge.graph_payload()
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
            NodeGroup.CONCEPT,
            [
                "a Package",
                "the plan executive: designators, plans, locations",
                "2 modules · 2 classes",
                "double-click to open",
            ],
        )
        assert payload.details["coraplex.plans"] == DetailEntry(
            "plans",
            NodeGroup.KLASS,
            [
                "a SubPackage of coraplex",
                "2 modules · 2 classes",
                "double-click to open",
            ],
        )
        contains_edges = [e for e in payload.edges if e.label == "contains"]
        assert contains_edges == [
            GraphEdge("cram", "root", EdgeKind.PROP, "contains"),
            GraphEdge("cram", "coraplex", EdgeKind.PROP, "contains"),
            GraphEdge("cram", "krrood", EdgeKind.PROP, "contains"),
            GraphEdge("coraplex", "coraplex.plans", EdgeKind.PROP, "contains"),
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
        payload = knowledge.graph_payload()
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
        payload = knowledge.graph_payload()
        targets = {e.target for e in payload.edges}
        assert "giskardpy.motion_statechart" not in targets
        assert "semantic_digital_twin" not in targets

    def test_plan_tree_cluster(self, fixture_scene):
        payload = knowledge.graph_payload()
        assert payload.details["plan"] == DetailEntry(
            "executed plan",
            NodeGroup.GOAL,
            [
                "the plan tree the demo actually executed",
                "4 nodes",
                "double-click to open",
            ],
        )
        plan_edges = [e for e in payload.edges if e.source == "plan"]
        assert plan_edges == [
            GraphEdge("plan", "pr2", EdgeKind.PROP, "executed by"),
            GraphEdge("plan", "prepare", EdgeKind.TYPE, "spans"),
            GraphEdge("plan", "transport_milk", EdgeKind.TYPE, "spans"),
        ]

    def test_status_string_reports_derived_counts(self, fixture_scene):
        """
        The status line's numbers must track the live payload/knowledge base, not a
        second hardcoded copy of them.
        """
        payload = knowledge.graph_payload()
        knowledge_base = knowledge.get_knowledge_base()
        assert payload.status == (
            "EQL ready · %d graph nodes · %d joints · %d CRAM classes"
            % (
                len(payload.nodes),
                len(knowledge_base.joints),
                len(knowledge_base.classes),
            )
        )


# %% characterization: expand_node() dispatch
class TestExpandNode:
    def test_robot_dispatches_to_urdf_view(self, fixture_scene):
        payload = knowledge.expand_node("pr2")
        assert payload.crumb == "pr2 · URDF"
        ids = {n.id for n in payload.nodes}
        assert "urdf:base_link" in ids

    def test_plan_dispatches_to_plan_view(self, fixture_scene):
        payload = knowledge.expand_node("plan")
        assert payload.to_payload()["crumb"] == "executed plan"
        assert len(payload.nodes) == 4
        assert len(payload.edges) == 3

    def test_package_dispatches_to_package_view(self, fixture_scene):
        payload = knowledge.expand_node("coraplex")
        assert {n.id for n in payload.nodes} == {"coraplex", "coraplex.plans"}
        assert payload.edges == [
            GraphEdge("coraplex", "coraplex.plans", EdgeKind.PROP, "contains")
        ]

    def test_subpackage_dispatches_to_subpackage_view(self, fixture_scene):
        payload = knowledge.expand_node("coraplex.plans")
        assert {n.id for n in payload.nodes} == {
            "coraplex.plans",
            "coraplex.src.coraplex.plans.plan.Plan",
            "coraplex.src.coraplex.plans.typed_plan.TypedPlan",
        }

    def test_class_dispatches_to_class_view(self, fixture_scene):
        payload = knowledge.expand_node("coraplex.src.coraplex.plans.plan.Plan")
        assert payload.crumb == "Plan"
        assert {n.id for n in payload.nodes} == {
            "coraplex.src.coraplex.plans.plan.Plan",
            "coraplex.src.coraplex.plans.typed_plan.TypedPlan",
        }

    def test_unknown_node_is_not_drillable(self, fixture_scene):
        assert knowledge.expand_node("does-not-exist") is None

    def test_class_view_resolves_an_internal_base(self, fixture_scene):
        """
        ``TypedPlan``'s base ``Plan`` is scanned from the same fixture repository, so it
        resolves to the real class node rather than an external stub.
        """
        payload = knowledge.expand_node(
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
            == NodeGroup.PYCLASS
        )

    def test_class_view_falls_back_to_an_external_base(self, fixture_scene):
        """
        ``EqlError``'s base ``Exception`` is not defined anywhere in the scanned
        repository, so it renders as an external stub instead of a real class node.
        """
        payload = knowledge.expand_node("krrood.src.krrood.errors.EqlError")
        assert payload.details["ext:Exception"] == DetailEntry(
            "Exception",
            NodeGroup.UPPER,
            ["external base class (outside the repo)"],
        )
        assert (
            GraphEdge(
                "krrood.src.krrood.errors.EqlError",
                "ext:Exception",
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
        payload = knowledge.expand_node("coraplex.src.coraplex.plans.plan.Plan")
        assert (
            GraphEdge(
                "coraplex.src.coraplex.plans.typed_plan.TypedPlan",
                "coraplex.src.coraplex.plans.plan.Plan",
                EdgeKind.TYPE,
                "inherits",
            )
            in payload.edges
        )

    def test_package_view_truncates_to_class_cap(self, fixture_scene):
        knowledge_base = knowledge.get_knowledge_base()
        synthetic_classes = [
            knowledge.PythonClass(
                name="Synthetic%d" % index,
                package="synthetic_pkg",
                subpackage="synthetic_pkg",
                module="synthetic_pkg.synthetic%d" % index,
                bases=(),
                methods=index,
                doc="",
            )
            for index in range(knowledge.CLASS_CAP + 1)
        ]
        knowledge_base.packages = knowledge_base.packages + [
            knowledge.Package(
                name="synthetic_pkg", description="", module_count=0, class_count=0
            )
        ]
        knowledge_base.classes = knowledge_base.classes + synthetic_classes
        payload = knowledge.expand_node("synthetic_pkg")
        assert payload.details["synthetic_pkg"].lines[-1] == (
            "showing the %d largest of %d classes (by method count)"
            % (knowledge.CLASS_CAP, knowledge.CLASS_CAP + 1)
        )

    def test_class_view_truncates_to_subclass_cap(self, fixture_scene):
        knowledge_base = knowledge.get_knowledge_base()
        base_class = knowledge.PythonClass(
            name="SyntheticBase",
            package="synthetic_pkg",
            subpackage="synthetic_pkg",
            module="synthetic_pkg.base",
            bases=(),
            methods=0,
            doc="",
        )
        synthetic_subclasses = [
            knowledge.PythonClass(
                name="SyntheticSubclass%d" % index,
                package="synthetic_pkg",
                subpackage="synthetic_pkg",
                module="synthetic_pkg.sub%d" % index,
                bases=("SyntheticBase",),
                methods=0,
                doc="",
            )
            for index in range(knowledge.SUBCLASS_CAP + 1)
        ]
        knowledge_base.classes = (
            knowledge_base.classes + [base_class] + synthetic_subclasses
        )
        payload = knowledge.expand_node("synthetic_pkg.base.SyntheticBase")
        assert payload.details["synthetic_pkg.base.SyntheticBase"].lines[-1] == (
            "showing %d of %d subclasses"
            % (knowledge.SUBCLASS_CAP, knowledge.SUBCLASS_CAP + 1)
        )


# %% smoke test: every generated preset must run without raising
class TestPresetSmoke:
    def test_every_preset_runs_and_returns_rows(self, fixture_scene):
        """
        Every preset ``get_presets()`` hands to the EQL panel must actually run.

        Replaces the module's former ``if __name__ == "__main__":`` smoke script, which
        logged OK/FAIL per preset instead of asserting anything.
        """
        for preset in knowledge.get_presets():
            result = knowledge.run_query(preset.code)
            assert result.ok, "%s: %s" % (preset.text, result)
            assert result.count == len(result.rows)

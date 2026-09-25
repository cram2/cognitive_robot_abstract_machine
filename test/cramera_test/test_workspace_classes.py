"""
Tests for making the workspace's own classes nameable in an EQL query.
"""

from __future__ import annotations

import pytest

from cramera.knowledge.architecture_entities import PythonClass
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.query_runner import EqlQueryRunner
from cramera.knowledge.workspace_classes import (
    WorkspaceClassIndex,
    WorkspaceClassNamespace,
    WorkspacePackage,
)


def scanned_class(name: str, package: str, module: str) -> PythonClass:
    """
    One class as the architecture scan reports it.

    :param name: The class's own name.
    :param package: Top-level package it was scanned under.
    :param module: Repository-relative module path the scan recorded.
    """
    return PythonClass(
        name=name,
        package=package,
        subpackage=package,
        module=module,
        bases=(),
        methods=0,
        docstring_summary=f"{name} of {package}.",
    )


BODY = scanned_class(
    "Body",
    "semantic_digital_twin",
    "semantic_digital_twin.src.semantic_digital_twin.world_description.world_entity",
)
"""
A class a query should be able to name, in a module that really is importable.
"""


# %% which scanned classes a query may name


class TestWorkspaceClassIndex:
    """
    Which of the scanned classes end up nameable, and under which module.
    """

    def test_a_scanned_class_is_indexed_under_its_importable_module(self):
        index = WorkspaceClassIndex.of_scanned_classes([BODY])

        candidate = index.candidates("Body")[0]

        assert candidate.module == (
            "semantic_digital_twin.world_description.world_entity"
        )
        assert candidate.package is WorkspacePackage.SEMANTIC_DIGITAL_TWIN
        assert candidate.docstring_summary == BODY.docstring_summary

    def test_a_class_outside_a_source_tree_is_not_nameable(self):
        demo_class = scanned_class(
            "DemoRobot", "coraplex", "coraplex.demos.coraplex_world_demo.demo"
        )

        index = WorkspaceClassIndex.of_scanned_classes([demo_class])

        assert index.candidates("DemoRobot") == ()

    def test_a_class_of_a_package_no_query_names_is_not_nameable(self):
        test_class = scanned_class("TestHelper", "test", "test.src.test.helper")

        index = WorkspaceClassIndex.of_scanned_classes([test_class])

        assert index.candidates("TestHelper") == ()

    @pytest.mark.parametrize(
        "generated_name", ["ColorDAO", "PoseDAO_poses_association"]
    )
    def test_a_generated_orm_class_is_not_nameable(self, generated_name: str):
        generated = scanned_class(
            generated_name, "semantic_digital_twin", "semantic_digital_twin.src.orm"
        )

        index = WorkspaceClassIndex.of_scanned_classes([generated])

        assert index.candidates(generated_name) == ()

    def test_a_name_in_several_packages_is_won_by_the_first_declared_one(self):
        index = WorkspaceClassIndex.of_scanned_classes(
            [
                scanned_class("Filter", "robokudo", "robokudo.src.robokudo.filter"),
                scanned_class("Filter", "coraplex", "coraplex.src.coraplex.filter"),
            ]
        )

        packages = [candidate.package for candidate in index.candidates("Filter")]

        assert packages == [WorkspacePackage.CORAPLEX, WorkspacePackage.ROBOKUDO]

    def test_the_indexed_names_are_reported_in_alphabetical_order(self):
        index = WorkspaceClassIndex.of_scanned_classes(
            [BODY, scanned_class("Arm", "coraplex", "coraplex.src.coraplex.arm")]
        )

        assert index.names() == ["Arm", "Body"]

    def test_resolving_a_name_returns_the_class_itself(self):
        from semantic_digital_twin.world_description.world_entity import Body

        index = WorkspaceClassIndex.of_scanned_classes([BODY])

        assert index.resolve("Body") is Body


# %% the namespace a query is evaluated in


class TestWorkspaceClassNamespace:
    """
    How a name a query uses reaches the class the index knows about.
    """

    def test_a_class_the_query_names_is_resolved_on_first_use(self):
        from semantic_digital_twin.world_description.world_entity import Body

        namespace = WorkspaceClassNamespace(
            index=WorkspaceClassIndex.of_scanned_classes([BODY])
        )

        assert namespace["Body"] is Body

    def test_a_resolved_class_is_kept_for_the_next_use(self):
        namespace = WorkspaceClassNamespace(
            index=WorkspaceClassIndex.of_scanned_classes([BODY])
        )
        namespace["Body"]

        assert dict.__contains__(namespace, "Body")

    def test_a_seeded_name_wins_over_the_index(self):
        namespace = WorkspaceClassNamespace(
            index=WorkspaceClassIndex.of_scanned_classes([BODY])
        )
        namespace["Body"] = "the seeded one"

        assert namespace["Body"] == "the seeded one"

    def test_an_unknown_name_is_missing_from_the_namespace(self):
        namespace = WorkspaceClassNamespace(
            index=WorkspaceClassIndex.of_scanned_classes([BODY])
        )

        with pytest.raises(KeyError):
            namespace["NoSuchClass"]


# %% queries naming a workspace class


class TestQueryingWorkspaceClasses:
    """
    What a query can name once the index backs the runner's namespace.
    """

    def test_a_query_may_name_a_class_of_the_workspace(self):
        from semantic_digital_twin.world_description.world_entity import Body

        runner = EqlQueryRunner(
            domains=[], class_index=WorkspaceClassIndex.of_scanned_classes([BODY])
        )

        assert runner.run("Body").rows == [{"value": repr(Body)}]

    def test_a_query_naming_nothing_known_still_fails_as_an_unknown_name(self):
        runner = EqlQueryRunner(
            domains=[], class_index=WorkspaceClassIndex.of_scanned_classes([BODY])
        )

        with pytest.raises(NameError):
            runner.run("NoSuchClass")

    def test_a_ready_made_variable_is_not_shadowed_by_a_workspace_class(self):
        from cramera.knowledge.entities import BenchObject

        index = WorkspaceClassIndex.of_scanned_classes(
            [
                scanned_class(
                    "BenchObject", "coraplex", "coraplex.src.coraplex.bench_object"
                )
            ]
        )
        runner = EqlQueryRunner(
            domains=[QueryDomain("scene_object", BenchObject, [])],
            class_index=index,
        )

        assert runner.namespace()["BenchObject"] is BenchObject


# %% the index of a whole repository


class TestRepositoryClassIndex:
    """
    How the index of a scanned repository is built and kept.
    """

    def test_the_index_of_one_root_is_built_once(self, fixture_scene):
        first = WorkspaceClassIndex.of_repository()

        assert WorkspaceClassIndex.of_repository() is first

    def test_the_scanned_architecture_decides_what_is_nameable(self, fixture_scene):
        index = WorkspaceClassIndex.of_repository()

        assert index.candidates("Plan")[0].module == "coraplex.plans.plan"

    def test_another_root_gets_an_index_of_its_own(
        self, fixture_scene, tmp_path, monkeypatch
    ):
        scanned = WorkspaceClassIndex.of_repository()
        monkeypatch.setenv("CRAMERA_ARCHITECTURE", str(tmp_path / "elsewhere"))

        assert WorkspaceClassIndex.of_repository() is not scanned
        assert WorkspaceClassIndex.of_repository().names() == []

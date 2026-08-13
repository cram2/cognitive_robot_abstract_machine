"""
Tests for the MJCF bundler.

:func:`bundle_mjcf` builds a URDF from a parsed
:class:`~semantic_digital_twin.world.World`, exactly like
:func:`cram_viz.onboard.bundle_gazebo.bundle_gazebo_world` does for SDF, so these tests
exercise the same kinds of cases against the fixtures the semantic_digital_twin MJCF
adapter tests already use: a purely fixed kinematic tree, slide/hinge joints (mapped to
prismatic/revolute/continuous), mesh geometry, and a missing source.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("mujoco")

from cram_viz.onboard import bundle_mjcf as bundler
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.urdf import URDFParser

RESOURCES_DIRECTORY = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "mjcf",
    )
)


@dataclass
class MjcfFixturePaths:
    """
    The paths of the MJCF files these tests bundle.
    """

    table: str
    kitchen_small: str
    jeroen_cups: str


@pytest.fixture
def mjcf_paths() -> MjcfFixturePaths:
    return MjcfFixturePaths(
        table=os.path.join(RESOURCES_DIRECTORY, "table.xml"),
        kitchen_small=os.path.join(RESOURCES_DIRECTORY, "kitchen-small.xml"),
        jeroen_cups=os.path.join(RESOURCES_DIRECTORY, "jeroen_cups.xml"),
    )


def joint_by_name(urdf_path: str, name: str) -> ElementTree.Element:
    """
    :param urdf_path: Path of a bundled URDF.
    :param name: Name of the joint to find.
    :return: The ``joint`` element with that name.
    """
    root = ElementTree.parse(urdf_path).getroot()
    return next(joint for joint in root.findall("joint") if joint.get("name") == name)


def joint_types(urdf_path: str) -> list:
    """
    :param urdf_path: Path of a bundled URDF.
    :return: The ``type`` attribute of every ``joint`` element, in document order.
    """
    root = ElementTree.parse(urdf_path).getroot()
    return [joint.get("type") for joint in root.findall("joint")]


# %% a purely fixed kinematic tree
class TestFixedKinematicTree:
    def test_every_body_becomes_a_link_and_every_connection_a_fixed_joint(
        self, mjcf_paths, tmp_path
    ):
        world = MJCFParser(mjcf_paths.table).parse()
        report = bundler.bundle_mjcf(mjcf_paths.table, "table", str(tmp_path))
        assert report["links"] == [
            str(body.name) for body in world.bodies_topologically_sorted
        ]
        assert len(report["joints"]) == len(report["links"]) - 1
        assert report["movable_joints"] == []
        assert report["meshes_copied"] == 0
        assert report["missing"] == []
        assert set(joint_types(report["urdf"])) == {"fixed"}


# %% joints
class TestJointTypes:
    def test_a_slide_joint_becomes_prismatic_with_its_limit_and_axis(
        self, mjcf_paths, tmp_path
    ):
        report = bundler.bundle_mjcf(mjcf_paths.kitchen_small, "kitchen", str(tmp_path))
        joint = joint_by_name(
            report["urdf"], "oven_area_area_middle_upper_drawer_main_joint"
        )
        assert joint.get("type") == "prismatic"
        assert joint.find("axis").get("xyz") == "1.0 0.0 0.0"
        limit = joint.find("limit")
        assert float(limit.get("lower")) == pytest.approx(0.0)
        assert float(limit.get("upper")) == pytest.approx(0.47999998927116394, abs=1e-6)

    def test_a_hinge_joint_with_a_real_range_becomes_revolute(
        self, mjcf_paths, tmp_path
    ):
        report = bundler.bundle_mjcf(mjcf_paths.kitchen_small, "kitchen", str(tmp_path))
        joint = joint_by_name(report["urdf"], "iai_fridge_door_joint")
        assert joint.get("type") == "revolute"
        assert joint.find("axis").get("xyz") == "0.0 0.0 1.0"
        limit = joint.find("limit")
        assert float(limit.get("lower")) == pytest.approx(0.0)
        assert float(limit.get("upper")) == pytest.approx(1.5707963705062866, abs=1e-6)

    def test_a_hinge_joint_with_a_degenerate_zero_range_becomes_continuous(
        self, mjcf_paths, tmp_path
    ):
        """
        MJCFParser treats a ``range="0.0 0.0"`` hinge as declaring no limits at all (see
        :meth:`MJCFParser.parse_dof`), leaving its ``RevoluteConnection`` without
        position limits, exactly like GazeboParser's "spin" joints in the Gazebo bundler
        tests.
        """
        report = bundler.bundle_mjcf(mjcf_paths.kitchen_small, "kitchen", str(tmp_path))
        joint = joint_by_name(report["urdf"], "oven_area_oven_door_joint")
        assert joint.get("type") == "continuous"
        assert joint.find("limit") is None


# %% forward-kinematics round trip
class TestForwardKinematicsRoundTrip:
    def test_bundled_bodies_keep_their_global_pose_after_reparsing_as_urdf(
        self, mjcf_paths, tmp_path
    ):
        """
        Bundling re-derives a joint's ``<origin>``/``<axis>`` from the connection's
        folded ``origin``/``axis`` rather than copying MJCF's own body/joint pose split,
        so re-parsing the bundled URDF must place every body at the exact same global
        pose the original MJCF world computed.
        """
        world_mjcf = MJCFParser(mjcf_paths.kitchen_small).parse()
        report = bundler.bundle_mjcf(mjcf_paths.kitchen_small, "kitchen", str(tmp_path))
        world_urdf = URDFParser.from_file(report["urdf"]).parse()

        for body_name in [
            "oven_area_area_middle_upper_drawer_main",
            "iai_fridge_door",
        ]:
            mjcf_body = world_mjcf.get_body_by_name(body_name)
            urdf_body = world_urdf.get_body_by_name(body_name)
            expected = world_mjcf.compute_forward_kinematics_np(
                world_mjcf.root, mjcf_body
            )
            actual = world_urdf.compute_forward_kinematics_np(
                world_urdf.root, urdf_body
            )
            assert np.allclose(expected, actual, atol=1e-6)


# %% meshes
class TestMeshBundling:
    def test_differently_scaled_reuses_of_one_mesh_file_are_copied_once(
        self, mjcf_paths, tmp_path
    ):
        report = bundler.bundle_mjcf(mjcf_paths.jeroen_cups, "cups", str(tmp_path))
        assert report["meshes_copied"] == 1
        assert report["missing"] == []
        assert ".stl" in report["mesh_exts"]
        bundled_mesh = tmp_path / "meshes" / "mjcf" / "stl" / "jeroen_cup.stl"
        assert bundled_mesh.is_file()

    def test_freejoint_bodies_float_and_mocap_bodies_are_fixed(
        self, mjcf_paths, tmp_path
    ):
        report = bundler.bundle_mjcf(mjcf_paths.jeroen_cups, "cups", str(tmp_path))
        assert sorted(joint_types(report["urdf"])) == [
            "fixed",
            "fixed",
            "fixed",
            "floating",
            "floating",
            "floating",
        ]

    def test_a_missing_source_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bundler.bundle_mjcf(str(tmp_path / "gone.xml"), "demo", str(tmp_path))

"""
World serialization and self-contained geometry asset bundling.
"""

from __future__ import annotations
import xml.etree.ElementTree as ElementTree
from pathlib import Path
import pytest
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from cramera.onboard import bundle_urdf as bundler
from cramera.onboard.world_to_urdf import UrdfDocument

ONE_MESH_URDF_TEXT = (
    '<robot name="demo">\n'
    '  <link name="base_link"/>\n'
    '  <link name="cup_link">\n'
    "    <visual><geometry>\n"
    '      <mesh filename="meshes/cup.stl"/>\n'
    "    </geometry></visual>\n"
    "  </link>\n"
    '  <joint name="cup_joint" type="fixed">\n'
    '    <parent link="base_link"/><child link="cup_link"/>\n'
    "  </joint>\n"
    "</robot>\n"
)


# %% TestSerializeUnclaimedBodies


class TestSerializeUnclaimedBodies:
    """
    A world built in code -- bodies constructed directly instead of parsed out of a
    URDF, MJCF or SDF file -- leaves no source to bundle, so the bodies themselves have
    to become a model.

    Only the bodies no parsed model already claims are serialized, which
    is what makes a root of their own necessary: their parent may be a body this document
    does not contain.
    """

    @pytest.fixture()
    def hand_built_world(self) -> World:
        """
        A world whose floor belongs to a parsed model, with a table and its drawer built
        in code on top of it.
        """
        world = World()
        floor = Body(name=PrefixedName("floor"))
        table = Body(
            name=PrefixedName("table"),
            visual=ShapeCollection(shapes=[Box(scale=Scale(1.0, 0.6, 0.7))]),
        )
        drawer = Body(
            name=PrefixedName("drawer"),
            visual=ShapeCollection(shapes=[Box(scale=Scale(0.3, 0.3, 0.2))]),
        )
        drawer_dof = DegreeOfFreedom(
            name=PrefixedName("drawer_dof"),
            limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-0.5),
                upper=DerivativeMap(position=0.5, velocity=0.5),
            ),
        )
        with world.modify_world():
            world.add_kinematic_structure_entity(floor)
            world.add_kinematic_structure_entity(table)
            world.add_kinematic_structure_entity(drawer)
            world.add_degree_of_freedom(drawer_dof)
            world.add_connection(
                FixedConnection(
                    parent=floor,
                    child=table,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.4, y=0.2, z=0.0
                    ),
                )
            )
            world.add_connection(
                PrismaticConnection(
                    parent=table,
                    child=drawer,
                    axis=Vector3.from_iterable([1, 0, 0]),
                    raw_dof=drawer_dof,
                )
            )
        return world

    def serialize(self, world: World, tmp_path) -> bundler.BundleReport:
        """
        Serialize everything but the floor, as a bundler would for a world whose only
        parsed source claimed the floor.
        """
        return UrdfDocument.of_bodies(
            bodies=[
                world.get_body_by_name("table"),
                world.get_body_by_name("drawer"),
            ],
            name="environment",
            output_directory=str(tmp_path / "bundle"),
            mesh_subdirectory="environment",
        )

    def test_the_subset_becomes_a_urdf_rooted_in_one_link(
        self, hand_built_world, tmp_path
    ):
        report = self.serialize(hand_built_world, tmp_path)

        urdf = Path(report.urdf).read_text()
        assert bundler.BundleReport.LINK_PATTERN.findall(urdf) == [
            UrdfDocument.SYNTHESIZED_ROOT_LINK,
            "table",
            "drawer",
        ]

    def test_a_connection_inside_the_subset_keeps_animating(
        self, hand_built_world, tmp_path
    ):
        """
        The drawer's joint is what the recorded connection positions drive, so it has to
        survive as a prismatic joint under the name the recording keys it by.
        """
        report = self.serialize(hand_built_world, tmp_path)

        # the key Recorder.record_frame writes the drawer's position under
        recorded_key = str(
            hand_built_world.get_body_by_name("drawer").parent_connection.name
        )
        assert report.movable_joints == [recorded_key]
        urdf = Path(report.urdf).read_text()
        assert dict(bundler.BundleReport.JOINT_PATTERN.findall(urdf))[recorded_key] == (
            "prismatic"
        )

    def test_a_body_whose_parent_is_absent_is_grafted_on_at_its_world_pose(
        self, hand_built_world, tmp_path
    ):
        """
        The table's parent is the floor, which this document does not contain, so the
        table hangs off the synthesized root -- and has to keep the place it stood in.
        """
        report = self.serialize(hand_built_world, tmp_path)

        urdf = ElementTree.fromstring(Path(report.urdf).read_text())
        graft = [
            joint
            for joint in urdf.findall("joint")
            if joint.find("child").attrib["link"] == "table"
        ]
        assert len(graft) == 1
        assert graft[0].attrib["type"] == "fixed"
        assert graft[0].find("parent").attrib["link"] == (
            UrdfDocument.SYNTHESIZED_ROOT_LINK
        )
        table_position = hand_built_world.get_body_by_name("table").global_pose.to_np()[
            :3, 3
        ]
        assert [
            float(value) for value in graft[0].find("origin").attrib["xyz"].split()
        ] == pytest.approx(list(table_position))

    def test_the_written_urdf_parses_back_into_a_world(
        self, hand_built_world, tmp_path
    ):
        """
        The viewer only ever loads URDF, so the document has to be a well-formed one:

        a single root, and no joint naming a link it does not contain.
        """
        report = self.serialize(hand_built_world, tmp_path)

        reparsed = URDFParser.from_file(report.urdf).parse()

        assert sorted(str(body.name).split("/")[-1] for body in reparsed.bodies) == (
            sorted(["table", "drawer", UrdfDocument.SYNTHESIZED_ROOT_LINK])
        )

    def test_a_joint_at_a_nonzero_position_is_written_at_its_zero(
        self, hand_built_world, tmp_path
    ):
        """
        URDF reads a joint as ``origin`` followed by the joint's own displacement, and
        the viewer supplies that displacement from the recording.

        So a world whose joints are already displaced when it is bundled -- an MJCF
        keyframe puts the Panda's arm in a home pose, for instance -- must still be
        written at its zero, or the recorded value is applied on top of the displacement
        that is already baked in and the joint ends up moving from the wrong place.
        """
        drawer = hand_built_world.get_body_by_name("drawer")
        drawer.parent_connection.position = 0.25
        zero_origin = drawer.parent_connection.parent_T_connection_expression.to_np()

        report = self.serialize(hand_built_world, tmp_path)

        urdf = ElementTree.fromstring(Path(report.urdf).read_text())
        [joint] = [
            element
            for element in urdf.findall("joint")
            if element.find("child").attrib["link"] == "drawer"
        ]
        written = [float(value) for value in joint.find("origin").attrib["xyz"].split()]
        assert written == pytest.approx(list(zero_origin[:3, 3]), abs=1e-6)

    def test_the_report_claims_the_real_bodies_only(self, hand_built_world, tmp_path):
        """
        The report's links say which bodies are now covered by a model, so the
        synthesized root -- which is no body of the world -- must not appear among them.
        """
        report = self.serialize(hand_built_world, tmp_path)

        assert report.links == ["table", "drawer"]


# %% TestResolveUri


class TestResolveUri:
    def test_a_recorded_resolution_wins(self, tmp_path):
        target = tmp_path / "cup.stl"
        target.write_text("solid cup\nendsolid cup\n")
        resolved = bundler.MeshReference("package://demo/cup.stl").resolve(
            hints={"package://demo/cup.stl": str(target)}
        )
        assert resolved == str(target)

    def test_a_relative_reference_resolves_against_the_urdf(self, tmp_path):
        mesh = tmp_path / "meshes" / "cup.stl"
        mesh.parent.mkdir()
        mesh.write_text("solid cup\nendsolid cup\n")
        assert bundler.MeshReference("meshes/cup.stl").resolve(
            base_directory=str(tmp_path)
        ) == str(mesh)

    def test_a_missing_relative_reference_is_unresolved(self, tmp_path):
        assert (
            bundler.MeshReference("meshes/gone.stl").resolve(
                base_directory=str(tmp_path)
            )
            is None
        )

    def test_a_file_uri_resolves_to_its_path(self, tmp_path):
        mesh = tmp_path / "cup.stl"
        mesh.write_text("solid cup\nendsolid cup\n")
        assert bundler.MeshReference("file://" + str(mesh)).resolve() == str(mesh)

    def test_an_absolute_path_that_exists_resolves_to_itself(self, tmp_path):
        mesh = tmp_path / "cup.stl"
        mesh.write_text("solid cup\nendsolid cup\n")
        assert bundler.MeshReference(str(mesh)).resolve() == str(mesh)

    def test_an_unresolvable_package_uri_is_unresolved(self, monkeypatch):
        """
        Without a recorded hint and with no ROS installation to ask,
        :class:`PackageUriResolver` fails to resolve the package, and the URI comes back
        unresolved rather than raising.
        """
        monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
        monkeypatch.delenv("ROS_PACKAGE_PATH", raising=False)
        monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
        assert (
            bundler.MeshReference("package://no_such_package/cup.stl").resolve() is None
        )


# %% TestReferenceLayout


class TestReferenceLayout:
    def test_a_package_reference_keeps_its_package_directory(self):
        assert bundler.MeshReference(
            "package://demo/meshes/cup.stl"
        ).bundled_relative_path() == ("demo/meshes/cup.stl")

    def test_a_local_reference_lands_in_one_flat_directory(self):
        assert (
            bundler.MeshReference("../far/away/cup.stl").bundled_relative_path()
            == "_local/cup.stl"
        )


# %% TestBundledAssets


class TestBundledAssets:
    def test_an_asset_is_copied_once_however_often_it_is_referenced(self, tmp_path):
        source = tmp_path / "cup.stl"
        source.write_text("solid cup endsolid")
        assets = bundler.BundledAssets()

        assert assets.copy(str(source), str(tmp_path / "out" / "cup.stl")) is True
        assert assets.copy(str(source), str(tmp_path / "elsewhere" / "cup.stl")) is True

        assert assets.copied == {str(source): str(tmp_path / "out" / "cup.stl")}
        assert not (tmp_path / "elsewhere").exists()

    def test_an_unresolved_reference_is_recorded_as_missing(self, tmp_path):
        assets = bundler.BundledAssets()
        assert assets.copy(None, str(tmp_path / "out" / "cup.stl")) is False
        assert assets.missing == [bundler.BundledAssets.UNRESOLVED_REFERENCE]

    def test_a_resolved_path_that_is_not_a_file_is_recorded_as_missing(self, tmp_path):
        assets = bundler.BundledAssets()
        gone = str(tmp_path / "gone.stl")
        assert assets.copy(gone, str(tmp_path / "out" / "gone.stl")) is False
        assert assets.missing == [gone]

    def test_the_textures_a_collada_mesh_names_are_copied_beside_it(self, tmp_path):
        source_directory = tmp_path / "src"
        source_directory.mkdir()
        (source_directory / "wood.png").write_bytes(b"png")
        mesh = source_directory / "table.dae"
        mesh.write_text(
            "<library_images><init_from>wood.png</init_from></library_images>"
        )
        bundled = tmp_path / "out" / "table.dae"

        assets = bundler.BundledAssets()
        assets.copy(str(mesh), str(bundled))
        assets.copy_side_assets(str(mesh), str(bundled))

        assert (tmp_path / "out" / "wood.png").read_bytes() == b"png"
        assert assets.missing == []

    def test_an_object_meshs_material_library_and_its_textures_are_copied(
        self, tmp_path
    ):
        source_directory = tmp_path / "src"
        source_directory.mkdir()
        (source_directory / "cup.mtl").write_text("newmtl body\nmap_Kd glaze.jpg\n")
        (source_directory / "glaze.jpg").write_bytes(b"jpg")
        mesh = source_directory / "cup.obj"
        mesh.write_text("mtllib cup.mtl\nv 0 0 0\n")
        bundled = tmp_path / "out" / "cup.obj"

        assets = bundler.BundledAssets()
        assets.copy(str(mesh), str(bundled))
        assets.copy_side_assets(str(mesh), str(bundled))

        assert (tmp_path / "out" / "cup.mtl").exists()
        assert (tmp_path / "out" / "glaze.jpg").read_bytes() == b"jpg"

    def test_a_stereolithography_mesh_has_no_side_assets(self, tmp_path):
        mesh = tmp_path / "cup.stl"
        mesh.write_text("solid cup endsolid")
        bundled = tmp_path / "out" / "cup.stl"

        assets = bundler.BundledAssets()
        assets.copy(str(mesh), str(bundled))
        assets.copy_side_assets(str(mesh), str(bundled))

        assert list(assets.copied) == [str(mesh)]

    def test_a_texture_beside_the_mesh_keeps_its_relative_location(self, tmp_path):
        """
        Gazebo model trees reference textures from a sibling directory, e.g.
        ``../materials/textures/wall.png``.

        The reference has to be resolved against the mesh and mirrored at the same
        relative place next to the bundled copy, or the browser asks for a file that is
        not there.
        """
        model = tmp_path / "model"
        (model / "meshes").mkdir(parents=True)
        (model / "materials" / "textures").mkdir(parents=True)
        (model / "materials" / "textures" / "wall.png").write_bytes(b"png")
        mesh = model / "meshes" / "wall.dae"
        mesh.write_text("<init_from>../materials/textures/wall.png</init_from>")
        bundled = tmp_path / "bundle" / "meshes" / "model" / "wall.dae"

        assets = bundler.BundledAssets(bundle_root=str(tmp_path / "bundle"))
        assets.copy(str(mesh), str(bundled))
        assets.copy_side_assets(str(mesh), str(bundled))

        texture = tmp_path / "bundle" / "meshes" / "materials" / "textures" / "wall.png"
        assert texture.read_bytes() == b"png"

    def test_a_reference_escaping_the_bundle_is_skipped(self, tmp_path):
        """
        A mesh sitting at the top of the bundle's mesh tree could otherwise write
        outside the bundle entirely.
        """
        source_directory = tmp_path / "src"
        source_directory.mkdir()
        (tmp_path / "outside.png").write_bytes(b"png")
        mesh = source_directory / "wall.dae"
        mesh.write_text("<init_from>../outside.png</init_from>")
        bundled = tmp_path / "bundle" / "wall.dae"

        assets = bundler.BundledAssets(bundle_root=str(tmp_path / "bundle"))
        assets.copy(str(mesh), str(bundled))
        assets.copy_side_assets(str(mesh), str(bundled))

        assert list(assets.copied) == [str(mesh)]

    def test_the_mesh_suffixes_are_sorted_and_deduplicated(self, tmp_path):
        assets = bundler.BundledAssets()
        for name in ("b.STL", "a.stl", "c.dae"):
            source = tmp_path / name
            source.write_text("x")
            assets.copy(str(source), str(tmp_path / "out" / name))
        assert assets.mesh_suffixes == [".dae", ".stl"]


# %% TestBundleUrdf


class TestBundleUrdf:
    @pytest.fixture()
    def source_tree(self, tmp_path):
        """
        A URDF referencing one mesh, both on disk next to each other.
        """
        (tmp_path / "meshes").mkdir()
        (tmp_path / "meshes" / "cup.stl").write_text("solid cup\nendsolid cup\n")
        urdf = tmp_path / "robot.urdf"
        urdf.write_text(ONE_MESH_URDF_TEXT)
        return urdf

    @pytest.fixture()
    def xacro_source_tree(self, tmp_path):
        """
        The same URDF content as :attr:`source_tree`, saved with a ``.xacro`` extension.
        """
        (tmp_path / "meshes").mkdir()
        (tmp_path / "meshes" / "cup.stl").write_text("solid cup\nendsolid cup\n")
        xacro = tmp_path / "robot.xacro"
        xacro.write_text(ONE_MESH_URDF_TEXT)
        return xacro

    def test_the_mesh_is_copied_next_to_the_rewritten_urdf(self, source_tree, tmp_path):
        output_directory = tmp_path / "bundle"
        report = bundler.BundleReport.of_source(
            str(source_tree), "demo", str(output_directory)
        )
        assert (output_directory / "demo.urdf").is_file()
        assert (output_directory / "meshes" / "_local" / "cup.stl").is_file()
        assert report.meshes_copied == 1
        assert report.missing == []

    def test_the_reference_is_rewritten_to_the_bundled_copy(
        self, source_tree, tmp_path
    ):
        output_directory = tmp_path / "bundle"
        bundler.BundleReport.of_source(str(source_tree), "demo", str(output_directory))
        rewritten = (output_directory / "demo.urdf").read_text()
        assert 'filename="meshes/_local/cup.stl"' in rewritten
        assert 'filename="meshes/cup.stl"' not in rewritten

    def test_links_and_joints_are_reported(self, source_tree, tmp_path):
        report = bundler.BundleReport.of_source(
            str(source_tree), "demo", str(tmp_path / "bundle")
        )
        assert report.links == ["base_link", "cup_link"]
        assert report.joints == ["cup_joint"]
        assert report.movable_joints == []

    def test_a_xacro_source_is_bundled_like_a_urdf_source(
        self, xacro_source_tree, tmp_path
    ):
        """
        Bundling a xacro source produces the same links, joints and mesh copy as
        bundling the equivalent URDF - the ElementTree round-trip
        :meth:`URDFParser.from_xacro` performs does not break the regex-based mesh
        rewriting.
        """
        report = bundler.BundleReport.of_source(
            str(xacro_source_tree), "demo", str(tmp_path / "bundle")
        )
        assert report.links == ["base_link", "cup_link"]
        assert report.joints == ["cup_joint"]
        assert report.meshes_copied == 1
        assert report.missing == []

    def test_an_unresolvable_mesh_is_reported_as_missing(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text(
            '<robot name="demo">\n'
            '  <link name="base_link">\n'
            '    <visual><geometry><mesh filename="meshes/gone.stl"/></geometry></visual>\n'
            "  </link>\n"
            "</robot>\n"
        )
        report = bundler.BundleReport.of_source(
            str(urdf), "demo", str(tmp_path / "bundle")
        )
        assert report.missing == [bundler.BundledAssets.UNRESOLVED_REFERENCE]
        assert report.meshes_copied == 0

    def test_a_missing_source_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bundler.BundleReport.of_source(
                str(tmp_path / "gone.urdf"), "demo", str(tmp_path)
            )


# %% TestUnsupportedConnections


class TestUnsupportedConnections:
    """
    A body behind a connection URDF cannot express must survive serialization: it is
    grafted onto the document root at its world pose instead of crashing the bundle.
    """

    def drive_world(self):
        """
        A world whose robot base hangs on an omnidirectional drive.
        """
        world = World()
        root = Body(name=PrefixedName("root", prefix="world"))
        base = Body(name=PrefixedName("base_link", prefix="pr2"))
        with world.modify_world():
            world.add_body(root)
            world.add_connection(
                OmniDrive.create_with_dofs(parent=root, child=base, world=world)
            )
        return world, root, base

    def test_an_omnidirectional_drive_becomes_a_floating_joint(self, tmp_path):
        world, root, base = self.drive_world()

        report = UrdfDocument.of_bodies(
            bodies=[root, base],
            name="environment",
            output_directory=str(tmp_path / "bundle"),
            mesh_subdirectory="environment",
        )

        urdf = Path(report.urdf).read_text()
        assert 'type="floating"' in urdf
        assert report.movable_joints == [str(base.parent_connection.name)]

    def test_a_connection_without_a_joint_type_grafts_the_child(
        self, tmp_path, monkeypatch
    ):
        world, root, base = self.drive_world()
        connection_types = dict(UrdfDocument.CONNECTION_JOINT_TYPES)
        del connection_types[OmniDrive]
        monkeypatch.setattr(UrdfDocument, "CONNECTION_JOINT_TYPES", connection_types)

        report = UrdfDocument.of_bodies(
            bodies=[root, base],
            name="environment",
            output_directory=str(tmp_path / "bundle"),
            mesh_subdirectory="environment",
        )

        urdf = Path(report.urdf).read_text()
        graft_name = "%s_to_%s" % (UrdfDocument.SYNTHESIZED_ROOT_LINK, "pr2/base_link")
        assert graft_name in urdf

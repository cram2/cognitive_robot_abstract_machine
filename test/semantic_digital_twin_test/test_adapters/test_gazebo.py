import os
from dataclasses import dataclass
from math import pi

import numpy as np
import pytest
from typing_extensions import ClassVar

from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    ModelUriResolver,
)
from semantic_digital_twin.adapters.gazebo import (
    GazeboParser,
    UnsupportedJointType,
    UnsupportedGeometryType,
    UnsupportedPoseReference,
    UnsupportedAxisReference,
)
from semantic_digital_twin.exceptions import (
    PathResolutionError,
)
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Cylinder, Mesh, Sphere


@dataclass
class GazeboFixturePaths:
    """
    The paths of the SDF files used in these tests.
    """

    directory: str
    simple_shapes: str
    hinged_door: str
    drawer: str
    mini_world: str
    mini_models: str
    unsupported_joint: str
    unsupported_geometry: str
    named_pose_frame: str
    named_pose_relative_to: str
    model_frame_axis: str
    expressed_in_axis: str


@pytest.fixture
def gazebo_paths():
    """
    Fixture providing the paths of the SDF files used in these tests.
    """
    directory = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "semantic_digital_twin",
            "resources",
            "gazebo",
        )
    )
    mini_warehouse = os.path.join(directory, "mini_warehouse")
    return GazeboFixturePaths(
        directory=directory,
        simple_shapes=os.path.join(directory, "simple_shapes.sdf"),
        hinged_door=os.path.join(directory, "hinged_door.sdf"),
        drawer=os.path.join(directory, "drawer.sdf"),
        mini_world=os.path.join(mini_warehouse, "worlds", "mini.world"),
        mini_models=os.path.join(mini_warehouse, "models"),
        unsupported_joint=os.path.join(directory, "unsupported_joint.sdf"),
        unsupported_geometry=os.path.join(directory, "unsupported_geometry.sdf"),
        named_pose_frame=os.path.join(directory, "named_pose_frame.sdf"),
        named_pose_relative_to=os.path.join(directory, "named_pose_relative_to.sdf"),
        model_frame_axis=os.path.join(directory, "model_frame_axis.sdf"),
        expressed_in_axis=os.path.join(directory, "expressed_in_axis.sdf"),
    )


def body_named(world, name):
    """
    :param world: The world to search.
    :param name: The unprefixed name of the body.
    :return: The single body of the world with that name.
    """
    return [body for body in world.bodies if body.name.name == name][0]


def connection_named(world, name):
    """
    :param world: The world to search.
    :param name: The unprefixed name of the connection.
    :return: The single connection of the world with that name.
    """
    return [
        connection for connection in world.connections if connection.name.name == name
    ][0]


# %% model uri resolution


class TestModelUriResolution:
    """
    Resolution of ``model://`` URIs against the directories that hold models.
    """

    def test_resolves_against_explicit_directory(self, gazebo_paths):
        resolver = ModelUriResolver(model_directories=[gazebo_paths.mini_models])
        assert resolver.resolve("model://shelf") == os.path.join(
            gazebo_paths.mini_models, "shelf"
        )

    def test_resolves_path_inside_model(self, gazebo_paths):
        resolver = ModelUriResolver(model_directories=[gazebo_paths.mini_models])
        assert resolver.resolve("model://shelf/model.sdf") == os.path.join(
            gazebo_paths.mini_models, "shelf", "model.sdf"
        )

    def test_resolves_against_environment(self, gazebo_paths, monkeypatch):
        monkeypatch.setenv("GAZEBO_MODEL_PATH", f"/nowhere:{gazebo_paths.mini_models}")
        resolver = ModelUriResolver()
        assert resolver.resolve("model://pallet") == os.path.join(
            gazebo_paths.mini_models, "pallet"
        )

    def test_explicit_directory_precedes_environment(self, gazebo_paths, monkeypatch):
        monkeypatch.setenv("GAZEBO_MODEL_PATH", gazebo_paths.directory)
        resolver = ModelUriResolver(model_directories=[gazebo_paths.mini_models])
        assert resolver.search_directories()[0] == gazebo_paths.mini_models

    def test_supports_only_model_uris(self):
        resolver = ModelUriResolver()
        assert resolver.supports("model://shelf")
        assert not resolver.supports("package://some_package/model.sdf")
        assert not resolver.supports("/absolute/path.dae")

    def test_error_lists_searched_directories(self, gazebo_paths, monkeypatch):
        monkeypatch.delenv("GAZEBO_MODEL_PATH", raising=False)
        resolver = ModelUriResolver(model_directories=[gazebo_paths.mini_models])
        with pytest.raises(PathResolutionError) as error:
            resolver.resolve("model://absent_model")
        assert gazebo_paths.mini_models in str(error.value)

    def test_inferred_from_world_file_location(self, gazebo_paths, monkeypatch):
        """
        A world next to its models resolves without any configuration, which is what
        makes a world parse straight out of the package that ships it.
        """
        monkeypatch.delenv("GAZEBO_MODEL_PATH", raising=False)
        resolver = GazeboParser.resolver_for_file(gazebo_paths.mini_world)
        assert resolver.resolve("model://shelf") == os.path.join(
            gazebo_paths.mini_models, "shelf"
        )


# %% model files


class TestModelFileSelection:
    """
    Selection of the description file a model directory offers.
    """

    def test_prefers_highest_declared_version(self, gazebo_paths):
        parser = GazeboParser.from_file(gazebo_paths.mini_world)
        model_file = parser.model_file_of_directory(
            os.path.join(gazebo_paths.mini_models, "shelf")
        )
        assert os.path.basename(model_file) == "model.sdf"

    def test_reports_directory_without_configuration(self, gazebo_paths):
        parser = GazeboParser.from_file(gazebo_paths.mini_world)
        with pytest.raises(PathResolutionError):
            parser.model_file_of_directory(gazebo_paths.directory)


# %% shapes and inertial properties


class TestSingleModelParsing:
    """
    Parsing of a document that describes one model.
    """

    def test_body_is_named_after_model_and_link(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        assert len(world.bodies) == 1
        assert world.bodies[0].name.name == "link"
        assert world.bodies[0].name.prefix == "simple_shapes"
        assert world.validate()

    def test_parses_every_visual_shape(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        shapes = body_named(world, "link").visual.shapes
        assert [type(shape) for shape in shapes] == [Box, Cylinder, Sphere]

    def test_parses_shape_dimensions(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        box, cylinder, sphere = body_named(world, "link").visual.shapes
        assert (box.scale.x, box.scale.y, box.scale.z) == (0.2, 0.4, 0.6)
        assert cylinder.width == 0.3
        assert cylinder.height == 0.8
        assert sphere.radius == 0.35

    def test_parses_shape_origin(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        box = body_named(world, "link").visual.shapes[0]
        assert np.allclose(box.origin.to_np()[:3, 3], [1.0, 0.0, 0.0])

    def test_parses_material_color(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        box = body_named(world, "link").visual.shapes[0]
        assert (box.color.R, box.color.G, box.color.B, box.color.A) == (
            0.25,
            0.5,
            0.75,
            1.0,
        )

    def test_collisions_are_separate_from_visuals(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        assert len(body_named(world, "link").collision.shapes) == 1

    def test_parses_inertial_properties(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        inertial = body_named(world, "link").inertial
        assert inertial.mass == 2.5
        assert np.allclose(inertial.center_of_mass.to_np()[:3], [0.1, 0.2, 0.3])
        assert np.allclose(np.diag(inertial.inertia.data), [0.4, 0.5, 0.6])


# %% joints


class TestJointParsing:
    """
    Reconstruction of the kinematic tree from the joints of a model.
    """

    def test_revolute_joint_becomes_revolute_connection(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.hinged_door).parse()
        assert isinstance(connection_named(world, "hinge"), RevoluteConnection)
        assert world.validate()

    def test_link_pose_is_relative_to_the_model(self, gazebo_paths):
        """
        SDF states link poses relative to the model, so at rest the child must sit where
        the model placed it rather than at the joint.
        """
        world = GazeboParser.from_file(gazebo_paths.hinged_door).parse()
        frame_T_door = world.compute_forward_kinematics_np(
            body_named(world, "frame"), body_named(world, "door")
        )
        assert np.allclose(frame_T_door[:3, 3], [0.0, 0.5, 1.0])

    def test_axis_is_expressed_in_the_joint_frame(self, gazebo_paths):
        """
        The hinge is turned a quarter turn about y, so its own x axis points along the
        negative z axis of the parent, and the door must swing about that.
        """
        world = GazeboParser.from_file(gazebo_paths.hinged_door).parse()
        connection = connection_named(world, "hinge")
        connection.position = pi / 2
        world.notify_state_change()

        frame_T_door = world.compute_forward_kinematics_np(
            body_named(world, "frame"), body_named(world, "door")
        )
        rotation_about_negative_z = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        assert np.allclose(frame_T_door[:3, :3], rotation_about_negative_z, atol=1e-9)

    def test_parses_axis(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.hinged_door).parse()
        axis = connection_named(world, "hinge").axis.to_np().flatten()[:3]
        assert np.allclose(axis, [1.0, 0.0, 0.0])

    def test_parses_limits(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.hinged_door).parse()
        limits = connection_named(world, "hinge").raw_dof.limits
        assert limits.lower.position == -1.4
        assert limits.upper.position == 1.4
        assert limits.lower.velocity == -2.5
        assert limits.upper.velocity == 2.5

    def test_parses_dynamics(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.hinged_door).parse()
        dynamics = connection_named(world, "hinge").dynamics
        assert dynamics.damping == 0.7
        assert dynamics.dry_friction == 0.3

    def test_prismatic_joint_becomes_prismatic_connection(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.drawer).parse()
        connection = connection_named(world, "slide")
        assert isinstance(connection, PrismaticConnection)
        assert connection.raw_dof.limits.upper.position == 0.45
        assert world.validate()

    def test_continuous_joint_has_no_position_limits(self, gazebo_paths):
        """
        A continuous joint turns without end, so the limits its file declares must not
        constrain its position.
        """
        world = GazeboParser.from_file(gazebo_paths.drawer).parse()
        limits = connection_named(world, "spin").raw_dof.limits
        assert limits.lower.position is None
        assert limits.upper.position is None
        assert limits.upper.velocity == 3.0


# %% worlds


class TestWorldParsing:
    """
    Parsing of a world that instantiates models by URI and inline.
    """

    def test_world_has_a_root_of_its_own(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.mini_world).parse()
        assert world.root.name.name == "mini_warehouse"
        assert world.validate()

    def test_instantiates_every_active_model(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.mini_world).parse()
        assert {body.name.prefix for body in world.bodies if body.name.prefix} == {
            "shelf_001",
            "shelf_002",
            "pallet_canonical",
            "inline_crate",
        }

    def test_wrapped_include_takes_its_name_from_the_wrapper(self, gazebo_paths):
        """
        Gazebo's non-canonical idiom names the instance on the wrapping model element
        rather than inside the include.
        """
        world = GazeboParser.from_file(gazebo_paths.mini_world).parse()
        shelf_bodies = [
            body for body in world.bodies if body.name.prefix == "shelf_001"
        ]
        assert [body.name.name for body in shelf_bodies] == ["link"]

    def test_wrapped_include_is_placed_by_the_sibling_pose(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.mini_world).parse()
        shelf = [body for body in world.bodies if body.name.prefix == "shelf_001"][0]
        root_T_shelf = world.compute_forward_kinematics_np(world.root, shelf)
        assert np.allclose(root_T_shelf[:3, 3], [2.5, -1.5, 0.0])

    def test_wrapper_pose_composes_with_include_pose(self, gazebo_paths):
        """
        When both the wrapper and the include carry a pose, the pose of the include is
        expressed in the wrapper, as Gazebo composes them.
        """
        world = GazeboParser.from_file(gazebo_paths.mini_world).parse()
        shelf = [body for body in world.bodies if body.name.prefix == "shelf_002"][0]
        root_T_shelf = world.compute_forward_kinematics_np(world.root, shelf)
        assert np.allclose(root_T_shelf[:3, 3], [-3.0, 4.0, 0.25])

    def test_canonical_include_takes_its_name_from_the_include(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.mini_world).parse()
        pallet = [
            body for body in world.bodies if body.name.prefix == "pallet_canonical"
        ][0]
        root_T_pallet = world.compute_forward_kinematics_np(world.root, pallet)
        assert np.allclose(root_T_pallet[:3, 3], [1.0, 2.0, 0.075])

    def test_inline_model_is_placed_at_its_pose(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.mini_world).parse()
        crate = [body for body in world.bodies if body.name.prefix == "inline_crate"][0]
        root_T_crate = world.compute_forward_kinematics_np(world.root, crate)
        assert np.allclose(root_T_crate[:3, 3], [0.0, 0.0, 0.5])

    def test_commented_out_model_is_absent(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.mini_world).parse()
        assert all(body.name.prefix != "shelf_003" for body in world.bodies)

    def test_static_model_is_attached_rigidly(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.mini_world).parse()
        shelf = [body for body in world.bodies if body.name.prefix == "shelf_001"][0]
        assert isinstance(shelf.parent_connection, FixedConnection)

    def test_movable_model_keeps_its_degrees_of_freedom(self, gazebo_paths):
        world = GazeboParser.from_file(gazebo_paths.mini_world).parse()
        pallet = [
            body for body in world.bodies if body.name.prefix == "pallet_canonical"
        ][0]
        assert isinstance(pallet.parent_connection, Connection6DoF)

    def test_instances_of_one_model_do_not_share_bodies(self, gazebo_paths):
        """
        A world usually instantiates the same model many times, and each instance must
        get its own bodies so that they can be placed independently.
        """
        world = GazeboParser.from_file(gazebo_paths.mini_world).parse()
        shelves = [
            body
            for body in world.bodies
            if body.name.prefix in {"shelf_001", "shelf_002"}
        ]
        assert len(shelves) == 2
        assert shelves[0] is not shelves[1]


# %% unsupported constructs


class TestUnsupportedConstructs:
    """
    The constructs outside the supported subset, each of which must name itself when it
    is rejected.

    These cover both the pre-1.7 and the post-1.7 spelling of the constructs whose
    syntax the frame semantics of SDF 1.7 changed.
    """

    def test_unsupported_joint_type(self, gazebo_paths):
        with pytest.raises(UnsupportedJointType) as error:
            GazeboParser.from_file(gazebo_paths.unsupported_joint).parse()
        assert "ball" in str(error.value)

    def test_unsupported_geometry_type(self, gazebo_paths):
        with pytest.raises(UnsupportedGeometryType) as error:
            GazeboParser.from_file(gazebo_paths.unsupported_geometry).parse()
        assert "plane" in str(error.value)

    def test_pose_with_named_frame_attribute(self, gazebo_paths):
        with pytest.raises(UnsupportedPoseReference) as error:
            GazeboParser.from_file(gazebo_paths.named_pose_frame).parse()
        assert "some_frame" in str(error.value)

    def test_pose_with_named_relative_to_attribute(self, gazebo_paths):
        with pytest.raises(UnsupportedPoseReference) as error:
            GazeboParser.from_file(gazebo_paths.named_pose_relative_to).parse()
        assert "some_frame" in str(error.value)

    def test_axis_in_the_parent_model_frame(self, gazebo_paths):
        with pytest.raises(UnsupportedAxisReference) as error:
            GazeboParser.from_file(gazebo_paths.model_frame_axis).parse()
        assert "lift" in str(error.value)

    def test_axis_expressed_in_another_frame(self, gazebo_paths):
        with pytest.raises(UnsupportedAxisReference) as error:
            GazeboParser.from_file(gazebo_paths.expressed_in_axis).parse()
        assert "__model__" in str(error.value)


# %% the AWS RoboMaker small warehouse world


@pytest.fixture(scope="module")
def aws_warehouse_world():
    """
    Fixture providing the parsed AWS RoboMaker small warehouse world.
    """
    return GazeboParser.from_file(TestSmallWarehouseWorld.world_uri).parse()


class TestSmallWarehouseWorld:
    """
    Parsing of the AWS RoboMaker small warehouse world, which is a world of SDF 1.6
    whose models are included by URI and drawn with COLLADA meshes.

    Reads the world from the ``aws_robomaker_small_warehouse_world`` package, which has
    to be built in the workspace.
    """

    world_uri: ClassVar[str] = (
        "package://aws_robomaker_small_warehouse_world/worlds/small_warehouse/"
        "small_warehouse.world"
    )
    """
    The URI of the world file inside the package.
    """

    instance_count: ClassVar[int] = 26
    """
    The number of active model instances the world declares.
    """

    def test_world_is_valid(self, aws_warehouse_world):
        assert aws_warehouse_world.validate()

    def test_instantiates_every_model(self, aws_warehouse_world):
        instance_names = {
            body.name.prefix for body in aws_warehouse_world.bodies if body.name.prefix
        }
        assert len(instance_names) == self.instance_count

    def test_meshes_resolve_to_files_on_disk(self, aws_warehouse_world):
        """
        The models refer to their meshes by ``model://`` URI and spell the COLLADA
        extension in upper case, so resolving them proves the whole chain from the world
        file to the asset.
        """
        meshes = [
            shape
            for body in aws_warehouse_world.bodies
            for shape in body.visual.shapes
            if isinstance(shape, Mesh)
        ]
        assert meshes
        assert all(os.path.isfile(mesh.filename) for mesh in meshes)

    def test_instance_is_placed_at_the_declared_pose(self, aws_warehouse_world):
        """
        The world places its instances with the pose that is a sibling of the include
        rather than a child of it.
        """
        shelf = [
            body
            for body in aws_warehouse_world.bodies
            if body.name.prefix == "aws_robomaker_warehouse_ShelfE_01_001"
        ][0]
        root_T_shelf = aws_warehouse_world.compute_forward_kinematics_np(
            aws_warehouse_world.root, shelf
        )
        assert np.allclose(root_T_shelf[:3, 3], [4.73156, 0.57943, 0.0])

    def test_instance_has_the_size_its_meshes_declare(self, aws_warehouse_world):
        """
        The models draw themselves with COLLADA meshes written in centimeters, so the
        parsed shelf must come out at the size a warehouse shelf actually has rather
        than a hundred times that.
        """
        shelf = [
            body
            for body in aws_warehouse_world.bodies
            if body.name.prefix == "aws_robomaker_warehouse_ShelfE_01_001"
        ][0]

        extents = shelf.collision.combined_mesh.extents

        assert extents == pytest.approx([3.918, 0.880, 2.613], abs=1e-3)

    def test_models_are_attached_rigidly(self, aws_warehouse_world):
        """
        Every model of the warehouse is static, so none of them may float freely.
        """
        placed_bodies = [
            body for body in aws_warehouse_world.bodies if body.name.prefix
        ]
        assert all(
            isinstance(body.parent_connection, FixedConnection)
            for body in placed_bodies
        )

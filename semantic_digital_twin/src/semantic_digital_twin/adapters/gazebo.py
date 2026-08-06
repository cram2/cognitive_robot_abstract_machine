from __future__ import annotations, absolute_import

import logging
import os
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from typing import List

from typing_extensions import ClassVar, Dict, List, Optional, Tuple, Type

from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    FileUriResolver,
    ModelUriResolver,
    PackageUriResolver,
    PathResolver,
    SearchPathFileResolver,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    NegativeConnectionVelocity,
    PathResolutionError,
    ParsingError,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import JointDynamics
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Mesh,
    Scale,
    Shape,
    Sphere,
)
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Connection

logger = logging.getLogger(__name__)


@dataclass
class JointDescription:
    """
    A joint that has been read from a description and validated, but not yet turned into
    a connection.

    Reading a joint can fail on constructs the parser does not support, while creating a
    connection modifies a world. Keeping the two apart lets a failure surface before a
    world has been touched.
    """

    name: PrefixedName
    """
    The name of the connection the joint becomes.
    """

    connection_type: Type[Connection]
    """
    The type of connection the joint becomes.
    """

    parent: Body
    """
    The body the joint moves the child relative to.
    """

    child: Body
    """
    The body the joint moves.
    """

    parent_T_connection: HomogeneousTransformationMatrix
    """
    The pose of the joint relative to the parent body.
    """

    connection_T_child: HomogeneousTransformationMatrix
    """
    The pose of the child body relative to the joint.
    """

    axis: Optional[Vector3] = None
    """
    The axis the joint moves along, in the frame of the joint.
    """

    limits: Optional[DegreeOfFreedomLimits] = None
    """
    The limits of the joint's degree of freedom.
    """

    dynamics: JointDynamics = field(default_factory=JointDynamics)
    """
    The dynamic properties of the joint.
    """


@dataclass
class GazeboParser:
    """
    Parses Gazebo SDF model and world descriptions into worlds.

    Files of SDF 1.4 through 1.8 are accepted: both the pre-1.7 ``frame`` and the
    post-1.7 ``relative_to`` pose syntax are read, and both are only supported with
    default frame semantics. Constructs outside the supported subset, such as named
    frame references, axes expressed in another frame, and joint or geometry types
    without a counterpart, raise a :class:`ParsingError` subclass naming the construct.
    The declared version is never branched on.

    .. note:: A model declared ``static`` is attached rigidly, but joints inside it stay
        movable.
    """

    connection_type_map: ClassVar[Dict[str, Type[Connection]]] = {
        "revolute": RevoluteConnection,
        "continuous": RevoluteConnection,
        "prismatic": PrismaticConnection,
        "fixed": FixedConnection,
    }
    """
    Maps a joint's declared type to the connection it becomes.
    """

    world_link_name: ClassVar[str] = "world"
    """
    The reserved link name that refers to the simulation world rather than a model link.
    """

    truthy_values: ClassVar[Tuple[str, ...]] = ("1", "true")
    """
    The texts that SDF accepts for a boolean element that is set.
    """

    interpreted_elements: ClassVar[Tuple[str, ...]] = (
        "model",
        "include",
        "link",
        "joint",
        "pose",
        "static",
    )
    """
    The child elements of a world or model that carry into the parsed world.
    """

    sdf: str
    """
    The SDF document.
    """

    prefix: Optional[str] = None
    """
    The prefix for every name used in this world.
    """

    path_resolver: PathResolver = field(default_factory=CompositePathResolver)
    """
    The path resolver used for the URIs referenced by this document.
    """

    model_element_cache: dict = field(default_factory=dict)
    """
    Maps a model description file path to its already parsed ``model`` element, so that
    a world instantiating the same model many times parses its file only once.
    """

    def __post_init__(self):
        self.root_element = ElementTree.fromstring(self.sdf)

    # %% construction

    @classmethod
    def from_file(
        cls,
        file_path: str,
        prefix: Optional[str] = None,
        path_resolver: Optional[PathResolver] = None,
    ) -> GazeboParser:
        """
        Creates a parser for a description file.

        Unless a resolver is given, one is built that finds ``model://`` URIs relative to
        the file, so that a world shipped next to its models parses without further
        configuration.

        :param file_path: The path of the file to parse.
        :param prefix: The prefix for every name used in this world.
        :param path_resolver: The resolver for the URIs referenced by the file.
        :return: A parser for the described world.
        """
        file_locator = path_resolver or CompositePathResolver()
        resolved_path = file_locator.resolve(file_path)
        path_resolver = path_resolver or cls.resolver_for_file(resolved_path)
        with open(resolved_path, "r") as file:
            sdf = file.read()
        parser = cls(sdf=sdf, prefix=prefix)
        parser.path_resolver = path_resolver
        return parser

    @staticmethod
    def resolver_for_file(file_path: str) -> CompositePathResolver:
        """
        Builds a resolver that searches for models and assets in the usual places
        relative to a description file, covering both flat and nested world directory
        layouts.

        Models are looked up in a ``models`` directory at or above the file, and the
        directories holding those are also searched for the relative paths that
        descriptions written for a resource search path use.

        :param file_path: The path of the file whose location is used for the search.
        :return: The resolver for the URIs referenced by that file.
        """
        directory = os.path.dirname(os.path.abspath(file_path))
        root_directories = [
            directory,
            os.path.normpath(os.path.join(directory, "..")),
            os.path.normpath(os.path.join(directory, "..", "..")),
        ]
        return CompositePathResolver(
            [
                ModelUriResolver(
                    model_directories=[
                        os.path.join(root, "models") for root in root_directories
                    ]
                ),
                SearchPathFileResolver(root_directories=root_directories),
                FileUriResolver(base_directory=directory),
                PackageUriResolver(),
            ]
        )

    # %% entry point

    def parse(self) -> World:
        """
        Parses the document into a world.

        :return: The world described by a ``world`` element, or by a single ``model``
            element if the document describes one model.
        """
        logger.debug("Parsing SDF version %s.", self.root_element.get("version"))

        world_element = self.root_element.find("world")
        if world_element is not None:
            return self.parse_world(world_element)

        model_element = self.root_element.find("model")
        if model_element is None:
            raise MissingRootElement(expected_elements=["world", "model"])

        return self.parse_model(model_element, self.prefix or model_element.get("name"))

    # %% worlds

    def parse_world(self, element: ElementTree.Element) -> World:
        """
        Parses a ``world`` element into a world with an explicit root body that all
        contained models are attached to.

        :param element: The ``world`` element to parse.
        :return: The world holding every model the element contains.
        """
        world_name = self.prefix or element.get("name")
        world = World()
        world.name = world_name
        with world.modify_world():
            world.add_kinematic_structure_entity(Body(name=PrefixedName(world_name)))

        self.attach_contained_models(world, element)
        return world

    def attach_contained_models(
        self, world: World, container: ElementTree.Element
    ) -> None:
        """
        Attaches every model the container references to the root of the world, whether
        it is included by URI or written inline.

        Gazebo's non-canonical idiom of wrapping an ``include`` in a ``model`` element is
        supported alongside the canonical form; the wrapper supplies the instance name
        and a pose that composes with the pose of the include.

        :param world: The world the models are attached to.
        :param container: The ``world`` or ``model`` element holding the references.
        """
        for include_element in container.findall("include"):
            self.attach_included_model(world, include_element, instance_name=None)

        for model_element in container.findall("model"):
            instance_name = model_element.get("name")
            include_element = model_element.find("include")
            if include_element is not None:
                self.attach_included_model(
                    world,
                    include_element,
                    instance_name=instance_name,
                    wrapper_pose=self.parse_pose(model_element.find("pose")),
                )
                continue

            sub_world = self.parse_model(model_element, instance_name)
            self.merge_at_pose(
                world,
                sub_world,
                self.parse_pose(model_element.find("pose")),
                self.parse_boolean(model_element.find("static")),
            )

        for skipped_element in container:
            if skipped_element.tag in self.interpreted_elements:
                continue
            logger.debug("Skipping unsupported element <%s>.", skipped_element.tag)

    def attach_included_model(
        self,
        world: World,
        include_element: ElementTree.Element,
        instance_name: Optional[str],
        wrapper_pose: Optional[HomogeneousTransformationMatrix] = None,
    ) -> None:
        """
        Resolves an ``include`` to its model file and attaches an instance of it.

        :param world: The world the instance is attached to.
        :param include_element: The ``include`` element to resolve.
        :param instance_name: The name of the instance, taken from a wrapping ``model``
            element if there is one.
        :param wrapper_pose: The pose of the wrapping ``model`` element, which the pose
            of the include is expressed in.
        """
        uri = include_element.findtext("uri").strip()
        model_element = self.load_model_element(uri)
        name = (
            include_element.findtext("name")
            or instance_name
            or model_element.get("name")
        )

        pose = self.parse_pose(include_element.find("pose"))
        if wrapper_pose is not None:
            pose = wrapper_pose @ pose

        static_element = include_element.find("static")
        if static_element is None:
            static_element = model_element.find("static")

        sub_world = self.parse_model(model_element, name.strip())
        self.merge_at_pose(world, sub_world, pose, self.parse_boolean(static_element))

    def merge_at_pose(
        self,
        world: World,
        sub_world: World,
        pose: HomogeneousTransformationMatrix,
        is_static: bool,
    ) -> None:
        """
        Merges a model into the world at a pose relative to the world root.

        A static model is attached rigidly, so that it does not gain the degrees of
        freedom a free-floating model would.

        :param world: The world the model is merged into.
        :param sub_world: The world holding the model.
        :param pose: The pose of the model relative to the root of the world.
        :param is_static: Whether the model is declared static.
        """
        if not is_static:
            world.merge_world_at_pose(sub_world, pose)
            return

        root_connection = FixedConnection(
            name=PrefixedName("root", sub_world.root.name.prefix),
            parent=world.root,
            child=sub_world.root,
            parent_T_connection_expression=pose,
        )
        world.merge_world(sub_world, root_connection)

    # %% model files

    def load_model_element(self, uri: str) -> ElementTree.Element:
        """
        Loads the ``model`` element a ``model://`` URI refers to.

        Documents are cached, because a world typically instantiates the same model many
        times.

        :param uri: The URI of the model directory.
        :return: The ``model`` element of the model's description file.
        """
        model_directory = self.path_resolver.resolve(uri)
        model_file = self.model_file_of_directory(model_directory)
        if model_file in self.model_element_cache:
            return self.model_element_cache[model_file]

        root_element = ElementTree.parse(model_file).getroot()
        model_element = root_element.find("model")
        if model_element is None:
            raise MissingRootElement(file_path=model_file, expected_elements=["model"])

        self.model_element_cache[model_file] = model_element
        return model_element

    def model_file_of_directory(self, model_directory: str) -> str:
        """
        Reads the ``model.config`` of a model directory and returns the description file
        it points at, preferring the highest declared version.

        The version only selects between the files a model ships; whether the parser
        supports the file's contents is decided per construct while parsing.

        :param model_directory: The directory holding the model.
        :return: The path of the model's description file.
        """
        config_path = os.path.join(model_directory, "model.config")
        if not os.path.isfile(config_path):
            raise PathResolutionError(
                uri=model_directory, details="no model.config in model directory"
            )

        candidates = [
            (
                self.version_of(sdf_element.get("version", "0")),
                "-" not in os.path.basename(sdf_element.text.strip()),
                sdf_element.text.strip(),
            )
            for sdf_element in ElementTree.parse(config_path).getroot().iter("sdf")
            if sdf_element.text
        ]
        if not candidates:
            raise PathResolutionError(
                uri=config_path, details="model.config declares no sdf file"
            )

        return os.path.join(model_directory, max(candidates)[2])

    @staticmethod
    def version_of(version: str) -> Tuple[int, ...]:
        """
        :param version: A dotted version such as ``1.6``.
        :return: The version as a tuple that orders like the version does.
        """
        return tuple(int(part) for part in version.split(".") if part.isdigit())

    # %% models

    def parse_model(self, element: ElementTree.Element, instance_name: str) -> World:
        """
        Parses a ``model`` element into a world of its own, whose root is the model's
        root link.

        Link poses in SDF are relative to the model rather than to the parent link, so
        the kinematic tree is reconstructed from the joints.

        :param element: The ``model`` element to parse.
        :param instance_name: The name that prefixes every name in the model, which
            distinguishes several instances of the same model.
        :return: The world holding the model.
        """
        bodies = {}
        model_T_link = {}
        for link_element in element.findall("link"):
            link_name = link_element.get("name")
            bodies[link_name] = self.parse_link(link_element, instance_name)
            model_T_link[link_name] = self.parse_pose(link_element.find("pose"))

        joint_elements = [
            joint_element
            for joint_element in element.findall("joint")
            if self.is_attached_to_a_link(joint_element)
        ]
        child_link_names = {
            joint_element.findtext("child").strip() for joint_element in joint_elements
        }
        is_static = self.parse_boolean(element.find("static"))

        joint_descriptions = [
            self.parse_joint(joint_element, bodies, model_T_link, instance_name)
            for joint_element in joint_elements
        ]

        world = World()
        world.name = instance_name
        root_body = self.root_body_of(bodies, child_link_names, instance_name)

        with world.modify_world():
            world.add_kinematic_structure_entity(root_body)
            for description in joint_descriptions:
                world.add_connection(self.create_connection(world, description))

            model_T_root = model_T_link[root_body.name.name]
            for link_name, body in bodies.items():
                if body is root_body or link_name in child_link_names:
                    continue
                self.attach_unjointed_link(
                    world,
                    root_body,
                    body,
                    model_T_root.inverse() @ model_T_link[link_name],
                    is_static,
                )

        self.attach_contained_models(world, element)
        return world

    @staticmethod
    def root_body_of(
        bodies: Dict[str, Body], child_link_names: set, instance_name: str
    ) -> Body:
        """
        Determines the root link of a model, which is the first link no joint moves.

        :param bodies: The bodies of the model, by link name.
        :param child_link_names: The names of the links that are the child of a joint.
        :param instance_name: The name of the model instance, used when it has no links.
        :return: The body that roots the model.
        """
        for link_name, body in bodies.items():
            if link_name not in child_link_names:
                return body
        return Body(name=PrefixedName(instance_name))

    def attach_unjointed_link(
        self,
        world: World,
        root_body: Body,
        body: Body,
        root_T_link: HomogeneousTransformationMatrix,
        is_static: bool,
    ) -> None:
        """
        Attaches a link that no joint connects to the root of its model.

        Such a link is a free body, unless the model is static, in which case it is fixed
        in place like the rest of the model.

        :param world: The world the link is added to.
        :param root_body: The root of the model.
        :param body: The body of the unjointed link.
        :param root_T_link: The pose of the link relative to the root of the model.
        :param is_static: Whether the model is declared static.
        """
        if is_static:
            world.add_connection(
                FixedConnection(
                    name=PrefixedName(body.name.name, body.name.prefix),
                    parent=root_body,
                    child=body,
                    parent_T_connection_expression=root_T_link,
                )
            )
            return

        connection = Connection6DoF.create_with_dofs(
            parent=root_body, child=body, world=world
        )
        world.add_connection(connection)
        connection.origin = root_T_link

    def is_attached_to_a_link(self, joint_element: ElementTree.Element) -> bool:
        """
        :param joint_element: The ``joint`` element to check.
        :return: Whether both ends of the joint are links of the model rather than the
            simulation world.
        """
        endpoints = (
            joint_element.findtext("parent", "").strip(),
            joint_element.findtext("child", "").strip(),
        )
        if self.world_link_name not in endpoints:
            return True

        logger.debug(
            "Skipping joint <%s>, which attaches a model to the world.",
            joint_element.get("name"),
        )
        return False

    # %% links

    def parse_link(self, element: ElementTree.Element, prefix: str) -> Body:
        """
        Parses a ``link`` element into a body with its shapes and inertial properties.

        :param element: The ``link`` element to parse.
        :param prefix: The prefix of the name of the body.
        :return: The body describing the link.
        """
        body = Body(name=PrefixedName(element.get("name"), prefix))
        body.visual = self.parse_shapes(element.findall("visual"), body)
        body.collision = self.parse_shapes(element.findall("collision"), body)

        inertial = self.parse_inertial(element.find("inertial"), body)
        if inertial is not None:
            body.inertial = inertial
        return body

    def parse_inertial(
        self, element: Optional[ElementTree.Element], body: Body
    ) -> Optional[Inertial]:
        """
        Parses an ``inertial`` element.

        SDF expresses the inertia tensor in the inertial frame of the link, so it is
        rotated into the link frame, which is the frame :class:`Inertial` expects.

        :param element: The ``inertial`` element to parse.
        :param body: The body the properties belong to, used as their reference frame.
        :return: The inertial properties, or ``None`` if the link declares none.
        """
        if element is None:
            return None

        link_T_inertial = self.parse_pose(element.find("pose")).to_np()
        inertia_element = element.find("inertia")
        inertia_in_inertial_frame = InertiaTensor.from_values(
            ixx=self.parse_float(inertia_element, "ixx", 0.0),
            iyy=self.parse_float(inertia_element, "iyy", 0.0),
            izz=self.parse_float(inertia_element, "izz", 0.0),
            ixy=self.parse_float(inertia_element, "ixy", 0.0),
            ixz=self.parse_float(inertia_element, "ixz", 0.0),
            iyz=self.parse_float(inertia_element, "iyz", 0.0),
        )

        link_R_inertial = link_T_inertial[:3, :3]
        inertia = link_R_inertial @ inertia_in_inertial_frame.data @ link_R_inertial.T

        return Inertial(
            mass=self.parse_float(element, "mass", 1.0),
            center_of_mass=Point3(*link_T_inertial[:3, 3], reference_frame=body),
            inertia=InertiaTensor(data=inertia),
        )

    # %% shapes

    def parse_shapes(
        self, elements: List[ElementTree.Element], body: Body
    ) -> ShapeCollection:
        """
        Parses the ``visual`` or ``collision`` elements of a link.

        :param elements: The elements to parse.
        :param body: The body the shapes belong to, used as their reference frame.
        :return: The shapes of the link.
        """
        shapes = [
            self.parse_geometry(
                element.find("geometry"),
                self.parse_pose(element.find("pose"), reference_frame=body),
                self.parse_color(element.find("material")),
            )
            for element in elements
            if element.find("geometry") is not None
        ]
        return ShapeCollection(shapes, reference_frame=body)

    def parse_geometry(
        self,
        element: ElementTree.Element,
        origin: HomogeneousTransformationMatrix,
        color: Color,
    ) -> Shape:
        """
        Parses a ``geometry`` element into the shape it describes.

        :param element: The ``geometry`` element to parse.
        :param origin: The pose of the shape relative to its link.
        :param color: The color of the shape.
        :return: The shape described by the element.
        """
        box_element = element.find("box")
        if box_element is not None:
            size = self.parse_vector(box_element.findtext("size"))
            return Box(origin=origin, scale=Scale(*size), color=color)

        sphere_element = element.find("sphere")
        if sphere_element is not None:
            radius = self.parse_float(sphere_element, "radius", 0.5)
            return Sphere(origin=origin, radius=radius, color=color)

        cylinder_element = element.find("cylinder")
        if cylinder_element is not None:
            radius = self.parse_float(cylinder_element, "radius", 0.5)
            return Cylinder(
                origin=origin,
                width=radius * 2,
                height=self.parse_float(cylinder_element, "length", 0.5),
                color=color,
            )

        mesh_element = element.find("mesh")
        if mesh_element is not None:
            scale = mesh_element.findtext("scale")
            return Mesh(
                origin=origin,
                filename=self.path_resolver.resolve(
                    mesh_element.findtext("uri").strip()
                ),
                scale=Scale(*self.parse_vector(scale)) if scale else Scale(),
                color=color,
            )

        declared_types = [child.tag for child in element]
        raise UnsupportedGeometryType(
            geometry_type=declared_types[0] if declared_types else "none",
            supported_types=["box", "cylinder", "mesh", "sphere"],
        )

    def parse_color(self, element: Optional[ElementTree.Element]) -> Color:
        """
        Parses the diffuse color of a ``material`` element.

        :param element: The ``material`` element to parse.
        :return: The color of the material, white if it declares none.
        """
        if element is None:
            return Color()

        diffuse = element.findtext("diffuse")
        if diffuse is None:
            return Color()

        return Color(*(float(value) for value in diffuse.split()))

    # %% joints

    def parse_joint(
        self,
        element: ElementTree.Element,
        bodies: Dict[str, Body],
        model_T_link: Dict[str, HomogeneousTransformationMatrix],
        prefix: str,
    ) -> JointDescription:
        """
        Parses a ``joint`` element into a validated description of the connection it
        becomes.

        The pose of an SDF joint is relative to its child link, so the connection is
        placed at the joint and the child is offset back to its own frame.

        :param element: The ``joint`` element to parse.
        :param bodies: The bodies of the model, by link name.
        :param model_T_link: The poses of the links relative to the model.
        :param prefix: The prefix of the name of the connection.
        :return: The description of the joint.
        :raises UnsupportedJointType: If the joint type has no connection counterpart.
        """
        joint_name = element.get("name")
        joint_type = element.get("type")
        connection_type = self.connection_type_map.get(joint_type)
        if connection_type is None:
            raise UnsupportedJointType(
                joint_name=joint_name,
                joint_type=joint_type,
                supported_types=list(self.connection_type_map),
            )

        parent_name = element.findtext("parent").strip()
        child_name = element.findtext("child").strip()
        child_T_connection = self.parse_pose(element.find("pose"))
        parent_T_connection = (
            model_T_link[parent_name].inverse()
            @ model_T_link[child_name]
            @ child_T_connection
        )

        description = JointDescription(
            name=PrefixedName(joint_name, prefix),
            connection_type=connection_type,
            parent=bodies[parent_name],
            child=bodies[child_name],
            parent_T_connection=parent_T_connection,
            connection_T_child=child_T_connection.inverse(),
        )
        if connection_type is FixedConnection:
            return description

        description.axis = self.parse_axis(element, joint_name, bodies[parent_name])
        description.limits = self.parse_limits(element, joint_type, joint_name)
        description.dynamics = self.parse_dynamics(element.find("axis/dynamics"))
        return description

    def create_connection(
        self, world: World, description: JointDescription
    ) -> Connection:
        """
        Creates the connection a joint description denotes, adding its degree of freedom
        to the world.

        :param world: The world the degree of freedom is added to.
        :param description: The description of the joint.
        :return: The connection describing the joint.
        """
        if description.connection_type is FixedConnection:
            return FixedConnection(
                name=description.name,
                parent=description.parent,
                child=description.child,
                parent_T_connection_expression=description.parent_T_connection,
                connection_T_child_expression=description.connection_T_child,
            )

        connection = description.connection_type.create_with_dofs(
            world=world,
            parent=description.parent,
            child=description.child,
            name=description.name,
            parent_T_connection_expression=description.parent_T_connection,
            connection_T_child_expression=description.connection_T_child,
            dof_limits=description.limits,
            axis=description.axis,
        )
        connection.dynamics = description.dynamics
        return connection

    def parse_axis(
        self, element: ElementTree.Element, joint_name: str, parent_body: Body
    ) -> Vector3:
        """
        Parses the axis of a ``joint`` element.

        The axis is expressed in the frame of the joint, which is the frame the
        connection places its motion in, so it is taken as it is written.

        :param element: The ``joint`` element to parse.
        :param joint_name: The name of the joint, used to report an unsupported frame.
        :param parent_body: The parent of the joint, used as the reference frame.
        :return: The axis the joint moves along.
        :raises UnsupportedAxisReference: If the axis is expressed in another frame.
        """
        axis_element = element.find("axis")
        if axis_element is None:
            return Vector3(0, 0, 1, reference_frame=parent_body)

        uses_model_frame = axis_element.findtext("use_parent_model_frame", "0")
        if uses_model_frame.strip().lower() in self.truthy_values:
            raise UnsupportedAxisReference(
                joint_name=joint_name, reference="the parent model frame"
            )

        xyz_element = axis_element.find("xyz")
        if xyz_element is None:
            return Vector3(0, 0, 1, reference_frame=parent_body)

        expressed_in = xyz_element.get("expressed_in", "")
        if expressed_in:
            raise UnsupportedAxisReference(
                joint_name=joint_name, reference=expressed_in
            )

        return Vector3(
            *self.parse_vector(xyz_element.text), reference_frame=parent_body
        )

    def parse_limits(
        self, element: ElementTree.Element, joint_type: str, joint_name: str
    ) -> DegreeOfFreedomLimits:
        """
        Parses the limits of a ``joint`` element.

        A continuous joint turns without end, so it is given no position limits.

        :param element: The ``joint`` element to parse.
        :param joint_type: The declared type of the joint.
        :param joint_name: The name of the joint, used to report a negative velocity.
        :return: The limits of the joint's degree of freedom.
        """
        lower_limits = DerivativeMap()
        upper_limits = DerivativeMap()
        limit_element = element.find("axis/limit")
        if limit_element is None:
            return DegreeOfFreedomLimits(lower=lower_limits, upper=upper_limits)

        if joint_type != "continuous":
            lower_limits.position = self.parse_optional_float(limit_element, "lower")
            upper_limits.position = self.parse_optional_float(limit_element, "upper")

        velocity = self.parse_optional_float(limit_element, "velocity")
        if velocity is not None and velocity < 0:
            raise NegativeConnectionVelocity(
                connection_name=joint_name, velocity=velocity
            )
        lower_limits.velocity = -velocity if velocity is not None else None
        upper_limits.velocity = velocity

        return DegreeOfFreedomLimits(lower=lower_limits, upper=upper_limits)

    def parse_dynamics(self, element: Optional[ElementTree.Element]) -> JointDynamics:
        """
        Parses a ``dynamics`` element.

        Properties the joint leaves undeclared keep their default.

        :param element: The ``dynamics`` element to parse.
        :return: The dynamic properties of the joint.
        """
        if element is None:
            return JointDynamics()

        return JointDynamics(
            damping=self.parse_float(element, "damping", 0.0),
            dry_friction=self.parse_float(element, "friction", 0.0),
        )

    # %% primitives

    def parse_pose(
        self,
        element: Optional[ElementTree.Element],
        reference_frame: Optional[Body] = None,
    ) -> HomogeneousTransformationMatrix:
        """
        Parses a ``pose`` element.

        Both the ``frame`` attribute of SDF below 1.7 and the ``relative_to`` attribute
        of later versions are read, and both are only supported when they are empty,
        which is the default frame of the element the pose belongs to.

        :param element: The ``pose`` element to parse.
        :param reference_frame: The frame the pose is expressed in.
        :return: The pose, or the identity if the element is absent.
        :raises UnsupportedPoseReference: If the pose refers to a named frame.
        :raises MalformedPose: If the pose does not hold 6 values.
        """
        if element is None:
            return HomogeneousTransformationMatrix(reference_frame=reference_frame)

        for attribute in ("frame", "relative_to"):
            reference = element.get(attribute, "").strip()
            if reference:
                raise UnsupportedPoseReference(attribute=attribute, reference=reference)

        text = element.text or ""
        values = text.split()
        if len(values) != 6:
            raise MalformedPose(text=text)

        return HomogeneousTransformationMatrix.from_xyz_rpy(
            *(float(value) for value in values), reference_frame=reference_frame
        )

    @staticmethod
    def parse_vector(text: Optional[str]) -> List[float]:
        """
        :param text: Whitespace separated numbers.
        :return: The numbers the text holds.
        """
        return [float(value) for value in (text or "").split()]

    @classmethod
    def parse_boolean(cls, element: Optional[ElementTree.Element]) -> bool:
        """
        :param element: The element holding the flag.
        :return: Whether the element is present and set.
        """
        if element is None:
            return False
        return (element.text or "").strip().lower() in cls.truthy_values

    @staticmethod
    def parse_float(
        element: Optional[ElementTree.Element], name: str, default: float
    ) -> float:
        """
        :param element: The element holding the value as a child.
        :param name: The name of the child element.
        :param default: The value to use when the child is absent.
        :return: The value of the child element.
        """
        if element is None:
            return default
        text = element.findtext(name)
        return float(text) if text is not None else default

    @staticmethod
    def parse_optional_float(
        element: ElementTree.Element, name: str
    ) -> Optional[float]:
        """
        :param element: The element holding the value as a child.
        :param name: The name of the child element.
        :return: The value of the child element, or ``None`` if it is absent.
        """
        text = element.findtext(name)
        return float(text) if text is not None else None


@dataclass
class MissingRootElement(ParsingError):
    """
    Raised when a description file contains none of the expected root elements.
    """

    expected_elements: List[str] = field(kw_only=True, default_factory=list)
    """
    The element names that were searched for.
    """

    def error_message(self) -> str:
        return f"None of the expected root elements {', '.join(self.expected_elements)} were found."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class MalformedPose(ParsingError):
    """
    Raised when a pose does not consist of a position and roll-pitch-yaw triple.
    """

    text: str = field(kw_only=True)
    """
    The pose text that could not be interpreted.
    """

    def error_message(self) -> str:
        return f"Pose '{self.text}' does not contain 6 values."

    def suggest_correction(self) -> str:
        return "Write the pose as 'x y z roll pitch yaw'."


@dataclass
class UnsupportedJointType(ParsingError):
    """
    Raised when a parsed joint uses a type that has no connection counterpart.
    """

    joint_name: str = field(kw_only=True)
    """
    The name of the joint that could not be mapped.
    """

    joint_type: str = field(kw_only=True)
    """
    The joint type that is not supported.
    """

    supported_types: List[str] = field(kw_only=True, default_factory=list)
    """
    The joint types that can be mapped to connections.
    """

    def error_message(self) -> str:
        return f"Joint '{self.joint_name}' has unsupported type '{self.joint_type}'."

    def suggest_correction(self) -> str:
        if not self.supported_types:
            return ""
        return f"Use one of the supported types: {', '.join(sorted(self.supported_types))}."


@dataclass
class UnsupportedGeometryType(ParsingError):
    """
    Raised when a parsed geometry uses a shape that has no counterpart.
    """

    geometry_type: str = field(kw_only=True)
    """
    The geometry type that is not supported.
    """

    supported_types: List[str] = field(kw_only=True, default_factory=list)
    """
    The geometry types that can be mapped to shapes.
    """

    def error_message(self) -> str:
        return f"Unsupported geometry type '{self.geometry_type}'."

    def suggest_correction(self) -> str:
        if not self.supported_types:
            return ""
        return f"Use one of the supported types: {', '.join(sorted(self.supported_types))}."


@dataclass
class UnsupportedPoseReference(ParsingError):
    """
    Raised when a pose is expressed relative to a named frame.

    Poses are only supported with their default reference, which is the frame of the
    element the pose belongs to.
    """

    attribute: str = field(kw_only=True)
    """
    The attribute carrying the frame reference, ``frame`` or ``relative_to``.
    """

    reference: str = field(kw_only=True)
    """
    The referenced frame.
    """

    def error_message(self) -> str:
        return (
            f"Pose is expressed relative to frame '{self.reference}' via "
            f"'{self.attribute}', but only default frame semantics are supported."
        )

    def suggest_correction(self) -> str:
        return f"Express the pose in its default frame and drop the '{self.attribute}' attribute."


@dataclass
class UnsupportedAxisReference(ParsingError):
    """
    Raised when a joint axis is expressed in a frame other than the joint frame.
    """

    joint_name: str = field(kw_only=True)
    """
    The name of the joint whose axis could not be interpreted.
    """

    reference: str = field(kw_only=True)
    """
    The frame the axis is expressed in.
    """

    def error_message(self) -> str:
        return (
            f"Axis of joint '{self.joint_name}' is expressed in '{self.reference}', "
            f"but only the joint frame is supported."
        )

    def suggest_correction(self) -> str:
        return "Express the axis in the joint frame."

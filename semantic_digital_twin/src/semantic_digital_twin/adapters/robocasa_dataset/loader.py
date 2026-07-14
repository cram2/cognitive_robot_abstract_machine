from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from typing_extensions import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.robocasa_dataset.exceptions import (
    RoboCasaApplianceNotFoundError,
    RoboCasaObjectAssetsNotFoundError,
    RoboCasaObjectHasNoCollisionError,
    RoboCasaObjectInstanceIndexError,
    RoboCasaTaskNotFoundError,
)
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaKitchenApplianceCategory,
    RoboCasaKitchenApplianceResolver,
    RoboCasaObjectCategory,
    RoboCasaObjectResolver,
)
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.semantic_annotations.mixins import (
    HasDoors,
    HasDrawers,
    HasHandle,
)
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.utils import camel_case_split
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

logger = logging.getLogger(__name__)

try:
    import robosuite
    import yaml
    from robocasa.environments.kitchen.kitchen import Kitchen
    from robocasa.models.scenes.kitchen_arena import KitchenArena
    from robocasa.models.scenes import scene_builder, scene_registry
    from robosuite.environments.base import REGISTERED_ENVS
    from robosuite.models.tasks import ManipulationTask
except ImportError:
    logger.warning(
        "robocasa/robosuite are required for RoboCasa dataset loading. Install robosuite from git "
        "('pip install git+https://github.com/ARISE-Initiative/robosuite.git') and then robocasa "
        "('pip install -e .' from a clone of https://github.com/robocasa/robocasa), then run "
        "'python -m robocasa.scripts.download_kitchen_assets' to fetch the fixture/object assets."
    )


@dataclass
class RoboCasaTask:
    """
    A RoboCasa manipulation task bound to the
    :class:`~semantic_digital_twin.world.World` it is defined over.

    RoboCasa expresses a task as an environment class that, on top of a kitchen scene, places the
    objects to manipulate, describes the activity in natural language, and positions the robot. This
    wrapper carries the parsed world together with those task-level facts, so a task and the scene it
    plays out in can be passed around as a single value.

    .. note::
        RoboCasa's robot is only used to compose the scene and is not represented here: the robot is
        owned by ``semantic_digital_twin`` and is expected to be spawned separately at
        :attr:`robot_base_pose`.
    """

    world: World
    """
    The kitchen scene the task is defined over, with semantic annotations attached to
    its appliances and manipulated objects.
    """

    instruction: str
    """
    The natural-language description of the task (for example ``"Press the start button
    on the microwave."``).
    """

    manipulated_objects: List[Body]
    """
    The bodies in :attr:`world` that the task requires the robot to interact with.
    """

    robot_base_pose: HomogeneousTransformationMatrix
    """
    The pose, relative to the world root, at which RoboCasa intends the robot to start,
    for spawning the ``semantic_digital_twin``-owned robot.
    """


def _mjcf_document_from_element_copy(element: ET.Element) -> str:
    """
    Wrap a copy of a single MJCF XML element (for example one kitchen appliance's
    geometry) into a minimal standalone MJCF document so it can be parsed on its own. A
    copy is used so the original element is not reparented out of whatever tree RoboCasa
    still holds it in.

    :param element: The XML element to wrap, typically a RoboCasa fixture's underlying
        body element.
    :return: The MJCF document as a string.
    """
    root = ET.Element("mujoco")
    worldbody = ET.SubElement(root, "worldbody")
    worldbody.append(copy.deepcopy(element))
    return ET.tostring(root, encoding="unicode")


def _category_from_class_name(class_name: str) -> str:
    """
    Convert a RoboCasa ``Fixture`` subclass name (upper camel case, for example
    ``"HingeCabinet"``) into a lower snake case category string (for example
    ``"hinge_cabinet"``) suitable for :meth:`~semantic_digital_twin.adapters.robocasa_da
    taset.semantics.RoboCasaCategoryResolver.resolve`.

    :param class_name: The RoboCasa ``Fixture`` subclass name.
    :return: The lower snake case category string.
    """
    return "_".join(token.lower() for token in camel_case_split(class_name))


@dataclass
class RoboCasaDatasetLoader:
    """
    Loader for objects, kitchen appliances, and full kitchen scenes from the RoboCasa dataset
    (https://github.com/robocasa/robocasa).

    RoboCasa composes its assets with robosuite/MuJoCo. This loader drives RoboCasa's own Python
    composition code to build the MJCF for a requested kitchen appliance or kitchen, and parses the
    resulting MJCF with the existing :class:`~semantic_digital_twin.adapters.mjcf.MJCFParser`.

    For this to work, ``robocasa`` and ``robosuite`` (installed from git) must be available, and the
    RoboCasa fixture/object assets must be downloaded via
    ``python -m robocasa.scripts.download_kitchen_assets``.

    .. note::
        RoboCasa does not version its internal Python module layout as a stable public API. The import
        paths used here match the module layout at the time this loader was written; if they no longer
        match the installed version, adjust the module-level imports at the top of this file
        accordingly.
    """

    directory: Path = field(
        default_factory=lambda: Path.home() / "robocasa-assets",
    )
    """
    The directory where the RoboCasa fixture/object assets were downloaded to via
    ``python -m robocasa.scripts.download_kitchen_assets``.
    """

    kitchen_appliance_annotator: RoboCasaKitchenApplianceResolver = field(
        default_factory=RoboCasaKitchenApplianceResolver
    )
    """
    Resolver mapping RoboCasa fixture category names to SemanticAnnotation subclasses.
    """

    object_annotator: RoboCasaObjectResolver = field(
        default_factory=RoboCasaObjectResolver
    )
    """
    Resolver mapping RoboCasa object category names to SemanticAnnotation subclasses.
    """

    self_contained_object_groups: ClassVar[Tuple[str, ...]] = (
        "objaverse",
        "lightwheel",
    )
    """
    RoboCasa object asset groups (subdirectories of ``objects``) whose ``model.xml``
    files reference their textures and meshes by relative paths and can therefore be
    parsed on any machine.

    The
    ``aigen_objs`` group is deliberately excluded: in the published assets its models reference
    textures by absolute paths from the dataset author's machine, so they fail to load elsewhere.
    """

    scene_composition_robot: ClassVar[str] = "PandaOmron"
    """
    The RoboCasa robot used to compose a task scene.

    RoboCasa requires a robot to build a task, but it is stripped from the resulting
    world because ``semantic_digital_twin`` owns the robot.
    """

    robot_body_name_prefixes: ClassVar[Tuple[str, ...]] = (
        "robot",
        "mobilebase",
        "gripper",
        "mount",
    )
    """
    Prefixes of the MJCF body names RoboCasa/robosuite give to the robot and its mounts,
    used to remove the robot from a composed task scene.
    """

    robot_referencing_mjcf_sections: ClassVar[Tuple[str, ...]] = (
        "actuator",
        "sensor",
        "tendon",
        "equality",
        "contact",
        "keyframe",
    )
    """
    Top-level MJCF sections that reference the robot's joints or bodies and must be
    removed together with the robot so the remaining document parses without dangling
    references.
    """

    def load_kitchen(
        self,
        layout_id: LayoutType,
        style_id: StyleType,
        random_number_generator: Optional[numpy.random.Generator] = None,
    ) -> World:
        """
        Compose a full RoboCasa kitchen scene and parse it into a World.

        :param layout_id: A member of ``robocasa.models.scenes.scene_registry.LayoutType``.
        :param style_id: A member of ``robocasa.models.scenes.scene_registry.StyleType``.
        :param random_number_generator: The random number generator used for the (deterministic,
            non-physical) fixture placement within the layout. Defaults to RoboCasa's own default
            generator.
        :return: The composed world, with a SemanticAnnotation attached to each appliance's root body.
        """
        with open(scene_registry.get_layout_path(layout_id)) as layout_file:
            layout_config = yaml.safe_load(layout_file)
        with open(scene_registry.get_style_path(style_id)) as style_file:
            style_config = yaml.safe_load(style_file)

        arena = KitchenArena(
            layout_id=layout_id, style_id=style_id, rng=random_number_generator
        )
        kitchen_appliances = scene_builder.create_fixtures(
            layout_config, style_config, rng=random_number_generator
        )

        task = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[],
            mujoco_objects=list(kitchen_appliances.values()),
        )

        world = MJCFParser.from_xml_string(task.get_xml()).parse()
        self._apply_kitchen_appliance_semantics(world, kitchen_appliances)
        return world

    def load_task(
        self,
        task_name: str,
        layout_id: LayoutType,
        style_id: StyleType,
    ) -> RoboCasaTask:
        """
        Compose a RoboCasa task scene and return it bound to its parsed World.

        RoboCasa builds the task's scene, places the objects to manipulate, and positions the robot
        when the task environment is reset. The robot is removed from the resulting world because
        ``semantic_digital_twin`` owns the robot; its intended start pose is preserved on the
        returned :class:`RoboCasaTask`.

        :param task_name: The name of a registered RoboCasa kitchen task environment (for example
            ``"TurnOnMicrowave"``).
        :param layout_id: A member of ``robocasa.models.scenes.scene_registry.LayoutType``.
        :param style_id: A member of ``robocasa.models.scenes.scene_registry.StyleType``.
        :return: The task bound to the world it is defined over.
        :raises RoboCasaTaskNotFoundError: if ``task_name`` is not a registered RoboCasa kitchen
            task.
        """
        if task_name not in self._available_task_names():
            raise RoboCasaTaskNotFoundError(task_name, self._available_task_names())

        environment = robosuite.make(
            env_name=task_name,
            robots=self.scene_composition_robot,
            layout_ids=int(layout_id),
            style_ids=int(style_id),
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
        )
        environment.reset()

        episode_metadata = environment.get_ep_meta()
        object_world_poses = self._object_world_poses(environment)
        stripped_document = self._strip_robot_and_bake_object_poses(
            environment.sim.model.get_xml(), object_world_poses
        )

        world = MJCFParser.from_xml_string(stripped_document).parse()
        self._apply_kitchen_appliance_semantics(world, environment.fixtures)
        manipulated_objects = self._apply_task_object_semantics(
            world, environment.object_cfgs, environment.objects
        )

        return RoboCasaTask(
            world=world,
            instruction=episode_metadata["lang"],
            manipulated_objects=manipulated_objects,
            robot_base_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                *episode_metadata["init_robot_base_pos"],
                *episode_metadata["init_robot_base_ori"],
            ),
        )

    @staticmethod
    def _available_task_names() -> List[str]:
        """
        Return the names of the registered RoboCasa kitchen task environments, in sorted
        order.

        :return: The registered kitchen task names.
        """
        return sorted(
            name
            for name, environment_class in REGISTERED_ENVS.items()
            if isinstance(environment_class, type)
            and issubclass(environment_class, Kitchen)
        )

    @staticmethod
    def _object_world_poses(environment: Any) -> Dict[str, Tuple[Any, Any]]:
        """
        Read the world pose of each manipulated object's root body from the reset
        simulation.

        RoboCasa places manipulated objects by writing their pose into the simulation
        state (they are attached with free joints), so the pose is not present in the
        composed MJCF document and must be read from the simulation and baked back in.

        :param environment: The reset RoboCasa task environment.
        :return: Mapping from each object's root body name to its ``(position,
            orientation)`` world pose, with the orientation as a scalar-first
            quaternion.
        """
        object_world_poses: Dict[str, Tuple[Any, Any]] = {}
        for robocasa_object in environment.objects.values():
            body_id = environment.sim.model.body_name2id(robocasa_object.root_body)
            object_world_poses[robocasa_object.root_body] = (
                environment.sim.data.body_xpos[body_id].copy(),
                environment.sim.data.body_xquat[body_id].copy(),
            )
        return object_world_poses

    @classmethod
    def _strip_robot_and_bake_object_poses(
        cls, mjcf_document: str, object_world_poses: Dict[str, Tuple[Any, Any]]
    ) -> str:
        """
        Remove the robot from a composed task's MJCF document and write the sampled
        world poses of the manipulated objects into it.

        The robot bodies and the MJCF sections referencing them are removed so the
        document parses without a robot. Each manipulated object body is given the pose
        RoboCasa sampled for it, which the composed document does not carry because
        free-jointed objects are placed via the simulation state.

        :param mjcf_document: The composed task's MJCF document.
        :param object_world_poses: Mapping from object root body name to its sampled
            world pose, as returned by :meth:`_object_world_poses`.
        :return: The robot-free MJCF document with object poses baked in.
        """
        root = ET.fromstring(mjcf_document)
        worldbody = root.find("worldbody")
        for body in list(worldbody.findall("body")):
            body_name = body.get("name") or ""
            if body_name.startswith(cls.robot_body_name_prefixes) or body_name.endswith(
                "_eef_target"
            ):
                worldbody.remove(body)

        for section_tag in cls.robot_referencing_mjcf_sections:
            for section in root.findall(section_tag):
                root.remove(section)

        for body in worldbody.iter("body"):
            pose = object_world_poses.get(body.get("name"))
            if pose is None:
                continue
            position, orientation = pose
            body.set("pos", cls._format_vector(position))
            body.set("quat", cls._format_vector(orientation))

        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def _format_vector(vector: Any) -> str:
        """
        Format a numeric vector as a space-separated MJCF attribute value.

        :param vector: The numeric vector to format.
        :return: The space-separated string representation.
        """
        return " ".join(f"{component:.10g}" for component in vector)

    def _apply_task_object_semantics(
        self,
        world: World,
        object_configurations: List[Dict[str, Any]],
        objects: Dict[str, Any],
    ) -> List[Body]:
        """
        Attach a SemanticAnnotation to each manipulated object's body and collect those
        bodies.

        The object's sampled category drives the annotation, falling back to
        NaturalLanguageWithTypeDescription when the category has no matching
        SemanticAnnotation subclass. An object whose body is not present in ``world``
        (RoboCasa's object sampling is stochastic per reset) is skipped.

        :param world: The world the task scene was parsed into.
        :param object_configurations: The task's object configurations, as produced by
            RoboCasa's ``env.object_cfgs``.
        :param objects: Mapping from object name to the RoboCasa object instance, as
            produced by RoboCasa's ``env.objects``.
        :return: The annotated manipulated object bodies.
        """
        manipulated_objects: List[Body] = []
        for object_configuration in object_configurations:
            robocasa_object = objects.get(object_configuration["name"])
            if robocasa_object is None:
                continue
            body = self._find_body(world, robocasa_object.root_body)
            if body is None:
                continue
            self._attach_semantic_annotation(
                world, body, str(object_configuration["info"]["cat"])
            )
            manipulated_objects.append(body)
        return manipulated_objects

    def load_kitchen_appliance(
        self,
        category: RoboCasaKitchenApplianceCategory,
        style_id: Optional[StyleType] = None,
    ) -> World:
        """
        Load a single RoboCasa kitchen appliance (for example a cabinet or a microwave)
        as a standalone world.

        The appliance is taken from a composed kitchen so that it carries the size, model, and
        texture configuration RoboCasa authored for it: the first fixture whose category matches
        ``category`` across the available kitchen layouts is used. RoboCasa realises a single
        semantic category with several concrete fixture variants (for example a cabinet as a hinged,
        single, open, panel, or housing cabinet); any of them satisfies the request.

        :param category: The appliance category to load.
        :param style_id: The visual style to compose the source kitchen with. Defaults to RoboCasa's
            first style.
        :return: The loaded world, with a SemanticAnnotation attached to the appliance's root body.
        :raises RoboCasaApplianceNotFoundError: if no layout contains a fixture of ``category``.
        """
        appliance = self._find_configured_appliance(category, style_id)
        world = MJCFParser.from_xml_string(appliance.get_xml()).parse()
        self._apply_kitchen_appliance_semantics(world, {appliance.name: appliance})
        return world

    def _find_configured_appliance(
        self,
        category: RoboCasaKitchenApplianceCategory,
        style_id: Optional[StyleType],
    ) -> Any:
        """
        Search the RoboCasa kitchen layouts for the first fully configured fixture whose
        category matches ``category``. Composing a kitchen is what gives each fixture
        the size, model, and texture configuration that standalone fixture construction
        lacks.

        :param category: The appliance category to search for.
        :param style_id: The visual style to compose candidate kitchens with, or None
            for the default.
        :return: The matching RoboCasa fixture instance.
        :raises RoboCasaApplianceNotFoundError: if no layout contains a matching
            fixture.
        """
        target_annotation_class = (
            self.kitchen_appliance_annotator.category_to_annotation_class[category]
        )
        style = (
            style_id if style_id is not None else next(iter(scene_registry.StyleType))
        )
        with open(scene_registry.get_style_path(style)) as style_file:
            style_config = yaml.safe_load(style_file)

        for layout in self._kitchen_layouts():
            with open(scene_registry.get_layout_path(layout)) as layout_file:
                layout_config = yaml.safe_load(layout_file)
            for appliance in scene_builder.create_fixtures(
                layout_config, style_config
            ).values():
                appliance_category = _category_from_class_name(type(appliance).__name__)
                if (
                    self.kitchen_appliance_annotator.resolve(appliance_category)
                    is target_annotation_class
                ):
                    return appliance
        raise RoboCasaApplianceNotFoundError(category)

    @staticmethod
    def _kitchen_layouts() -> List[LayoutType]:
        """
        Return the concrete RoboCasa kitchen layouts, excluding the aggregate selectors
        (such as ``ALL`` or ``TRAIN``) that do not denote a single kitchen.

        :return: The concrete layouts, in registry order.
        """
        return [
            layout
            for layout in scene_registry.LayoutType
            if layout.name.startswith("LAYOUT")
        ]

    def load_object(
        self, category: RoboCasaObjectCategory, instance_index: int = 0
    ) -> World:
        """
        Load a single RoboCasa object as a standalone world.

        :param category: The object category, a key of
            ``robocasa.models.objects.kitchen_objects.OBJ_CATEGORIES``. RoboCasa groups its object
            assets by source (for example ``objaverse``); only the self-contained groups listed in
            :attr:`self_contained_object_groups` are searched.
        :param instance_index: Which of the category's downloaded asset instances to load.
        :return: The loaded world, with a SemanticAnnotation attached to the object's root body.
        """
        objects_directory = self.directory / "objects"
        model_files = sorted(
            model_file
            for group in self.self_contained_object_groups
            for model_file in (objects_directory / group).glob(
                f"{category}/**/model.xml"
            )
        )
        if not model_files:
            raise RoboCasaObjectAssetsNotFoundError(
                category, self.self_contained_object_groups, objects_directory
            )
        if instance_index >= len(model_files):
            raise RoboCasaObjectInstanceIndexError(
                category,
                instance_index,
                len(model_files),
                self.self_contained_object_groups,
                objects_directory,
            )

        world = MJCFParser(str(model_files[instance_index])).parse()
        self._apply_object_semantics(world, category)
        return world

    def _apply_kitchen_appliance_semantics(
        self, world: World, kitchen_appliances: Dict[str, Any]
    ) -> None:
        """
        Attach a SemanticAnnotation to the root body of each kitchen appliance, using
        the appliance annotator where the appliance's category matches a known
        SemanticAnnotation subclass, and falling back to
        NaturalLanguageWithTypeDescription otherwise.

        :param world: The world the appliances were parsed into.
        :param kitchen_appliances: Mapping from appliance name to the RoboCasa
            ``Fixture`` instance it was built from.
        """
        for appliance_name, kitchen_appliance in kitchen_appliances.items():
            body = self._find_body(world, appliance_name)
            if body is None:
                continue
            category = _category_from_class_name(type(kitchen_appliance).__name__)
            self._attach_semantic_annotation(world, body, category)

    def _apply_object_semantics(
        self, world: World, category: RoboCasaObjectCategory
    ) -> None:
        """
        Attach a SemanticAnnotation to the root body of a loaded object.

        The object's own body is the first body in the world with collision geometry:
        MJCFParser.parse() always creates an empty placeholder root body named after the
        MJCF worldbody, distinct from the loaded content.

        :param world: The world the object was parsed into.
        :param category: The RoboCasa object category the object belongs to.
        """
        bodies_with_collision = world.bodies_with_collision
        if not bodies_with_collision:
            raise RoboCasaObjectHasNoCollisionError(category)
        self._attach_semantic_annotation(world, bodies_with_collision[0], category)

    def _attach_semantic_annotation(
        self, world: World, body: Body, category: str
    ) -> None:
        """
        Attach the SemanticAnnotation matching ``category`` to ``body``, falling back to
        NaturalLanguageWithTypeDescription if no matching SemanticAnnotation subclass is
        known, and attaching any handle/door/drawer sub-part annotations found under
        ``body``.

        :param world: The world ``body`` belongs to.
        :param body: The body to annotate.
        :param category: The RoboCasa appliance or object category of ``body``.
        """
        annotation_class = self.kitchen_appliance_annotator.resolve(
            category
        ) or self.object_annotator.resolve(category)

        with world.modify_world():
            if annotation_class is not None:
                annotation = annotation_class(root=body)
            else:
                annotation = NaturalLanguageWithTypeDescription(
                    root=body, description=category, type_description=category
                )
            world.add_semantic_annotation(annotation)
            self._attach_sub_part_annotations(world, annotation, body)

    def _attach_sub_part_annotations(
        self, world: World, parent_annotation: SemanticAnnotation, parent_body: Body
    ) -> None:
        """
        Detect handle/door/drawer bodies already present under a kitchen appliance's
        root body (RoboCasa appliances like cabinets ship these as real articulated sub-
        bodies in their own MJCF, not something this adapter synthesizes) and attach
        them as parts of the nearest enclosing SemanticAnnotation by their RoboCasa
        body-naming convention (mirroring the naming-convention detection
        ``adapters/procthor/procthor_pipelines.py`` already uses for ProcTHOR dressers).

        Each match is attached via :meth:`~semantic_digital_twin.semantic_annotations.mixins.PartWholeRelationship.add`,
        the framework's normal part-whole mechanism, recursing into each direct child so that, for
        example, a handle nested inside a door is attached to the door's annotation rather than the
        enclosing cabinet's. This is safe: a sub-body is only ever offered to its own direct
        kinematic parent, so ``World.move_branch`` (invoked by ``add()``'s default mount strategy) is
        a no-op re-parent that leaves the body's real connection type and degree of freedom
        untouched.

        :param world: The world ``parent_body`` belongs to.
        :param parent_annotation: The SemanticAnnotation of ``parent_body`` to attach newly found
            direct sub-parts to.
        :param parent_body: The body to search direct children of for sub-part bodies.
        """
        for child_body in parent_body.child_kinematic_structure_entities:
            if not isinstance(child_body, Body):
                continue
            child_name = child_body.name.name.lower()
            child_annotation = parent_annotation

            if (
                "handle" in child_name
                and isinstance(parent_annotation, HasHandle)
                and parent_annotation.handle is None
            ):
                child_annotation = Handle(root=child_body)
                world.add_semantic_annotation(child_annotation)
                parent_annotation.add(child_annotation)
            elif "door" in child_name and isinstance(parent_annotation, HasDoors):
                child_annotation = Door(root=child_body)
                world.add_semantic_annotation(child_annotation)
                parent_annotation.add(child_annotation)
            elif "drawer" in child_name and isinstance(parent_annotation, HasDrawers):
                child_annotation = Drawer(root=child_body)
                world.add_semantic_annotation(child_annotation)
                parent_annotation.add(child_annotation)

            self._attach_sub_part_annotations(world, child_annotation, child_body)

    @staticmethod
    def _find_body(world: World, name: str) -> Optional[Body]:
        """
        Look up a body by name, returning None instead of raising if it is not present.

        Falls back to the first body whose name starts with ``name`` if no exact match
        exists, since robosuite may rename a merged object's root body (for example to
        ``f"{name}_main"``).

        :param world: The world to search.
        :param name: The name of the body to look up.
        :return: The body, or None if no matching body exists in the world.
        """
        try:
            return world.get_body_by_name(name)
        except WorldEntityNotFoundError:
            pass

        matching_bodies = [
            body for body in world.bodies if body.name.name.startswith(name)
        ]
        return matching_bodies[0] if matching_bodies else None

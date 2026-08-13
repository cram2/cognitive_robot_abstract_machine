import copy
import dataclasses
import importlib
import logging
import uuid
from collections.abc import Iterable

import numpy as np
import open3d as o3d
from trimesh import Trimesh
from typing_extensions import Callable, List, Protocol, Dict, Any, Optional, Set, Self

import robokudo.cas
from krrood.utils import recursive_subclasses
from robokudo.defs import PACKAGE_NAME
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
)
from robokudo.types.cv import TSDFAnnotation
from robokudo.utils.annotator_helper import get_cam_to_world_transform_matrix
from robokudo.utils.comparators import (
    TranslationComparator,
    HistogramComparator,
    SemanticColorComparator,
    RoiComparator,
    AdditionalDataComparator,
    BboxComparator,
    ClassificationComparator,
)
from robokudo.utils.transform import (
    get_transform_matrix_from_q,
    get_translation_from_transform_matrix,
    get_quaternion_from_transform_matrix,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Shape, Color, Scale, Box
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)
from semantic_digital_twin.world_description.world_modification import (
    AddConnectionModification,
    AddDegreeOfFreedomModification,
    AddSemanticAnnotationModification,
    RemoveSemanticAnnotationModification,
    RemoveConnectionModification,
    RemoveDegreeOfFreedomModification,
    AddKinematicStructureEntityModification,
    RemoveKinematicStructureEntityModification,
    WorldModification,
)


@dataclasses.dataclass
class Object:
    """A wrapper around perception data that is used to store data used for object comparison."""

    data: Dict[str, Any]
    """The actual data that is used for comparison."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Object):
            return False
        for key, value in self.data.items():
            if key not in other.data:
                return False
            elif isinstance(value, Iterable) and any(
                a != b for a, b in zip(value, other.data[key])
            ):
                return False
            elif value != other.data[key]:
                return False
        return True


@dataclasses.dataclass
class TrackedObject:
    obj: Object
    """Object that stores relevant perception data for comparison with other objects."""

    body: Body
    """Reference to the body representing the object in the world."""

    semantic_annotations: List[SemanticAnnotation]
    """References to semantic annotations attached to the bodies."""

    conns: List[Connection6DoF]
    """References to the connections used by the bodies."""

    uid: uuid.UUID = dataclasses.field(default_factory=lambda: uuid.uuid4())
    """Unique identifier for the tracked object."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrackedObject):
            return False
        obj_eq = self.obj == other.obj
        body_eq = self.body == other.body

        if len(self.semantic_annotations) != len(other.semantic_annotations):
            semantic_annotations_eq = False
        else:
            semantic_annotations_eq = all(
                a == b
                for a, b in zip(self.semantic_annotations, other.semantic_annotations)
            )

        if len(self.conns) != len(other.conns):
            conns_eq = False
        else:
            conns_eq = all(a == b for a, b in zip(self.conns, other.conns))

        return obj_eq and body_eq and semantic_annotations_eq and conns_eq


class AddCollisionCommand(WorldModification):
    def __init__(self, body: Body, new_collision: Shape) -> None:
        """Instantiate a new AddCollisionCommand.

        :param body: Body to add or remove the collision to or from.
        :param new_collision: Collision to add or remove to or from the body.
        """

        self.body = body
        """The body to add the collision to."""

        self.new_collision = new_collision
        """The collision to add to the body."""

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> Self:
        return cls(
            body=kwargs["body"],
            new_collision=kwargs["new_collision"],
        )

    def apply(self, world: World) -> None:
        self.body.collision.append(self.new_collision)

    def undo(self, world: World) -> None:
        self.body.collision.shapes.remove(self.new_collision)


class UpdateCollisionCommand(WorldModification):
    def __init__(self, old_collision: Shape, new_collision: Shape) -> None:
        """Instantiate a new UpdateCollisionCommand.

        :param old_collision: Collision to update from.
        :param new_collision: Collision to update to.
        """

        self.collision = old_collision
        """The collision to update."""

        self.old_collision = copy.deepcopy(old_collision)
        """A deep copy of the old collision for undo."""

        self.new_collision = new_collision
        """The new collision to update to."""

        if type(self.old_collision) != type(self.new_collision):
            raise ValueError("cannot update collision to different shape type")

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> Self:
        return cls(
            old_collision=kwargs["old_collision"],
            new_collision=kwargs["new_collision"],
        )

    def apply(self, world: World) -> None:
        old_fields = dataclasses.fields(self.old_collision)
        for field in old_fields:
            old_value = getattr(self.old_collision, field.name)
            new_value = getattr(self.new_collision, field.name)

            if old_value != new_value:
                setattr(self.collision, field.name, new_value)

    def undo(self, world: World) -> None:
        old_fields = dataclasses.fields(self.old_collision)
        for field in old_fields:
            old_value = getattr(self.old_collision, field.name)
            new_value = getattr(self.new_collision, field.name)

            if old_value != new_value:
                setattr(self.collision, field.name, old_value)


class RemoveCollisionCommand(WorldModification):
    def __init__(self, body: Body, old_collision: Shape) -> None:
        """Instantiate a new RemoveCollisionCommand.

        :param body: Body to remove the collision from.
        :param old_collision: Collision to remove from the body.
        """

        self.body = body
        """The body to remove the collision from."""

        self.old_collision = old_collision
        """The collision to remove from the body."""

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> Self:
        return cls(
            body=kwargs["body"],
            old_collision=kwargs["old_collision"],
        )

    def apply(self, world: World) -> None:
        self.body.collision.shapes.remove(self.old_collision)

    def undo(self, world: World) -> None:
        self.body.collision.append(self.old_collision)


class WorldDiff(Protocol):
    def apply(self) -> None:
        """Apply the diff to the word stored in the adapter."""
        pass

    def undo(self) -> None:
        """Undo the diff in the world stored in the adapter."""
        pass


class AddObjectDiff:
    def __init__(
        self, adapter: "SemanticDigitalTwinAdapter", new_object: Object
    ) -> None:
        """Create a new AddObjectDiff instance.

        :param adapter: The SemanticDigitalTwinAdapter instance that is used to store the diff.
        :param new_object: The new object to add to the world.
        """
        self.adapter = adapter
        """SemDT adapter instance that is used to store the diff."""

        self.new_object = new_object
        """The newly created object data."""

        self.tracked_object = self.adapter.object_to_tracked_object(new_object)
        """A tracked object created from the newly created object data."""

        self.commands: List[WorldModification] = list()
        """The commands to apply with this diff."""

        self.commands.append(
            AddKinematicStructureEntityModification(
                kinematic_structure_entity=self.tracked_object.body
            )
        )

        conn_name = PrefixedName(
            name=f"{self.adapter.root.name.name}_{self.tracked_object.body.name.name}"
        )

        dofs = {}
        for name in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
            prefixed_name = PrefixedName(name=name, prefix=conn_name.name)
            dof = DegreeOfFreedom(name=prefixed_name)

            dofs[name] = dof

            self.commands.append(AddDegreeOfFreedomModification(degree_of_freedom=dof))

        conn = Connection6DoF(
            parent=self.adapter.root,
            child=self.tracked_object.body,
            x_id=dofs["x"].id,
            y_id=dofs["y"].id,
            z_id=dofs["z"].id,
            qx_id=dofs["qx"].id,
            qy_id=dofs["qy"].id,
            qz_id=dofs["qz"].id,
            qw_id=dofs["qw"].id,
        )
        self.commands.append(AddConnectionModification(connection=conn))

        for semantic_annotation in self.tracked_object.semantic_annotations:
            self.commands.append(
                AddSemanticAnnotationModification(
                    semantic_annotation=semantic_annotation
                )
            )

    def apply(self) -> None:
        with self.adapter.world.modify_world():
            for command in self.commands:
                command.apply(self.adapter.world)
        self.adapter.tracked_objects.append(self.tracked_object)

    def undo(self) -> None:
        with self.adapter.world.modify_world():
            for command in self.commands:
                command.undo(self.adapter.world)
        self.adapter.tracked_objects.remove(self.tracked_object)


class UpdateObjectDiff:
    def __init__(
        self,
        adapter: "SemanticDigitalTwinAdapter",
        old_object: TrackedObject,
        new_object: Object,
    ) -> None:
        """Create a new UpdateObjectDiff instance.

        :param adapter: The SemanticDigitalTwinAdapter instance that is used to store the diff.
        :param old_object: The old object to update.
        :param new_object: The new object to update the old object to.
        """

        self.adapter = adapter
        """SemDT adapter instance used to store the diff."""

        self.old_tracked_object = old_object
        """The tracked object instance that will be updated."""

        self.old_object = copy.deepcopy(self.old_tracked_object.obj)
        """A copy of the old object data that will be overwritten in the update."""

        self.new_object = new_object
        """The new object data that will be updated to."""

        self.new_tracked_object = self.adapter.object_to_tracked_object(new_object)
        """The new object data as a tracked object for easier diff creation."""

        self.commands: List[WorldModification] = list()
        """The commands to apply with this diff."""

        # Assume a single simple body as collision for perceived, dynamic objects
        if (
            len(self.old_tracked_object.body.collision) == 0
            and len(self.new_tracked_object.body.collision) == 1
        ):
            # Add the collision
            self.commands.append(
                AddCollisionCommand(
                    self.old_tracked_object.body,
                    self.new_tracked_object.body.collision[0],
                )
            )
        elif (
            len(self.old_tracked_object.body.collision) == 1
            and len(self.new_tracked_object.body.collision) == 0
        ):
            # Remove the collision
            self.commands.append(
                RemoveCollisionCommand(
                    self.old_tracked_object.body,
                    self.old_tracked_object.body.collision[0],
                )
            )
        elif (
            len(self.old_tracked_object.body.collision) == 1
            and len(self.new_tracked_object.body.collision) == 1
        ):
            # Update the collision
            self.commands.append(
                UpdateCollisionCommand(
                    self.old_tracked_object.body.collision[0],
                    self.new_tracked_object.body.collision[0],
                )
            )

    def apply(self) -> None:
        with self.adapter.world.modify_world():
            for command in self.commands:
                command.apply(self.adapter.world)
        self.old_tracked_object.obj = self.new_object

    def undo(self) -> None:
        with self.adapter.world.modify_world():
            for command in self.commands:
                command.undo(self.adapter.world)
        self.old_tracked_object.obj = self.old_object


class RemoveObjectDiff:
    def __init__(
        self, adapter: "SemanticDigitalTwinAdapter", old_object: TrackedObject
    ) -> None:
        """Create a new RemoveObjectDiff instance.

        :param adapter: The SemanticDigitalTwinAdapter instance that is used to store the diff.
        :param old_object: The object to remove from the world.
        """

        self.adapter = adapter
        """The SemDT adapter instance used to store the diff."""

        self.old_object = old_object
        """The object to remove."""

        self.commands: List[WorldModification] = list()
        """The commands to be executed by this diff."""

        self.commands.append(
            RemoveKinematicStructureEntityModification(
                kinematic_structure_id=old_object.body.id
            )
        )

        for connection in self.old_object.conns:
            self.commands.append(
                RemoveConnectionModification(
                    parent_id=connection.parent.id, child_id=old_object.body.id
                )
            )
            for dof in connection.dofs:
                self.commands.append(RemoveDegreeOfFreedomModification(dof_id=dof.id))

        for semantic_annotation in self.old_object.semantic_annotations:
            self.commands.append(
                RemoveSemanticAnnotationModification(
                    semantic_annotation=semantic_annotation
                )
            )

    def apply(self) -> None:
        with self.adapter.world.modify_world():
            for command in self.commands:
                command.apply(self.adapter.world)
        self.adapter.tracked_objects.remove(self.old_object)

    def undo(self) -> None:
        with self.adapter.world.modify_world():
            for command in self.commands:
                command.undo(self.adapter.world)
        self.adapter.tracked_objects.append(self.old_object)


class SemanticDigitalTwinAdapter:
    """Class to convert RoboKudo concepts to the SemanticWorld."""

    def __init__(
        self,
        cas_fn: Callable[..., robokudo.cas.CAS],
        urdf_path: Optional[str] = None,
        semantic_annotation_sources: Optional[List] = None,
    ) -> None:
        """Create a SemanticDigitalTwinAdapter instance.

        :param cas_fn: Callable that returns a cas instance when called.
        :param urdf_path: Optional path to a URDF file to load the world from.
        :param semantic_annotation_sources: List of modules that should be used to search semantic annotations in
        """

        if semantic_annotation_sources is None:
            semantic_annotation_sources = [
                "semantic_digital_twin.semantic_annotations.semantic_annotations"
            ]
        for source in semantic_annotation_sources:
            importlib.import_module(source)

        self.world = (
            World()
            if urdf_path is None
            else URDFParser.from_file(file_path=urdf_path).parse()
        )
        """World instance that will be modified by this adapter."""

        if urdf_path is not None:
            self.root = self.world.root
            self.world.validate()
        else:
            self.root = Body(name=PrefixedName(name="map"))

        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(self.root)

        self.rk_logger = logging.getLogger(PACKAGE_NAME)
        """Logger instance for this adapter."""

        self.cas_fn = cas_fn
        """Callable that returns a cas instance when called."""

        self.tracked_objects: list[TrackedObject] = list()
        """List of world objects currently tracked by the adapter."""

        self.comparators = {
            "translation_vector": TranslationComparator(weight=0.4, max_distance=0.5),
            "class": ClassificationComparator(weight=0.4),
            "bbox": BboxComparator(weight=0.2),
            "color_histogram": HistogramComparator(weight=0.3),
            "semantic_color": SemanticColorComparator(weight=0.2),
            "roi": RoiComparator(weight=0.2),
            "oh_roi": RoiComparator(weight=0.4),
        }
        """Mapping of data keys to comparators that are used to compare objects."""

        self.semantic_color_to_rgb = {
            "red": Color(R=1.0, G=0.0, B=0.0, A=1.0),
            "yellow": Color(R=1.0, G=1.0, B=0.0, A=1.0),
            "green": Color(R=0.0, G=1.0, B=0.0, A=1.0),
            "cyan": Color(R=0.0, G=1.0, B=1.0, A=1.0),
            "blue": Color(R=0.0, G=0.0, B=1.0, A=1.0),
            "magenta": Color(R=1.0, G=0.0, B=1.0, A=1.0),
            "white": Color(R=1.0, G=1.0, B=1.0, A=1.0),
            "black": Color(R=0.2, G=0.2, B=0.2, A=1.0),
            "grey": Color(R=0.5, G=0.5, B=0.5, A=1.0),
        }
        """Mapping of color names to actual color values used for visuals in created objects."""

    def compute_diffs(self, new_objects: list[Object]) -> List[WorldDiff]:
        """Compute a list of diffs between the current tracked objects and the novel objects provided.

        :param new_objects: List of new objects to compare to the current tracked objects.
        :return: List of diffs between the current tracked objects and the novel objects provided.
        """

        diffs: List[WorldDiff] = []
        # Old objects that were already matched to a new object
        matched_objects: set[uuid.UUID] = set()
        for new_object in new_objects:
            best_matching_object = None
            best_matching_confidence = -float("inf")

            for old_object in self.tracked_objects:
                if old_object.uid in matched_objects:
                    continue

                conf = self.compute_obj_diff(old_object.obj, new_object)
                if conf > best_matching_confidence:
                    best_matching_object = old_object
                    best_matching_confidence = conf

            if best_matching_object is not None and best_matching_confidence > 0.0:
                diffs.append(UpdateObjectDiff(self, best_matching_object, new_object))

                matched_objects.add(best_matching_object.uid)
            else:
                diffs.append(AddObjectDiff(self, new_object))

        for obj in self.tracked_objects:
            if obj.uid not in matched_objects:
                diffs.append(RemoveObjectDiff(self, obj))
        return diffs

    def compute_obj_diff(self, old_object: Object, new_object: Object) -> float:
        """Compute similarity value between a known object and a novel object.

        :param old_object: The known object to compare to.
        :param new_object: The novel object to check for similarity with.
        """

        old_data, new_data = old_object.data, new_object.data
        old_keys, new_keys = set(old_data.keys()), set(new_data.keys())
        comparable_data = new_keys & old_keys
        if len(comparable_data) == 0:
            return 0.0

        total_similarity = 0.0
        total_weight = 0.0

        for key in comparable_data:
            comparator = self.comparators.get(key, AdditionalDataComparator(1.0))
            similarity = comparator.compute_similarity(old_data[key], new_data[key])

            total_similarity += comparator.weight * similarity
            total_weight += comparator.weight

        confidence = total_similarity / total_weight
        return confidence

    def object_to_tracked_object(self, obj: Object) -> TrackedObject:
        """Creates a TrackedObject from a RoboKudo object.

        :param obj: Object to wrap into a TrackedObject.
        :return: The object wrapped into a TrackedObject instance.
        """

        body = Body()

        semantic_annotations: List[SemanticAnnotation] = list()

        if "semantic_color" in obj.data:
            color = self.semantic_color_to_rgb[obj.data["semantic_color"].color]
        else:
            color = Color(R=0.5, G=0.5, B=0.5, A=0.5)

        if "bbox" in obj.data:
            bb: BoundingBox3DAnnotation = obj.data["bbox"]

            pose_mat = get_transform_matrix_from_q(
                bb.pose.rotation, bb.pose.translation
            )
            cam_to_world_transform = get_cam_to_world_transform_matrix(self.cas_fn())
            pose_in_world_mat = np.matmul(cam_to_world_transform, pose_mat)

            rotation = list(get_quaternion_from_transform_matrix(pose_in_world_mat))
            translation = list(get_translation_from_transform_matrix(pose_in_world_mat))

            origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
                pos_x=translation[0],
                pos_y=translation[1],
                pos_z=translation[2],
                quat_x=rotation[0],
                quat_y=rotation[1],
                quat_z=rotation[2],
                quat_w=rotation[3],
                reference_frame=self.root,
            )

            scale = Scale(x=bb.x_length, y=bb.y_length, z=bb.z_length)

            body.visual.append(Box(color=color, origin=origin, scale=scale))

        if "tsdf" in obj.data:
            volume_an: TSDFAnnotation = obj.data["tsdf"]
            mesh: o3d.geometry.TriangleMesh = volume_an.volume.extract_triangle_mesh()

            pose_mat = volume_an.transform
            cam_to_world_transform = get_cam_to_world_transform_matrix(self.cas_fn())
            pose_in_world_mat = np.matmul(cam_to_world_transform, pose_mat)

            rotation = list(get_quaternion_from_transform_matrix(pose_in_world_mat))
            translation = list(get_translation_from_transform_matrix(pose_in_world_mat))

            origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
                pos_x=translation[0],
                pos_y=translation[1],
                pos_z=translation[2],
                quat_x=rotation[0],
                quat_y=rotation[1],
                quat_z=rotation[2],
                quat_w=rotation[3],
                reference_frame=self.root,
                child_frame=body,
            )

            obj_trimesh = TriangleMesh(
                origin=origin,
                scale=Scale(1.0, 1.0, 1.0),
                mesh=Trimesh(
                    vertices=mesh.vertices,
                    vertex_colors=mesh.vertex_colors,
                    faces=mesh.triangles,
                    face_normals=mesh.triangle_normals,
                ),
            )

            body.visual.append(obj_trimesh)

        if "class" in obj.data:
            semantic_annotation = self.class_to_semantic_annotation(
                obj.data["class"].classname, root=body
            )
            semantic_annotations.append(semantic_annotation)

        return TrackedObject(
            obj=obj,
            body=body,
            semantic_annotations=semantic_annotations,
            conns=[],
        )

    @staticmethod
    def class_to_semantic_annotation(
        class_name: str, **kwargs: Any
    ) -> SemanticAnnotation:
        """Convert a classification name to a SemanticWorld SemanticAnnotation.

        :param class_name: Class to convert to a semantic annotation.
        :return: SemanticAnnotation instance.
        :raises ValueError: If there is no class name equivalent semantic annotation.
        """

        available_semantic_annotations = recursive_subclasses(SemanticAnnotation)
        if len(available_semantic_annotations) == 0:
            raise ValueError(
                "no semantic_annotations available for conversion from class name"
            )

        class_map: Dict[str, str] = {
            "Cornflakes": "Cereal",
            "Salt": "SaltContainer",
            # "Cup": "Cup",
            # "Milk": "Milk",
            "Bueno": "Candy",
        }

        class_candidates = []
        for semantic_annotation_cls in available_semantic_annotations:
            if (
                class_name in class_map.keys()
                and semantic_annotation_cls.__name__ == class_map[class_name]
            ):
                class_candidates.append(semantic_annotation_cls)
            elif semantic_annotation_cls.__name__ == class_name:
                class_candidates.append(semantic_annotation_cls)

        for class_candidate in class_candidates:
            required_fields: Set[str] = set()
            optional_fields: Set[str] = set()
            for field in dataclasses.fields(class_candidate):
                if (
                    field.default == dataclasses.MISSING
                    and field.default_factory == dataclasses.MISSING
                ):
                    required_fields.add(field.name)
                else:
                    optional_fields.add(field.name)

            provided_fields = set(kwargs.keys())
            provided_optional_fields = provided_fields - required_fields

            # All required fields must be provided, all provided optional fields must be valid
            if all(req in provided_fields for req in required_fields) and all(
                opt in optional_fields for opt in provided_optional_fields
            ):
                return class_candidate(**kwargs)
        raise ValueError(
            f"could not convert class name {class_name} to semantic_annotation, candidates checked: {class_candidates}"
        )

    @staticmethod
    def apply_diffs(diffs: List[WorldDiff]) -> None:
        """Applies all the given diffs to the world stored in the adapter.

        :param diffs: The diffs to apply to the world instance.
        """
        for diff in diffs:
            diff.apply()

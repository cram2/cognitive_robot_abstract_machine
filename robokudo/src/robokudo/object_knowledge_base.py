"""
Object knowledge base for RoboKudo.

This module provides classes for storing and managing knowledge about objects
in RoboKudo. It includes support for object components, features, and their
spatial relationships.
"""

from dataclasses import field, dataclass
from typing_extensions import Dict, List, Any, Optional

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body, SemanticAnnotation
from . import defs


@dataclass
class ObjectKnowledge(defs.Region3DWithName):
    """Knowledge representation for a single object.

    This class extends Region3DWithName to add support for object components
    and features. Each object can have multiple components (physical parts)
    and features (characteristics).
    """

    components: List[Any] = field(default_factory=list)
    """List of component objects that make up this object"""

    features: List[Any] = field(default_factory=list)
    """List of features associated with this object"""

    mesh_ros_package: str = ""
    """ROS Package name where a mesh of this object is located"""

    mesh_relative_path: str = ""
    """Relative path to the actual mesh file. This path is relative to mesh_ros_package!"""

    def is_frame_in_camera_coordinates(self) -> bool:
        """Check whether the object is defined in camera coordinates.

        :return: True if the object is defined in camera coordinates, False otherwise.
        """
        return self.frame is None or self.frame == ""


@dataclass(eq=False)
class PredefinedObject(SemanticAnnotation):
    body: Body = field(default=None)


class BaseObjectKnowledgeBase:
    """Base class for managing object knowledge.

    This class provides functionality to store and manage knowledge about
    different objects. Each object is stored as an ObjectKnowledge instance
    and can be accessed by its name.
    """

    def __init__(self) -> None:
        """Initialize an empty object knowledge base."""

        """Dictionary mapping object names to their knowledge"""
        self.entries: Dict[str, ObjectKnowledge] = dict()

        self.world = World()
        root = Body(name=PrefixedName(name="root", prefix="world"))
        with self.world.modify_world():
            self.world.add_body(body=root)

    def get_predefined_object_bodies(self) -> List[Body]:
        """Get list of pre-defined objects."""
        predefined_object_annotations = self.world.get_semantic_annotations_by_type(PredefinedObject)
        return list([predefined_object.body for predefined_object in predefined_object_annotations])

    def add_entry(self, entry: ObjectKnowledge) -> None:
        """Add a single object knowledge entry.

        :param entry: The object knowledge entry to add
        :raises Exception: If the entry has no name attribute
        """
        if not hasattr(entry, "name"):
            raise Exception("Can't read name from ObjectKnowledge")

        self.entries[entry.name] = entry

    def add_entries(self, entries: List[ObjectKnowledge]) -> None:
        """Add multiple object knowledge entries.

        :param entries: List of object knowledge entries to add
        """
        for entry in entries:
            self.add_entry(entry)

    @staticmethod
    def has_parthood_childs(object_knowledge: ObjectKnowledge) -> bool:
        """Check if an object has any components or features.

        :param object_knowledge: The object knowledge to check
        :return: True if the object has components or features
        """
        return (
            len(object_knowledge.features) > 0 or len(object_knowledge.components) > 0
        )

from __future__ import annotations

import logging
import inspect
import hashlib
import os
import tempfile
from pathlib import Path
from typing_extensions import Generic, Any, ClassVar, Iterator, TypeVar

import time
from datetime import timedelta
from functools import lru_cache
import trimesh
import PIL.ImageFile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag, StrEnum
from types import NoneType

import numpy
import mujoco
from numpy.typing import NDArray
from physics_simulators.mujoco_simulator import MujocoSimulator
from physics_simulators.base_simulator import (
    BaseSimulator,
    SimulatorState,
    SimulatorCallbackResult,
    SimulatorConstraints,
)
from krrood.exceptions import UnboundGenericParameter
from krrood.ormatic.utils import classproperty
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import recursive_subclasses
from scipy.spatial.transform import Rotation
from trimesh.visual import TextureVisuals

from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.unique_entity_names import UniqueEntityNames
from semantic_digital_twin.exceptions import (
    QuaternionConversionError,
    MujocoEntityNotFoundError,
    SimulationAlreadyRunningError,
    SimulationNotStartedError,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Quaternion,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.math import inverse_frame
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    ActiveConnection1DOF,
    FixedConnection,
    Connection6DoF,
    OmniDrive,
    DifferentialDrive,
    WheeledDrive,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Sphere,
    Shape,
    Mesh,
    MeshFileFormat,
    Color,
)
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.world_entity import (
    Region,
    Body,
    GravityCompensation,
    KinematicStructureEntity,
    Connection,
    WorldEntity,
    WorldEntityWithID,
    Actuator,
    PositionServo,
    VelocityServo,
)
from semantic_digital_twin.mixin import (
    SimulatorAttributeName,
    SimulatorAdditionalProperty,
    UniqueSimulatorProperty,
)
from semantic_digital_twin.world_description.world_modification import (
    AddKinematicStructureEntityModification,
    AddActuatorModification,
    AddConnectionModification,
)

logger = logging.getLogger(__name__)


def cas_pose_to_list(pose: HomogeneousTransformationMatrix) -> list[float]:
    """
    Converts a CAS TransformationMatrix to a list of 7 floats (position + quaternion).

    :param pose: The CAS TransformationMatrix to convert.
    :return: A list of 7 floats ([px, py, pz, qw, qx, qy, qz]) representing the position and quaternion.
    """
    pose = pose.evaluate()
    pos = pose[:3, 3]
    rotation_matrix = pose[:3, :3]
    try:
        quat = Rotation.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    except Exception as e:
        raise QuaternionConversionError(rotation_matrix, str(e))
    return [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]


class MujocoElementType(StrEnum):
    """
    A kind of element of a MuJoCo model that an entity of the world is added to a running
    simulation as.
    """

    BODY = "body"
    """
    A body, which a kinematic structure entity becomes.
    """

    JOINT = "joint"
    """
    A joint, which a connection becomes.
    """

    GEOM = "geom"
    """
    A geometry attached to a body, which a shape becomes.
    """

    ACTUATOR = "actuator"
    """
    An actuator, which an actuator of the world becomes.
    """


class MujocoActuatorProperty(StrEnum):
    """
    A property of a MuJoCo actuator that says what the actuator drives.
    """

    TARGET = "target"
    """
    The name of the joint or tendon the actuator drives.
    """

    TRANSMISSION_TYPE = "trntype"
    """
    Whether :attr:`TARGET` names a joint or a tendon.
    """


class GeomVisibilityAndCollisionType(IntEnum):
    """
    Enumeration of geometric visibility and collision attributes.

    - Use VISIBLE_AND_COLLIDABLE_1 or VISIBLE_AND_COLLIDABLE_2 for geometries that should be both visible and collidable. These two values are equivalent and can be used for grouping.
    - Use ONLY_VISIBLE for geometries that are visible but not collidable (URDF: <visual>).
    - Use ONLY_COLLIDABLE for geometries that are collidable but not visible (URDF: <collision>).
    - UNDEFINED_1 and UNDEFINED_2 are placeholders used only when parsing from MuJoCo, and are treated as invisible by default.

    For more information, see:
    https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom
    """

    VISIBLE_AND_COLLIDABLE_1 = 0
    """
    Geometry is both visible and collidable (variant 1).
    """

    VISIBLE_AND_COLLIDABLE_2 = 1
    """
    Geometry is both visible and collidable (variant 2).
    """

    ONLY_VISIBLE = 2
    """
    Geometry is only visible, not collidable.
    """

    ONLY_COLLIDABLE = 3
    """
    Geometry is only collidable, not visible.
    """

    UNDEFINED_1 = 4
    """
    Undefined geometry type (variant 1).
    """

    UNDEFINED_2 = 5
    """
    Undefined geometry type (variant 2).
    """


@dataclass(eq=False)
class MultiSimCamera(SimulatorAdditionalProperty):
    """
    Additional property representing a camera in MultiSim.
    """

    name: str = ""
    """
    The name of the camera.
    """

    body: Any = None
    """
    The body that the camera is attached to. This can be set to the name of the body or a reference to the body object itself.
    """


@dataclass(eq=False)
class MultiSimLight(SimulatorAdditionalProperty):
    """
    Additional property representing a light in MultiSim.
    """

    name: str = ""
    """
    The name of the light.
    """

    body: Body | None = None
    """
    The body the light is attached to.
    """


@dataclass
class InertialConverter:
    """
    A converter to convert inertia representations to diagonal form and update the inertia quaternion accordingly.
    """

    mass: float
    """
    The mass of the body.
    """

    inertia_pos: Point3
    """
    The position of the inertia frame relative to the body frame [x, y, z].
    """

    inertia_quat: Quaternion
    """
    The orientation of the inertia frame relative to the body frame as a quaternion [qw, qx, qy, qz].
    """

    diagonal_inertia: list[float]
    """
    The diagonal inertia tensor in the form [Ixx, Iyy, Izz].
    """

    def __post_init__(self):
        assert self.mass > 0, "Mass must be positive."
        assert len(self.diagonal_inertia) == 3, "Diagonal inertia must have 3 elements."
        assert all(i >= 0 for i in self.diagonal_inertia), (
            "Inertia values must be non-negative."
        )

    @staticmethod
    def _update_quaternion(
        quat: numpy.ndarray, eigenvectors: numpy.ndarray
    ) -> Quaternion:
        """
        Updates the inertia quaternion based on the eigenvectors of the inertia matrix.

        :param quat: The original inertia quaternion [qw, qx, qy, qz].
        :param eigenvectors: The eigenvectors of the inertia matrix.

        :return: The updated inertia quaternion [qw, qx, qy, qz].
        """
        R_orig = Rotation.from_quat(quat, scalar_first=True)  # type: ignore
        R_diag = Rotation.from_matrix(eigenvectors)  # type: ignore
        updated_quat = (R_orig * R_diag).as_quat(scalar_first=True)
        return Quaternion(
            x=updated_quat[1],
            y=updated_quat[2],
            z=updated_quat[3],
            w=updated_quat[0],
        )


@dataclass
class EntityConverter(ABC):
    """
    A converter to convert an entity object (WorldEntity, Shape, Connection) to a dictionary of properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[Any]] = Any
    """
    The type of the entity to convert.
    """

    name_str: str = "name"
    """
    The key for the name property in the output dictionary.
    """

    @classmethod
    def convert(cls, entity: entity_type, **kwargs) -> dict[str, Any]:  # type: ignore
        """
        Converts an entity object to a dictionary of properties for Multiverse simulator.

        :param entity: The object to convert.
        :return: A dictionary of properties.
        """
        for subclass in recursive_subclasses(cls) + [cls]:
            if (
                not inspect.isabstract(subclass)
                and not inspect.isabstract(subclass.entity_type)
                and type(entity) is subclass.entity_type
            ):
                entity_props = subclass()._convert(entity, **kwargs)
                return subclass()._post_convert(entity, entity_props, **kwargs)
        raise NotImplementedError(f"No converter found for entity type {type(entity)}.")

    def _convert(self, entity: entity_type, **kwargs) -> dict[str, Any]:  # type: ignore
        """
        The actual conversion method to be implemented by subclasses.

        :param entity: The object to convert.
        :return: A dictionary of properties, by default containing the name.
        """
        if hasattr(entity, "name") and isinstance(entity.name, PrefixedName):
            resolved_name = entity.name.name
        elif hasattr(entity, "name") and isinstance(entity.name, str) and entity.name:
            # SimulatorAdditionalProperty entities (e.g. MultiSimCamera) carry a plain
            # string name rather than a PrefixedName; honor it if one was given.
            resolved_name = entity.name
        else:
            resolved_name = f"{type(entity).__name__.lower()}_{id(entity)}"
        return {self.name_str: resolved_name}

    @abstractmethod
    def _post_convert(
        self,
        entity: entity_type,
        entity_props: dict[str, Any],
        **kwargs,  # type: ignore
    ) -> dict[str, Any]:
        """
        Post-processes the converted entity properties. This method can be overridden by subclasses to update the properties after conversion.

        :param entity: The object that was converted.
        :param entity_props: The dictionary of properties that was converted.
        :return: The updated dictionary of properties.
        """
        raise NotImplementedError


@dataclass
class KinematicStructureEntityConverter(EntityConverter, ABC):
    """
    Converts a KinematicStructureEntity object to a dictionary of body properties for Multiverse simulator.
    For inheriting classes, the following string attributes must be defined:
    - pos_str: The key for the position property in the output dictionary.
    - quat_str: The key for the quaternion property in the output dictionary.
    """

    entity_type: ClassVar[type[KinematicStructureEntity]] = KinematicStructureEntity
    pos_str: str = "pos"
    quat_str: str = "quat"

    def _convert(self, entity: entity_type, **kwargs) -> dict[str, Any]:
        """
        Converts a KinematicStructureEntity object to a dictionary of body properties for Multiverse simulator.

        :param entity: The KinematicStructureEntity object to convert.
        :return: A dictionary of body properties, by default containing position and quaternion.
        """

        kinematic_structure_entity_props = EntityConverter._convert(self, entity)
        # The simulator joint supplies the variable part, so the static frame must
        # exclude it (see Connection.reference_origin_expression).
        [
            px,
            py,
            pz,
            qx,
            qy,
            qz,
            qw,
        ] = entity.parent_connection.reference_origin_as_position_quaternion().evaluate()[
            0
        ]
        kinematic_structure_entity_pos = [px, py, pz]
        kinematic_structure_entity_quat = [qw, qx, qy, qz]
        kinematic_structure_entity_props.update(
            {
                self.pos_str: kinematic_structure_entity_pos,
                self.quat_str: kinematic_structure_entity_quat,
            }
        )
        return kinematic_structure_entity_props


@dataclass
class BodyConverter(KinematicStructureEntityConverter, ABC):
    """
    Converts a Body object to a dictionary of body properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[WorldEntity]] = Body
    """
    The type of the entity to convert.
    """

    # Attributes for specifying body properties in the Mujoco simulator.

    mass_str: str = "mass"
    """
    The key for the mass property in the output dictionary.
    """

    inertia_pos_str: str = "inertia_pos"
    """
    The key for the inertia position property in the output dictionary.
    """

    inertia_quat_str: str = "inertia_quat"
    """
    The key for the inertia quaternion property in the output dictionary.
    """

    diagonal_inertia_str: str = "diagonal_inertia"
    """
    The key for the diagonal inertia tensor property in the output dictionary.
    """

    def _convert(self, entity: Body, **kwargs) -> dict[str, Any]:
        """
        Converts a Body object to a dictionary of body properties for Multiverse simulator.

        :param entity: The Body object to convert.
        :return: A dictionary of body properties, including additional mass and inertia properties.
        """
        body_props = KinematicStructureEntityConverter._convert(self, entity)
        inertial = entity.inertial
        if inertial is not None:
            mass = inertial.mass
            inertia_pos = inertial.center_of_mass.to_np()[:3]
            inertia = inertial.inertia
            principal_moments, principal_axes = inertia.to_principal_moments_and_axes()
            diagonal_inertia = principal_moments.data
            inertia_quat = principal_axes.to_rotation_matrix().quaternion.to_np()
            inertia_quat[:] = (
                inertia_quat[3],
                inertia_quat[0],
                inertia_quat[1],
                inertia_quat[2],
            )  # Convert from (x, y, z, w) to (w, x, y, z)
            body_props.update(
                {
                    self.mass_str: mass,
                    self.inertia_pos_str: inertia_pos,
                    self.inertia_quat_str: inertia_quat,
                    self.diagonal_inertia_str: diagonal_inertia,
                }
            )
        return body_props


@dataclass
class RegionConverter(KinematicStructureEntityConverter, ABC):
    """
    Converts a Region object to a dictionary of region properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[WorldEntity]] = Region
    """
    The type of the entity to convert.
    """


@dataclass
class ShapeConverter(EntityConverter, ABC):
    """
    Converts a Shape object to a dictionary of shape properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[Shape]] = Shape
    """
    The type of the entity to convert.
    """

    pos_str: str = "pos"
    """
    The key for the shape position property in the output dictionary.
    """

    quat_str: str = "quat"
    """
    The key for the shape quaternion property in the output dictionary.
    """

    rgba_str: str = "rgba"
    """
    The key for the shape RGBA color property in the output dictionary.
    """

    def _convert(self, entity: Shape, **kwargs) -> dict[str, Any]:
        """
        Converts a Shape object to a dictionary of shape properties for Multiverse simulator.

        :param entity: The Shape object to convert.
        :return: A dictionary of shape properties, by default containing position, quaternion, and RGBA color.
        """
        geom_props = EntityConverter._convert(self, entity, **kwargs)
        px, py, pz, qw, qx, qy, qz = cas_pose_to_list(entity.origin)
        geom_pos = [px, py, pz]
        geom_quat = [qw, qx, qy, qz]
        r, g, b, a = (
            entity.color.R,
            entity.color.G,
            entity.color.B,
            entity.color.A,
        )
        geom_color = [r, g, b, a]
        geom_props.update(
            {
                self.pos_str: geom_pos,
                self.quat_str: geom_quat,
                self.rgba_str: geom_color,
            }
        )
        if entity.texture is not None:
            geom_props["texture_file_path"] = entity.texture.file_path
            geom_props["texture_repeat"] = entity.texture.repeat
            geom_props["texture_uniform"] = entity.texture.uniform
        return geom_props


@dataclass
class BoxConverter(ShapeConverter, ABC):
    """
    Converts a Box object to a dictionary of box properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[Box]] = Box


@dataclass
class SphereConverter(ShapeConverter, ABC):
    """
    Converts a Sphere object to a dictionary of sphere properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[Sphere]] = Sphere


@dataclass
class CylinderConverter(ShapeConverter, ABC):
    """
    Converts a Cylinder object to a dictionary of cylinder properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[Cylinder]] = Cylinder


@dataclass
class MeshConverter(ShapeConverter, ABC):
    """
    Converts a Mesh object to a dictionary of mesh properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[Mesh]] = Mesh


@dataclass
class ConnectionConverter(EntityConverter, ABC):
    """
    Converts a Connection object to a dictionary of joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[Connection]] = Connection
    """
    The type of the entity to convert.
    """

    def _convert(self, entity: Connection, **kwargs) -> dict[str, Any]:
        """
        Converts a Connection object to a dictionary of joint properties for Multiverse simulator.

        :param entity: The Connection object to convert.
        :return: A dictionary of joint properties.
        """
        return EntityConverter._convert(self, entity)


@dataclass
class Connection1DOFConverter(ConnectionConverter, ABC):
    """
    Converts an ActiveConnection1DOF object to a dictionary of joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[ActiveConnection1DOF]] = ActiveConnection1DOF
    """
    The type of the entity to convert.
    """

    axis_str: str = "axis"
    """
    The key for the joint axis property in the output dictionary.
    """

    range_str: str = "range"
    """
    The key for the joint range property in the output dictionary.
    """

    pos_str: str = "pos"
    """
    The key for the joint position property in the output dictionary.
    """

    quat_str: str = "quat"
    """
    The key for the joint quaternion property in the output dictionary.
    """

    armature_str: str = "armature"
    """
    The key for the joint armature property in the output dictionary.
    """

    dry_friction_str: str = "dry_friction"
    """
    The key for the joint dry friction property in the output dictionary.
    """

    damping_str: str = "damping"
    """
    The key for the joint damping property in the output dictionary.
    """

    def _convert(self, entity: ActiveConnection1DOF, **kwargs) -> dict[str, Any]:
        """
        Converts an ActiveConnection1DOF object to a dictionary of joint properties for Multiverse simulator.

        :param entity: The ActiveConnection1DOF object to convert.
        :return: A dictionary of joint properties, including additional axis, range, position, quaternion, armature, dry friction, and damping properties.
        """
        joint_props = ConnectionConverter._convert(self, entity)
        dofs = list(entity.dofs)
        assert len(dofs) == 1, "ActiveConnection1DOF must have exactly one DOF."
        dof = dofs[0]
        child_T_connection_transform = entity.connection_T_child_expression.inverse()
        px, py, pz, qw, qx, qy, qz = cas_pose_to_list(child_T_connection_transform)
        joint_pos = [px, py, pz]
        joint_quat = [qw, qx, qy, qz]
        # entity.dof, not the raw dof: it applies this connection's own multiplier and
        # sign, which a negative-multiplier mimic (a gripper's second finger) needs to
        # get its own displayed range rather than the shared raw dof's.
        adjusted_limits = entity.dof.limits
        joint_range = [adjusted_limits.lower.position, adjusted_limits.upper.position]
        if any([r is None for r in joint_range]):
            joint_range = [0, 0]
        joint_props.update(
            {
                self.pos_str: joint_pos,
                self.quat_str: joint_quat,
                self.axis_str: entity.axis.to_np().tolist()[:3],
                self.range_str: joint_range,
                self.armature_str: entity.dynamics.armature,
                self.dry_friction_str: entity.dynamics.dry_friction,
                self.damping_str: entity.dynamics.damping,
            }
        )
        if dof.name.name != joint_props["name"] and dof.name.name != "dof":
            joint_props["equality_joint"] = {
                "joint": dof.name.name,
                "data": [entity.offset, entity.multiplier, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            }
        return joint_props


@dataclass
class ConnectionRevoluteConverter(Connection1DOFConverter, ABC):
    """
    Converts a RevoluteConnection object to a dictionary of revolute joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[RevoluteConnection]] = RevoluteConnection
    """
    The type of the entity to convert.
    """


@dataclass
class ConnectionPrismaticConverter(Connection1DOFConverter, ABC):
    """
    Converts a PrismaticConnection object to a dictionary of prismatic joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[PrismaticConnection]] = PrismaticConnection
    """
    The type of the entity to convert.
    """


@dataclass
class Connection6DOFConverter(ConnectionConverter, ABC):
    """
    Converts a Connection6DoF object to a dictionary of 6DoF joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[Connection6DoF]] = Connection6DoF
    """
    The type of the entity to convert.
    """

    def _convert(self, entity: Connection6DoF, **kwargs) -> dict[str, Any]:
        """
        Converts a Connection6DoF object to a dictionary of joint properties for Multiverse simulator.

        :param entity: The Connection6DoF object to convert.
        :return: A dictionary of joint properties.
        """
        joint_props = ConnectionConverter._convert(self, entity)
        assert len(entity.dofs) == 7, "Connection6DoF must have exactly six DOFs."
        return joint_props


@dataclass
class ActuatorConverter(EntityConverter, ABC):
    """
    Converts an Actuator object to a dictionary of actuator properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[Actuator]] = Actuator
    """
    The type of the entity to convert.
    """

    def _convert(self, entity: Actuator, **kwargs) -> dict[str, Any]:
        """
        Converts an Actuator object to a dictionary of joint properties for Multiverse simulator.

        :param entity: The Actuator object to convert.
        :return: A dictionary of actuator properties, by default containing list of DOF names.
        """
        actuator_props = EntityConverter._convert(self, entity)
        actuator_props["dof_names"] = [dof.name.name for dof in entity.dofs]
        return actuator_props


@dataclass
class CameraConverter(EntityConverter, ABC):
    """
    Converts a Camera object to a dictionary of camera properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[MultiSimCamera]] = MultiSimCamera
    """
    The type of the entity to convert.
    """

    def _convert(self, entity: MultiSimCamera, **kwargs) -> dict[str, Any]:
        """
        Converts a Camera object to a dictionary of camera properties for Multiverse simulator.

        :param entity: The Camera object to convert.
        :return: A dictionary of camer properties, by default containing list of DOF names.
        """
        camera_props = EntityConverter._convert(self, entity)
        camera_props["body"] = entity.body.name.name
        return camera_props


@dataclass
class LightConverter(EntityConverter, ABC):
    """
    Converts a Light object to a dictionary of light properties for Multiverse simulator.
    """

    entity_type: ClassVar[type[MultiSimLight]] = MultiSimLight
    """
    The type of the entity to convert.
    """

    def _convert(self, entity: MultiSimLight, **kwargs) -> dict[str, Any]:
        """
        Converts a Light object to a dictionary of light properties for Multiverse simulator.

        :param entity: The Light object to convert.
        :return: A dictionary of light properties, by default containing the parent body's name.
        """
        light_props = EntityConverter._convert(self, entity)
        light_props["body"] = entity.body.name.name
        return light_props


@dataclass
class MujocoActuator(UniqueSimulatorProperty):
    """
    Represents a MuJoCo-specific actuator in the world model.
    For more information, see: https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general
    """

    activation_limited: mujoco.mjtLimited = field(
        default=mujoco.mjtLimited.mjLIMITED_AUTO,
        metadata=SimulatorAttributeName("actlimited").as_dict(),
    )
    """
    If mujoco.mjtLimited.mjLIMITED_TRUE, the internal state (activation) associated with this actuator is automatically clamped to actrange at runtime. 
    If mujoco.mjtLimited.mjLIMITED_FALSE, activation clamping is disabled. 
    If mujoco.mjtLimited.mjLIMITED_AUTO and autolimits is set in compiler, activation clamping will automatically be set to mujoco.mjtLimited.mjLIMITED_TRUE if activation_range is defined without explicitly setting this attribute to mujoco.mjtLimited.mjLIMITED_TRUE. 
    """

    activation_range: list[float] = field(
        default_factory=lambda: [0.0, 0.0],
        metadata=SimulatorAttributeName("actrange").as_dict(),
    )
    """
    Range for clamping the activation state. The first value must be no greater than the second value.
    """

    control_limited: mujoco.mjtLimited = field(
        default=mujoco.mjtLimited.mjLIMITED_AUTO,
        metadata=SimulatorAttributeName("ctrllimited").as_dict(),
    )
    """
    If mujoco.mjtLimited.mjLIMITED_TRUE, the control input to this actuator is automatically clamped to ctrl_range at runtime. 
    If mujoco.mjtLimited.mjLIMITED_FALSE, control input clamping is disabled. 
    If mujoco.mjtLimited.mjLIMITED_AUTO and autolimits is set in compiler, control clamping will automatically be set to mujoco.mjtLimited.mjLIMITED_TRUE if ctrl_range is defined without explicitly setting this attribute to mujoco.mjtLimited.mjLIMITED_TRUE.
    """

    control_range: list[float] = field(
        default_factory=lambda: [0.0, 0.0],
        metadata=SimulatorAttributeName("ctrlrange").as_dict(),
    )
    """
    The range of the control input.
    """

    force_limited: mujoco.mjtLimited = field(
        default=mujoco.mjtLimited.mjLIMITED_AUTO,
        metadata=SimulatorAttributeName("forcelimited").as_dict(),
    )
    """
    If mujoco.mjtLimited.mjLIMITED_TRUE, the force output of this actuator is automatically clamped to force_range at runtime. 
    If mujoco.mjtLimited.mjLIMITED_FALSE, force clamping is disabled. 
    If mujoco.mjtLimited.mjLIMITED_AUTO and autolimits is set in compiler, force clamping will automatically be set to mujoco.mjtLimited.mjLIMITED_TRUE if force_range is defined without explicitly setting this attribute to mujoco.mjtLimited.mjLIMITED_TRUE.
    """

    force_range: list[float] = field(
        default_factory=lambda: [0.0, 0.0],
        metadata=SimulatorAttributeName("forcerange").as_dict(),
    )
    """
    Range for clamping the force output. The first value must be no greater than the second value.
    """

    bias_parameters: list[float] = field(
        default_factory=lambda: [0.0] * 10,
        metadata=SimulatorAttributeName("biasprm").as_dict(),
    )
    """
    Bias parameters. The affine bias type uses three parameters.
    """

    bias_type: mujoco.mjtBias = field(
        default=mujoco.mjtBias.mjBIAS_NONE,
        metadata=SimulatorAttributeName("biastype").as_dict(),
    )
    """
    The keywords have the following meaning:
    mujoco.mjtBias.mjBIAS_NONE:     bias_term = 0
    mujoco.mjtBias.mjBIAS_AFFINE:   bias_term = biasprm[0] + biasprm[1]*length + biasprm[2]*velocity
    mujoco.mjtBias.mjBIAS_MUSCLE:   bias_term = mju_muscleBias(…)
    mujoco.mjtBias.mjBIAS_USER:     bias_term = mjcb_act_bias(…)
    """

    dynamics_parameters: list[float] = field(
        default_factory=lambda: [1.0] + [0.0] * 9,
        metadata=SimulatorAttributeName("dynprm").as_dict(),
    )
    """
    Activation dynamics parameters.
    """

    dynamics_type: mujoco.mjtDyn = field(
        default=mujoco.mjtDyn.mjDYN_NONE,
        metadata=SimulatorAttributeName("dyntype").as_dict(),
    )
    """
    Activation dynamics type for the actuator.
    The keywords have the following meaning:
    mujoco.mjtDyn.mjDYN_NONE:           No internal state
    mujoco.mjtDyn.mjDYN_INTEGRATOR:     act_dot = ctrl
    mujoco.mjtDyn.mjDYN_FILTER:         act_dot = (ctrl - act) / dynprm[0]
    mujoco.mjtDyn.mjDYN_FILTEREXACT:    Like filter but with exact integration
    mujoco.mjtDyn.mjDYN_MUSCLE:         act_dot = mju_muscleDynamics(…)
    mujoco.mjtDyn.mjDYN_USER:           act_dot = mjcb_act_dyn(…)
    """

    gain_parameters: list[float] = field(
        default_factory=lambda: [0.0] * 10,
        metadata=SimulatorAttributeName("gainprm").as_dict(),
    )
    """
    Gain parameters.
    """

    gain_type: mujoco.mjtGain = field(
        default=mujoco.mjtGain.mjGAIN_FIXED,
        metadata=SimulatorAttributeName("gaintype").as_dict(),
    )
    """
    The gain and bias together determine the output of the force generation mechanism, which is currently assumed to be affine.
    The keywords have the following meaning:
    mujoco.mjtGain.mjGAIN_FIXED:    gain_term = gainprm[0]
    mujoco.mjtGain.mjGAIN_AFFINE:   gain_term = gain_prm[0] + gain_prm[1]*length + gain_prm[2]*velocity
    mujoco.mjtGain.mjGAIN_MUSCLE:   gain_term = mju_muscleGain(…)
    mujoco.mjtGain.mjGAIN_USER:     gain_term = mjcb_act_gain(…)
    """

    @classmethod
    def create_servo(cls, servo: PositionServo) -> MujocoActuator:
        """
        The MuJoCo actuator realising a position servo: a PD law with the servo's
        gains, clamped to its torque limit and to the driven degree of freedom's own
        position limits, if it has any.

        :param servo: The servo to realise; it drives exactly one degree of freedom.
        :return: The actuator.
        """
        gains = servo.gains
        [degree_of_freedom] = servo.dofs
        limits = degree_of_freedom.limits
        actuator = cls(
            dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
            gain_type=mujoco.mjtGain.mjGAIN_FIXED,
            gain_parameters=[gains.stiffness] + [0.0] * 9,
            bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
            bias_parameters=[0.0, -gains.stiffness, -gains.damping] + [0.0] * 7,
            force_range=[-gains.torque_limit, gains.torque_limit],
        )
        if limits.lower.position is None or limits.upper.position is None:
            actuator.control_limited = mujoco.mjtLimited.mjLIMITED_FALSE
            return actuator
        actuator.control_range = [limits.lower.position, limits.upper.position]
        return actuator

    @classmethod
    def create_velocity_servo(cls, servo: VelocityServo) -> MujocoActuator:
        """
        The MuJoCo actuator realising a velocity servo: a force proportional to the
        difference between the commanded and the actual velocity, clamped to the
        servo's torque limit. The commanded velocity itself is not clamped, since the
        joint it reaches may move along other axes than the degree of freedom the
        servo drives.

        :param servo: The servo to realise; it drives exactly one degree of freedom.
        :return: The actuator.
        """
        gains = servo.gains
        return cls(
            dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
            gain_type=mujoco.mjtGain.mjGAIN_FIXED,
            gain_parameters=[gains.damping] + [0.0] * 9,
            bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
            bias_parameters=[0.0, 0.0, -gains.damping] + [0.0] * 7,
            force_range=[-gains.torque_limit, gains.torque_limit],
            control_limited=mujoco.mjtLimited.mjLIMITED_FALSE,
        )


@dataclass
class MujocoCamera(MultiSimCamera):
    """
    Additional property representing a MuJoCo camera in the world model.
    """

    name: str = ""
    """
    Name of the camera.
    """

    mode: mujoco.mjtCamLight = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    """
    This attribute specifies how the camera position and orientation in world coordinates are computed in forward kinematics 
    (which in turn determine what the camera sees).
    """

    orthographic: bool = False
    """
    Whether the camera uses a perspective projection (the default) or an orthographic projection.
    Setting this attribute changes the semantic of the fovy attribute.
    """

    fovy: float = 45.0
    """
    Vertical field-of-view of the camera.
    """

    resolution: list[float] = field(default_factory=lambda: [1, 1])
    """
    Resolution of the camera in pixels [width height].
    """

    focal_length: list[float] = field(default_factory=lambda: [0, 0])
    """
    Focal length of the camera in length units. It is mutually exclusive with fovy.
    """

    focal_pixel: list[float] = field(default_factory=lambda: [0, 0])
    """
    Focal length of the camera in pixel units. If both focal and focalpixel are specified, the former is ignored.
    """

    principal_length: list[float] = field(default_factory=lambda: [0, 0])
    """
    Offset of the principal point of the camera with respect to the camera center in length units. It is mutually exclusive with fovy.
    """

    principal_pixel: list[float] = field(default_factory=lambda: [0, 0])
    """
    Offset of the principal point of the camera with respect to the camera center in pixel units. 
    If both principal and principalpixel are specified, the former is ignored.
    """

    sensor_size: list[float] = field(default_factory=lambda: [0, 0])
    """
    Size of the camera sensor in length units. It is mutually exclusive with fovy.
    """

    inter_pupilary_distance: float = 0.068
    """
    Inter-pupilary distance. This attribute only has an effect during stereoscopic rendering.
    """

    position: list[float] = field(default_factory=lambda: [0, 0, 0])
    """
    Position of the camera frame.
    """

    quaternion: list[float] = field(default_factory=lambda: [1, 0, 0, 0])
    """
    Orientation of the camera frame.
    """

    @staticmethod
    def _look_at_rotation(
        camera_position: Point3, target_position: Point3
    ) -> RotationMatrix:
        """
        Computes the world-frame orientation of a camera at ``camera_position`` looking at
        ``target_position``, following MuJoCo's convention (camera looks down its local -Z
        axis, +Y is up).

        :param camera_position: The world-frame position of the camera.
        :param target_position: The world-frame position the camera should look at.
        :return: The camera's world-frame orientation.
        """
        forward = target_position - camera_position
        forward.scale(1)
        up_hint = Vector3.Z()
        if abs(float(forward.dot(up_hint).to_np().item())) > 0.99:
            up_hint = Vector3.Y()

        z_axis = -forward
        x_axis = up_hint.cross(z_axis)
        x_axis.scale(1)
        return RotationMatrix.from_vectors(x=x_axis, z=z_axis)

    @classmethod
    def overview_pose(
        cls,
        bounds: numpy.ndarray,
        minimum_distance: float = 1.0,
        distance_factor: float = 1.5,
    ) -> HomogeneousTransformationMatrix:
        """
        Computes a fixed diagonal viewpoint that frames an axis-aligned bounding box.

        :param bounds: A ``(2, 3)`` array of the scene's ``[minimum, maximum]`` corners.
        :param minimum_distance: Floor (in meters) for the camera's distance to the box center,
            so a box that collapses to a point still gets a sensibly framed camera.
        :param distance_factor: Multiplier applied to the box's bounding diagonal to place the camera.
        :return: The framing camera's world-frame pose.
        """
        minimum = Point3.from_iterable(bounds[0])
        maximum = Point3.from_iterable(bounds[1])
        diagonal_vector = maximum - minimum
        center = minimum + diagonal_vector * 0.5
        diagonal = float(diagonal_vector.norm().to_np().item())
        distance = max(diagonal, minimum_distance) * distance_factor
        direction = Vector3.from_iterable([1.0, -1.0, 1.0])
        direction.scale(1)
        position = center + direction * distance
        rotation = cls._look_at_rotation(position, center)
        return HomogeneousTransformationMatrix.from_point_rotation_matrix(
            position, rotation
        )


@dataclass
class MujocoLight(MultiSimLight):
    """
    Additional property representing a MuJoCo light in the world model.
    For more information, see: https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-light
    """

    name: str = ""
    """
    Name of the light.
    """

    mode: mujoco.mjtCamLight = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    """
    This attribute specifies how the light position and orientation in world coordinates are computed in forward kinematics.
    """

    directional: bool = False
    """
    Whether the light is directional (parallel rays, e.g. sunlight) or a positional point/spot light.
    """

    active: bool = True
    """
    Whether the light is active.
    """

    cast_shadow: bool = True
    """
    Whether this light casts shadows.
    """

    position: list[float] = field(default_factory=lambda: [0, 0, 0])
    """
    Position of the light frame.
    """

    direction: list[float] = field(default_factory=lambda: [0, 0, -1])
    """
    Direction the light points in, relevant only for directional and spot lights.
    """

    ambient: list[float] = field(default_factory=lambda: [0, 0, 0])
    """
    Ambient color of the light, as ``[r, g, b]``.
    """

    diffuse: list[float] = field(default_factory=lambda: [0.7, 0.7, 0.7])
    """
    Diffuse color of the light, as ``[r, g, b]``.
    """

    specular: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.3])
    """
    Specular color of the light, as ``[r, g, b]``.
    """

    attenuation: list[float] = field(default_factory=lambda: [1, 0, 0])
    """
    Constant, linear, and quadratic attenuation coefficients for a positional light.
    """

    cutoff: float = 45.0
    """
    Cutoff angle, in degrees, for a spot light.
    """

    exponent: float = 10.0
    """
    Spotlight attenuation exponent for a spot light.
    """

    bulb_radius: float = 0.02
    """
    Radius of the light's bulb, used for soft shadows.
    """


@dataclass
class MujocoEquality(SimulatorAdditionalProperty):
    """
    Additional properties representing a MuJoCo equality constraint in the world model.
    """

    type: mujoco.mjtEq = field(kw_only=True)
    """
    The type of the equality constraint.
    """

    object_type: mujoco.mjtObj = field(kw_only=True)
    """
    The type of the objects being constrained.
    """

    name_1: str = field(kw_only=True)
    """
    The name of the first entity being constrained.
    """

    name_2: str = field(kw_only=True)
    """
    The name of the second entity being constrained.
    """

    data: list[float] = field(kw_only=True)
    """
    The data associated with the equality constraint.
    """


@dataclass
class MujocoTendon(SimulatorAdditionalProperty):
    """
    Additional properties representing a MuJoCo tendon in the world model.
    """

    name: str = ""
    """
    Name of the tendon.
    """

    actuator_force_limited: mujoco.mjtLimited = mujoco.mjtLimited.mjLIMITED_AUTO
    """
    This attribute specifies whether actuator forces acting on the tendon should be clamped.
    """

    actuator_force_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    """
    Range for clamping total actuator forces acting on this tendon.
    """

    armature: float = 0.0
    """
    Inertia associated with changes in tendon length.
    """

    damping: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """
    Damping coefficient. A positive value generates a damping force (linear in velocity) acting along the tendon.
    """

    frictionloss: float = 0.01
    """
    Friction loss caused by dry friction. To enable friction loss, set this attribute to a positive value.
    """

    group: int = 0
    """
    Defaults class for setting unspecified attributes.
    """

    limited: mujoco.mjtLimited = mujoco.mjtLimited.mjLIMITED_AUTO
    """
    If this attribute is “true”, the length limits defined by the range attribute below are imposed by the constraint solver. 
    If this attribute is “auto”, and autolimits is set in compiler, length limits will be enabled if range is defined.
    """

    margin: float = 0.0
    """
    The limit constraint becomes active when the absolute value of the difference between the tendon length and either limit of the specified range falls below this margin. 
    """

    material: str = ""
    """
    Material used to set the appearance of the tendon.
    """

    range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    """
    Range of allowed tendon lengths.
    """

    rgba: Color = field(
        default_factory=lambda: Color.from_rgba(red=0.5, green=0.5, blue=0.5, opacity=1)
    )
    """
    Color and transparency of the tendon.
    """

    solver_impedance_friction: list[float] = field(
        default_factory=lambda: [0.9, 0.95, 0.001, 0.5, 2]
    )
    """
    Constraint solver parameters for simulating dry friction in the tendon.
    """

    solver_impedance_limit: list[float] = field(
        default_factory=lambda: [0.9, 0.95, 0.001, 0.5, 2]
    )
    """
    Constraint solver parameters for simulating tendon limits.
    """

    solver_reference_friction: list[float] = field(default_factory=lambda: [0.02, 1.0])
    """
    Constraint solver parameters for simulating dry friction in the tendon.
    """

    solver_reference_limit: list[float] = field(default_factory=lambda: [0.02, 1.0])
    """
    Constraint solver parameters for simulating tendon limits.
    """

    spring_length: list[float] = field(default_factory=lambda: [-1.0, -1.0])
    """
    Spring resting position, can take either one or two values. If one value is given, it corresponds to the length of the tendon at rest.
    """

    stiffness: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """
    Stiffness coefficient. A positive value generates a spring force (linear in position) acting along the tendon.
    """

    width: float = 0.003
    """
    Radius of the cross-section area of the spatial tendon, used for rendering. 
    Parts of the tendon that wrap around geom obstacles are rendered with reduced width.
    """

    joints: dict[str, float] = field(default_factory=dict)
    """
    This element adds a joint to the computation of the fixed tendon length.
    The position or angle of each included joint is multiplied by the corresponding coef value, 
    and added up to obtain the tendon length.
    """


class ContactCategories(IntFlag):
    """
    A set of MuJoCo's 32 contact categories, one bit each.

    Only the first category has a name: MuJoCo puts every geom into it by default, so by
    default every geom collides with every other one. Every other bit is a category of
    the model's own choosing, so a mask may hold bits beyond the named ones.
    """

    NONE = 0
    """
    No category at all.
    """

    DEFAULT = 1
    """
    The category MuJoCo puts every geom into unless told otherwise.
    """


@dataclass(eq=False)
class MujocoGeom(UniqueSimulatorProperty):
    """
    An additional property declaring that a Shape is a MujocoGeom.
    """

    contact_type: ContactCategories = field(
        default=ContactCategories.DEFAULT,
        metadata=SimulatorAttributeName("contype").as_dict(),
    )
    """
    Which contact categories this geom offers when it presses into another geom;
    MuJoCo's ``contype``.

    A contact between geoms A and B is generated only if A offers a category B accepts
    or B offers one A accepts, i.e. ``A.contact_type & B.contact_affinity`` or
    ``B.contact_type & A.contact_affinity`` is nonzero. Clearing this mask stops the
    geom from initiating a contact while other geoms can still collide into it.
    """

    contact_affinity: ContactCategories = field(
        default=ContactCategories.DEFAULT,
        metadata=SimulatorAttributeName("conaffinity").as_dict(),
    )
    """
    Which contact categories this geom accepts pressing into it; MuJoCo's
    ``conaffinity``. Clearing it makes nothing collide with the geom.
    """


@dataclass(eq=False)
class MujocoJoint(UniqueSimulatorProperty):
    """
    An additional property declaring that a Connection is a MujocoJoint.
    """

    stiffness: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """
    The stiffness of the joint.
    """

    actuator_force_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    """
    Range for clamping total actuator forces acting on this joint. 
    It is available only for scalar joints (hinge and slider) and ignored for ball and free joints.
    The compiler expects the first value to be smaller than the second value.
    """


@dataclass(eq=False)
class MujocoBody(UniqueSimulatorProperty):
    """
    Additional properties representing a MuJoCo body in the world model.
    """

    motion_capture: bool = False
    """
    If True, the body is treated as a motion capture body.
    """


@dataclass
class MujocoConverter(EntityConverter, ABC): ...


@dataclass
class MujocoKinematicStructureEntityConverter(
    MujocoConverter, KinematicStructureEntityConverter, ABC
):
    pos_str: str = "pos"
    quat_str: str = "quat"


@dataclass
class MujocoBodyConverter(MujocoKinematicStructureEntityConverter, BodyConverter):
    mass_str: str = "mass"
    inertia_pos_str: str = "ipos"
    inertia_quat_str: str = "iquat"
    diagonal_inertia_str: str = "inertia"

    def _post_convert(
        self, entity: Body, body_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        return body_props


@dataclass
class MujocoRegionConverter(MujocoKinematicStructureEntityConverter, RegionConverter):
    def _post_convert(
        self, entity: Region, region_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        return region_props


@dataclass
class MujocoGeomConverter(MujocoConverter, ShapeConverter, ABC):
    pos_str: str = "pos"
    quat_str: str = "quat"
    rgba_str: str = "rgba"
    type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_SPHERE

    def _post_convert(
        self, entity: Shape, shape_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        shape_props.update(
            {
                "type": self.type,
            }
        )
        is_visible = kwargs.get("visible", True)
        is_collidable = kwargs.get("collidable", True)
        if is_visible and is_collidable:
            shape_props["group"] = (
                GeomVisibilityAndCollisionType.VISIBLE_AND_COLLIDABLE_1
            )
        elif is_visible and not is_collidable:
            shape_props["contype"] = 0
            shape_props["conaffinity"] = 0
            shape_props["group"] = GeomVisibilityAndCollisionType.ONLY_VISIBLE
        elif not is_visible and is_collidable:
            shape_props["group"] = GeomVisibilityAndCollisionType.ONLY_COLLIDABLE
        return shape_props


@dataclass
class MujocoBoxConverter(MujocoGeomConverter, BoxConverter):
    type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_BOX

    def _post_convert(
        self, entity: Box, shape_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        shape_props.update(
            MujocoGeomConverter._post_convert(self, entity, shape_props, **kwargs)
        )
        shape_props.update(
            {"size": [entity.scale.x / 2, entity.scale.y / 2, entity.scale.z / 2]}
        )
        return shape_props


@dataclass
class MujocoSphereConverter(MujocoGeomConverter, SphereConverter):
    type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_SPHERE

    def _post_convert(
        self, entity: Sphere, shape_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        shape_props.update(
            MujocoGeomConverter._post_convert(self, entity, shape_props, **kwargs)
        )
        shape_props.update({"size": [entity.radius, entity.radius, entity.radius]})
        return shape_props


@dataclass
class MujocoCylinderConverter(MujocoGeomConverter, CylinderConverter):
    type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_CYLINDER

    def _post_convert(
        self, entity: Cylinder, shape_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        shape_props.update(
            MujocoGeomConverter._post_convert(self, entity, shape_props, **kwargs)
        )
        shape_props.update({"size": [entity.radius, entity.height / 2, 0.0]})
        return shape_props


@dataclass
class MujocoMeshConverter(MujocoGeomConverter, MeshConverter):
    type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_MESH

    def _post_convert(
        self, entity: Mesh, shape_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        shape_props.update(
            MujocoGeomConverter._post_convert(self, entity, shape_props, **kwargs)
        )
        shape_props.update({"mesh": entity})
        if isinstance(entity.mesh.visual, TextureVisuals) and isinstance(
            entity.mesh.visual.material.name, str
        ):
            texture_file_path = self._resolve_texture_file_path(
                entity.mesh.visual.material, str(entity.local_file.parent)
            )
            if texture_file_path is not None:
                shape_props["texture_file_path"] = texture_file_path
        return shape_props

    @staticmethod
    def _resolve_texture_file_path(material: Any, mesh_directory: str) -> str | None:
        """
        Resolves the on-disk file backing a mesh's texture.

        trimesh reports a texture path relative to the mesh file that references it, so
        every candidate is resolved against that mesh's directory rather than the process
        working directory. An already absolute candidate passes through unchanged.

        :param material: The trimesh material (``TextureVisuals.material``) to resolve.
        :param mesh_directory: Directory of the mesh file the material came from.
        :return: The texture's file path, or ``None`` if the material has no image at
            all (a flat colour with no texture) or a programmatically generated image
            with no backing file.
        """
        image = material.image
        if image is None:
            return None
        candidates = [material.name, image.info.get("file_path", "")]
        if isinstance(image, PIL.ImageFile.ImageFile):
            candidates.append(image.filename)
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate:
                continue
            resolved = os.path.join(mesh_directory, candidate)
            if os.path.isfile(resolved):
                return resolved
        return None


@dataclass
class MujocoJointConverter(ConnectionConverter, ABC):
    pos_str: str = "pos"
    quat_str: str = "quat"
    type: mujoco.mjtJoint = mujoco.mjtJoint.mjJNT_HINGE
    armature_str: str = "armature"
    dry_friction_str: str = "frictionloss"
    damping_str: str = "damping"

    def _post_convert(
        self, entity: Connection, joint_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        joint_props["type"] = self.type
        return joint_props


@dataclass
class Mujoco1DOFJointConverter(MujocoJointConverter, Connection1DOFConverter):
    axis_str: str = "axis"
    range_str: str = "range"

    def _post_convert(
        self, entity: ActiveConnection1DOF, joint_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        joint_props = MujocoJointConverter._post_convert(self, entity, joint_props)
        if not numpy.allclose(joint_props["quat"], [1.0, 0.0, 0.0, 0.0]):
            joint_axis = numpy.array(joint_props["axis"])
            R_joint = Rotation.from_quat(quat=joint_props["quat"], scalar_first=True)  # type: ignore
            joint_props["axis"] = R_joint.apply(joint_axis).tolist()
        del joint_props["quat"]
        return joint_props


@dataclass
class MujocoRevoluteJointConverter(
    Mujoco1DOFJointConverter, ConnectionRevoluteConverter
):
    type: mujoco.mjtJoint = mujoco.mjtJoint.mjJNT_HINGE

    def _post_convert(
        self, entity: ActiveConnection1DOF, joint_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        joint_props = super()._post_convert(entity, joint_props, **kwargs)
        joint_range = joint_props.pop("range")
        if not any(limit is None for limit in joint_range):
            joint_props["range"] = joint_range
        return joint_props


@dataclass
class MujocoPrismaticJointConverter(
    Mujoco1DOFJointConverter, ConnectionPrismaticConverter
):
    type: mujoco.mjtJoint = mujoco.mjtJoint.mjJNT_SLIDE


@dataclass
class Mujoco6DOFJointConverter(MujocoJointConverter, Connection6DOFConverter):
    type: mujoco.mjtJoint = mujoco.mjtJoint.mjJNT_FREE


@dataclass
class SimulatorElementNames(UniqueEntityNames):
    """
    Names one kind of simulator element, such as every body, unique among that kind.
    """

    def preferred_name(self, entity: WorldEntityWithID) -> str:
        return entity.name.name


@dataclass
class SimulatorModelNames:
    """
    The name every entity of a world is built into a simulator model under.

    Each kind of element has its own names, as the simulator only requires names to be
    unique among one kind.
    """

    _bodies: SimulatorElementNames = field(
        init=False, default_factory=SimulatorElementNames
    )
    """
    The names of the bodies, which bodies and regions both become.
    """

    _joints: SimulatorElementNames = field(
        init=False, default_factory=SimulatorElementNames
    )
    """
    The names of the joints, which connections become.
    """

    _actuators: SimulatorElementNames = field(
        init=False, default_factory=SimulatorElementNames
    )
    """
    The names of the actuators.
    """

    def of(self, entity: KinematicStructureEntity | Connection | Actuator) -> str:
        """
        :param entity: An entity built into the model.
        :return: The name the entity is built into the model under, giving it one if
            it has none yet.
        """
        if isinstance(entity, Connection):
            return self._joints.assign(entity)
        if isinstance(entity, Actuator):
            return self._actuators.assign(entity)
        return self._bodies.assign(entity)


ConnectionType = TypeVar("ConnectionType", bound=Connection)
"""
The type of connection a realisation realises.
"""


@dataclass
class MujocoConnectionJoints(Generic[ConnectionType], SubClassSafeGeneric, ABC):
    """
    How a connection is realised by the joints of a MuJoCo model: which joints it is
    built as, which values they start at, and which of them moves each of its degrees of
    freedom.

    A realisation binds the type of connection it realises; a subclass of that type is
    realised the same way unless a realisation for a more specific type exists. The
    builder looks up the realisation of a connection by the connection's type, so a new
    kind of connection is simulated by adding a subclass for it instead of changing the
    builder.
    """

    connection: ConnectionType
    """
    The connection realised.
    """

    world: World
    """
    The world the connection belongs to.
    """

    element_names: SimulatorModelNames
    """
    The name every entity of the world is built into the model under.
    """

    @classmethod
    def of(
        cls,
        connection: Connection,
        world: World,
        element_names: SimulatorModelNames,
    ) -> MujocoConnectionJoints:
        """
        :param connection: A connection in the world.
        :param world: The world the connection belongs to.
        :param element_names: The name every entity of the world is built into the
            model under.
        :return: The realisation for the most specific type the connection is of.
        """
        return cls._realisation_for_type(type(connection))(
            connection, world, element_names
        )

    @classmethod
    @lru_cache(maxsize=None)
    def _realisation_for_type(
        cls, connection_type: type[Connection]
    ) -> type[MujocoConnectionJoints]:
        """
        :param connection_type: The type of a connection.
        :return: The realisation for the most specific type a connection of that type
            is of.
        """
        realisation_by_type = {}
        for realisation in recursive_subclasses(cls) + [cls]:
            try:
                realised_type = realisation.get_type_of_generic_parameter(
                    ConnectionType
                )
            except UnboundGenericParameter:
                continue
            realisation_by_type[realised_type] = realisation
        for base_type in connection_type.__mro__:
            if base_type in realisation_by_type:
                return realisation_by_type[base_type]
        raise NotImplementedError(
            f"No MuJoCo realisation found for connection type {connection_type}."
        )

    @abstractmethod
    def build(self, builder: MujocoBuilder) -> None:
        """
        Add the joints realising the connection to the model being built.

        :param builder: The builder whose model they are added to.
        """

    @abstractmethod
    def home_joint_positions(self) -> list[float]:
        """
        :return: The values the joints start at, in the order they are built in.
        """

    @abstractmethod
    def joint_name_of(self, degree_of_freedom: DegreeOfFreedom) -> str:
        """
        :param degree_of_freedom: One of the connection's degrees of freedom.
        :return: The name of the joint moving it.
        """

    def control_of(self, actuator: Actuator) -> float:
        """
        :param actuator: An actuator driving one of the connection's degrees of freedom.
        :return: The control input of the MuJoCo actuator realising it: by default the
            actuator's set point.
        """
        return actuator.set_point

    def own_actuator_controls(self) -> dict[str, float]:
        """
        :return: The control input of every MuJoCo actuator the realisation builds
            itself, beyond those realising the world's actuators, by its name; none by
            default.
        """
        return {}

    def measured_positions(
        self, simulator: MujocoSimulator
    ) -> dict[DegreeOfFreedom, float]:
        """
        :param simulator: The simulator running the model.
        :return: The position of each degree of freedom of the connection that the
            simulation measures beyond its joints' own, such as a drive's odometry; none
            by default.
        """
        return {}


@dataclass
class MujocoSingleJoint(MujocoConnectionJoints[ConnectionType]):
    """
    A connection built as the one joint :class:`MujocoJointConverter` converts it to,
    named after the connection.
    """

    def build(self, builder: MujocoBuilder) -> None:
        joint_props = MujocoJointConverter.convert(self.connection)
        joint_name = self.element_names.of(self.connection)
        joint_props["name"] = joint_name
        if "equality_joint" in joint_props:
            equality_joint = joint_props.pop("equality_joint")
            equality = builder.spec.add_equality()
            equality.type = mujoco.mjtEq.mjEQ_JOINT
            equality.objtype = mujoco.mjtObj.mjOBJ_JOINT
            equality.name1 = joint_name
            equality.name2 = self._mimicked_joint_name()
            equality.data = equality_joint["data"]
        mujoco_joint = self.connection.get_simulator_property_of_type(MujocoJoint)
        if mujoco_joint is not None:
            joint_props["stiffness"] = (
                mujoco_joint.stiffness[0]
                if mujoco.mj_version() < 3007000
                else mujoco_joint.stiffness
            )
            joint_props["actfrcrange"] = mujoco_joint.actuator_force_range

        joint_spec = builder.body_spec_of(self.connection.child).add_joint(
            **joint_props
        )
        if joint_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=joint_name,
                entity_type=mujoco.mjtObj.mjOBJ_JOINT,
                action="add",
            )

    def _mimicked_joint_name(self) -> str:
        """
        :return: The name of the joint of the connection that the degree of freedom
            this connection mimics is named after, or the name of the degree of freedom
            if no connection is.
        """
        degree_of_freedom = self.connection.raw_dof
        mimicked_connection = next(
            (
                other
                for other in self.world.connections
                if degree_of_freedom in other.dofs
                and other.name.name == degree_of_freedom.name.name
            ),
            None,
        )
        if mimicked_connection is None:
            return degree_of_freedom.name.name
        return self.element_names.of(mimicked_connection)

    def home_joint_positions(self) -> list[float]:
        return [
            self.world.state[degree_of_freedom.id].position
            for degree_of_freedom in self.connection.active_dofs
            + self.connection.passive_dofs
        ]

    def joint_name_of(self, degree_of_freedom: DegreeOfFreedom) -> str:
        return self.element_names.of(self.connection)


@dataclass
class MujocoConnectionJoint(MujocoSingleJoint[Connection]):
    """
    A connection no more specific realisation exists for, built as a single joint.
    """


@dataclass
class MujocoFreeJoint(MujocoSingleJoint[Connection6DoF]):
    """
    A 6DoF connection, built as a free joint.
    """

    def home_joint_positions(self) -> list[float]:
        # A free joint's DOF state is relative to the connection frame and stays at the
        # identity by default; the body's actual placement lives in
        # parent_T_connection_expression instead. Reading the raw DOF state here would
        # bake a zero pose into the keyframe regardless of where the body is actually
        # placed, so use the fully evaluated origin instead.
        (
            x,
            y,
            z,
            quaternion_x,
            quaternion_y,
            quaternion_z,
            quaternion_w,
        ) = self.connection.origin_as_position_quaternion().evaluate()[0]
        return [x, y, z, quaternion_w, quaternion_x, quaternion_y, quaternion_z]


@dataclass
class MujocoNoJoint(MujocoConnectionJoints[ConnectionType]):
    """
    A connection built as no joint at all: in MuJoCo, every body not connected to its
    parent by a joint is fixed to it.
    """

    def build(self, builder: MujocoBuilder) -> None:
        pass

    def home_joint_positions(self) -> list[float]:
        return []

    def joint_name_of(self, degree_of_freedom: DegreeOfFreedom) -> str:
        raise MujocoEntityNotFoundError(
            entity_name=degree_of_freedom.name.name,
            entity_type=mujoco.mjtObj.mjOBJ_JOINT,
        )


@dataclass
class MujocoFixedConnectionJoints(MujocoNoJoint[FixedConnection]):
    """
    A fixed connection, built as no joint at all.
    """


@dataclass
class DriveAxis:
    """
    One of the joints a wheeled drive's base moves across the floor with.
    """

    measured: DegreeOfFreedom
    """
    The degree of freedom the joint's position is measured as.
    """

    joint_type: mujoco.mjtJoint
    """
    The kind of joint.
    """

    direction: NDArray[numpy.float64]
    """
    The joint's axis in the drive's ``odom`` frame.
    """

    commanded: DegreeOfFreedom | None = None
    """
    The degree of freedom whose servo drives the joint; ``None`` where no servo of the
    world drives it.
    """


DriveType = TypeVar("DriveType", bound=WheeledDrive)
"""
The type of wheeled drive a realisation realises.
"""


@dataclass
class MujocoWheeledDriveJoints(MujocoConnectionJoints[DriveType], ABC):
    """
    A wheeled drive. If servos drive the degrees of freedom it is commanded with, it is
    built as a slide along x, a slide along y and a hinge about z, all in the drive's
    ``odom`` frame: the servos push its base across the floor at the commanded
    velocity, and the positions the joints reach are the drive's odometry. Otherwise it
    is built as no joint at all and its base stays where it is built.
    """

    @classproperty
    @abstractmethod
    def kind(cls) -> str:
        """
        :return: What the drive's joints are named as, after the robot the drive moves
            and before the axis each joint moves along.
        """

    @property
    @abstractmethod
    def axes_when_servoed(self) -> list[DriveAxis]:
        """
        :return: The joints the base moves with, in the order they are built in, once
            servos drive it.
        """

    @property
    def servoed(self) -> bool:
        """
        :return: Whether servos drive every degree of freedom the drive is commanded
            with.
        """
        actuated = {
            degree_of_freedom.id
            for actuator in self.world.actuators
            for degree_of_freedom in actuator.dofs
        }
        return {
            degree_of_freedom.id for degree_of_freedom in self.connection.active_dofs
        } <= actuated

    @property
    def axes(self) -> list[DriveAxis]:
        """
        :return: The joints the base moves with, in the order they are built in; none
            if the drive is not servoed.
        """
        return self.axes_when_servoed if self.servoed else []

    def build(self, builder: MujocoBuilder) -> None:
        axes = self.axes
        if not axes:
            return
        child_body_spec = builder.body_spec_of(self.connection.child)
        child_T_connection = inverse_frame(
            self.connection.connection_T_child_expression.to_np()
        )
        for axis in axes:
            child_body_spec.add_joint(
                name=self.joint_name_of(axis.measured),
                type=axis.joint_type,
                axis=(child_T_connection[:3, :3] @ axis.direction).tolist(),
                pos=child_T_connection[:3, 3].tolist(),
            )

    def home_joint_positions(self) -> list[float]:
        return [self.world.state[axis.measured.id].position for axis in self.axes]

    def joint_name_of(self, degree_of_freedom: DegreeOfFreedom) -> str:
        """
        :param degree_of_freedom: A degree of freedom the drive is commanded with or
            measured as.
        :return: The name of the joint it belongs to, after the robot the drive moves
            and the axis the joint moves along.
        """
        for axis in self.axes:
            if degree_of_freedom in (axis.commanded, axis.measured):
                robot_name = self.connection.name.prefix or self.connection.name.name
                return f"{robot_name}_{self.kind}_{axis.measured.name.name}"
        raise MujocoEntityNotFoundError(
            entity_name=degree_of_freedom.name.name,
            entity_type=mujoco.mjtObj.mjOBJ_JOINT,
        )

    def measured_positions(
        self, simulator: MujocoSimulator
    ) -> dict[DegreeOfFreedom, float]:
        """
        :param simulator: The simulator running the model.
        :return: The drive's odometry: the x, y and yaw its joints have reached.
        """
        model = simulator._mj_model
        measured_positions = {}
        for axis in self.axes:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_name_of(axis.measured)
            )
            if joint_id == -1:
                continue
            measured_positions[axis.measured] = float(
                simulator._mj_data.qpos[model.jnt_qposadr[joint_id]]
            )
        return measured_positions

    def _in_odom(self, base_velocity: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
        """
        :param base_velocity: A velocity along the base's x and y.
        :return: The same velocity along the x and the y of the drive's ``odom``, turned
            by the drive's yaw, as the drive's ``update_state`` turns it.
        """
        yaw = self.world.state[self.connection.yaw.id].position
        odom_R_base = numpy.array(
            [[numpy.cos(yaw), -numpy.sin(yaw)], [numpy.sin(yaw), numpy.cos(yaw)]]
        )
        return odom_R_base @ base_velocity


@dataclass
class MujocoOmniDriveJoints(MujocoWheeledDriveJoints[OmniDrive]):
    """
    An omnidirectional drive, whose servos drive its velocities along the base's x and
    y and its yaw.
    """

    @classproperty
    def kind(cls) -> str:
        return "omni_drive"

    @property
    def axes_when_servoed(self) -> list[DriveAxis]:
        drive = self.connection
        return [
            DriveAxis(
                measured=drive.x,
                joint_type=mujoco.mjtJoint.mjJNT_SLIDE,
                direction=numpy.array([1.0, 0.0, 0.0]),
                commanded=drive.x_velocity,
            ),
            DriveAxis(
                measured=drive.y,
                joint_type=mujoco.mjtJoint.mjJNT_SLIDE,
                direction=numpy.array([0.0, 1.0, 0.0]),
                commanded=drive.y_velocity,
            ),
            DriveAxis(
                measured=drive.yaw,
                joint_type=mujoco.mjtJoint.mjJNT_HINGE,
                direction=numpy.array([0.0, 0.0, 1.0]),
                commanded=drive.yaw,
            ),
        ]

    def control_of(self, actuator: Actuator) -> float:
        """
        The drive is commanded with velocities in the frame of its base, while the
        slides move along the axes of its ``odom``, so the commanded translational
        velocity is turned by the drive's yaw.

        :param actuator: An actuator driving one of the drive's degrees of freedom.
        :return: The velocity the joint it reaches is commanded to.
        """
        drive = self.connection
        [degree_of_freedom] = actuator.dofs
        if degree_of_freedom not in (drive.x_velocity, drive.y_velocity):
            return super().control_of(actuator)
        state = self.world.state
        odom_velocity = self._in_odom(
            numpy.array(
                [
                    state[drive.x_velocity.id].velocity,
                    state[drive.y_velocity.id].velocity,
                ]
            )
        )
        return float(odom_velocity[0 if degree_of_freedom == drive.x_velocity else 1])


@dataclass
class MujocoDifferentialDriveJoints(MujocoWheeledDriveJoints[DifferentialDrive]):
    """
    A differential drive, whose servos drive its velocity along the base's x and its
    yaw.

    The base only moves along its own x, which in ``odom`` takes both slides, so the
    slide along y gets a velocity servo of its own, with the gains of the servo of the
    forward velocity.
    """

    @classproperty
    def kind(cls) -> str:
        return "differential_drive"

    @property
    def axes_when_servoed(self) -> list[DriveAxis]:
        drive = self.connection
        return [
            DriveAxis(
                measured=drive.x,
                joint_type=mujoco.mjtJoint.mjJNT_SLIDE,
                direction=numpy.array([1.0, 0.0, 0.0]),
                commanded=drive.x_velocity,
            ),
            DriveAxis(
                measured=drive.y,
                joint_type=mujoco.mjtJoint.mjJNT_SLIDE,
                direction=numpy.array([0.0, 1.0, 0.0]),
            ),
            DriveAxis(
                measured=drive.yaw,
                joint_type=mujoco.mjtJoint.mjJNT_HINGE,
                direction=numpy.array([0.0, 0.0, 1.0]),
                commanded=drive.yaw,
            ),
        ]

    @property
    def sideways_servo_name(self) -> str:
        """
        :return: The name of the servo of the slide along y.
        """
        return f"{self.joint_name_of(self.connection.y)}_servo"

    @property
    def _forward_velocity_in_odom(self) -> NDArray[numpy.float64]:
        """
        :return: The velocity the drive is commanded to along its base's x, along the
            x and the y of its ``odom``.
        """
        forward_velocity = self.world.state[self.connection.x_velocity.id].velocity
        return self._in_odom(numpy.array([forward_velocity, 0.0]))

    def build(self, builder: MujocoBuilder) -> None:
        super().build(builder)
        if not self.servoed:
            return
        [forward_servo] = [
            actuator
            for actuator in self.world.actuators
            if self.connection.x_velocity in actuator.dofs
        ]
        builder.build_actuator_on_joint(
            forward_servo,
            self.sideways_servo_name,
            self.joint_name_of(self.connection.y),
        )

    def control_of(self, actuator: Actuator) -> float:
        """
        The forward velocity is commanded along the base's x, while the slide moves
        along the x of its ``odom``, so it is turned by the drive's yaw.

        :param actuator: An actuator driving one of the drive's degrees of freedom.
        :return: The velocity the joint it reaches is commanded to.
        """
        drive = self.connection
        [degree_of_freedom] = actuator.dofs
        if degree_of_freedom != drive.x_velocity:
            return super().control_of(actuator)
        return float(self._forward_velocity_in_odom[0])

    def own_actuator_controls(self) -> dict[str, float]:
        """
        :return: The velocity the slide along y is commanded to: the forward velocity,
            turned by the drive's yaw.
        """
        if not self.servoed:
            return {}
        return {self.sideways_servo_name: float(self._forward_velocity_in_odom[1])}


@dataclass
class MujocoActuatorConverter(ActuatorConverter, ABC):
    entity_type: ClassVar[type[Actuator]] = Actuator


@dataclass
class MujocoGeneralActuatorConverter(MujocoActuatorConverter, ActuatorConverter):
    def _post_convert(
        self, entity: Actuator, actuator_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        mujoco_actuator = entity.get_simulator_property_of_type(MujocoActuator)
        if mujoco_actuator is not None:
            actuator_props.update(mujoco_actuator.to_dict())
        return actuator_props


@dataclass
class MujocoPositionServoConverter(MujocoActuatorConverter):
    """
    Converts a position servo to the properties of the MuJoCo actuator realising it
    (see :meth:`MujocoActuator.create_servo`).
    """

    entity_type: ClassVar[type[PositionServo]] = PositionServo

    def _post_convert(
        self, entity: PositionServo, actuator_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        actuator_props.update(MujocoActuator.create_servo(entity).to_dict())
        return actuator_props


@dataclass
class MujocoVelocityServoConverter(MujocoActuatorConverter):
    """
    Converts a velocity servo to the properties of the MuJoCo actuator realising it
    (see :meth:`MujocoActuator.create_velocity_servo`).
    """

    entity_type: ClassVar[type[VelocityServo]] = VelocityServo

    def _post_convert(
        self, entity: VelocityServo, actuator_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        actuator_props.update(MujocoActuator.create_velocity_servo(entity).to_dict())
        return actuator_props


@dataclass
class MujocoCameraConverter(CameraConverter, ABC):
    entity_type: ClassVar[type[MujocoCamera]] = MujocoCamera

    def _post_convert(
        self, entity: MujocoCamera, camera_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        camera_props["mode"] = entity.mode
        if mujoco.mj_version() >= 3005000:
            camera_props["proj"] = (
                mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
                if entity.orthographic
                else mujoco.mjtProjection.mjPROJ_PERSPECTIVE
            )
        else:
            camera_props["orthographic"] = entity.orthographic
        camera_props["fovy"] = entity.fovy
        camera_props["resolution"] = entity.resolution
        camera_props["focal_length"] = entity.focal_length
        camera_props["focal_pixel"] = entity.focal_pixel
        camera_props["principal_length"] = entity.principal_length
        camera_props["principal_pixel"] = entity.principal_pixel
        camera_props["sensor_size"] = entity.sensor_size
        camera_props["ipd"] = entity.inter_pupilary_distance
        camera_props["pos"] = entity.position
        camera_props["quat"] = entity.quaternion
        return camera_props


@dataclass
class MujocoLightConverter(LightConverter, ABC):
    entity_type: ClassVar[type[MujocoLight]] = MujocoLight

    def _post_convert(
        self, entity: MujocoLight, light_props: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        light_props["mode"] = entity.mode
        light_props["type"] = (
            mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            if entity.directional
            else mujoco.mjtLightType.mjLIGHT_SPOT
        )
        light_props["active"] = entity.active
        light_props["castshadow"] = entity.cast_shadow
        light_props["pos"] = entity.position
        light_props["dir"] = entity.direction
        light_props["ambient"] = entity.ambient
        light_props["diffuse"] = entity.diffuse
        light_props["specular"] = entity.specular
        light_props["attenuation"] = entity.attenuation
        light_props["cutoff"] = entity.cutoff
        light_props["exponent"] = entity.exponent
        light_props["bulbradius"] = entity.bulb_radius
        return light_props


@dataclass
class MultiSimBuilder(ABC):
    """
    A builder to build a world in the Multiverse simulator.
    """

    _world: World | None = None
    """
    The world to be built.
    """

    element_names: SimulatorModelNames = field(
        init=False, default_factory=SimulatorModelNames
    )
    """
    The name every entity of the last world built is built under.
    """

    def build_world(self, world: World, file_path: str):
        """
        Builds the world in the simulator and saves it to a file.

        :param world: The world to be built.
        :param file_path: The file path to save the world to.
        """
        self._world = world
        self._asset_folder_path = os.path.join(os.path.dirname(file_path), "assets")

        root = Body(name=PrefixedName("world"))

        if not os.path.exists(self.asset_folder_path):
            os.makedirs(self.asset_folder_path)
        if len(self.world.bodies) == 0:
            with self.world.modify_world():
                self.world.add_body(root)
        elif self.world.root.name != root.name:
            # search for all Connection6DoF joints that are connected to the non "world" root
            # to change their parent to the new "world" root later.
            # Mujoco identifies all Connection6DoF joints as free joints.
            # Free joints in Mujoco need to be attached to the top level link
            # and the top level link needs the name "world"
            free_joint_bodies = [
                body
                for body in self.world.bodies
                if isinstance(body.parent_connection, Connection6DoF)
                and body.parent_connection.parent == self.world.root
            ]

            with world.modify_world():
                root_bodies = [
                    body for body in self.world.bodies if body.parent_connection is None
                ]
                self.world.add_body(root)
                for root_body in root_bodies:
                    connection = FixedConnection(parent=root, child=root_body)
                    self.world.add_connection(connection)

            # attach free joint bodies to the new top level body in mujoco
            with world.modify_world():
                for free_joint_body in free_joint_bodies:
                    self.world.move_branch(free_joint_body, root)

        self.element_names = SimulatorModelNames()
        self._start_build(file_path=file_path)

        for body in world.bodies_topologically_sorted:
            self.build_body(body=body)

        for region in world.regions:
            self.build_region(region=region)

        for connection in world.connections_topologically_sorted:
            self._build_connection(connection=connection)

        for actuator in world.actuators:
            self._build_actuator(actuator=actuator)

        self._end_build(file_path=file_path)

    def build_body(self, body: Body):
        """
        Builds a body in the simulator including its shapes.

        :param body: The body to build.
        """
        self._build_body(body=body)
        for shape in {
            id(s): s for s in body.visual.shapes + body.collision.shapes
        }.values():
            self._build_shape(
                parent=body,
                shape=shape,
                is_visible=shape in body.visual,
                is_collidable=shape in body.collision,
            )
        for additional_property in body.simulator_additional_properties:
            if isinstance(additional_property, MultiSimCamera):
                self._build_camera(camera=additional_property)
            elif isinstance(additional_property, MultiSimLight):
                self._build_light(light=additional_property)

    def build_region(self, region: Region):
        """
        Builds a region in the simulator including its shapes.

        :param region: The region to build.
        """
        self._build_region(region=region)
        for shape in region.area:
            self._build_shape(
                parent=region, shape=shape, is_visible=True, is_collidable=False
            )

    @abstractmethod
    def _start_build(self, file_path: str):
        """
        Starts the building process for the simulator.

        :param file_path: The file path to save the world to.
        """
        raise NotImplementedError

    @abstractmethod
    def _end_build(self, file_path: str):
        """
        Ends the building process for the simulator and saves the world to a file.

        :param file_path: The file path to save the world to.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_body(self, body: Body):
        """
        Builds a body in the simulator.

        :param body: The body to build.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_region(self, region: Region):
        """
        Builds a region in the simulator.

        :param region: The region to build.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_shape(
        self,
        parent: Body | Region,
        shape: Shape,
        is_visible: bool,
        is_collidable: bool,
    ):
        """
        Builds a shape in the simulator and attaches it to its parent body or region.

        :param parent: The parent body or region to attach the shape to.
        :param shape: The shape to build.
        :param is_visible: Whether the shape is visible.
        :param is_collidable: Whether the shape is collidable.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_connection(self, connection: Connection):
        """
        Builds a connection in the simulator.

        :param connection: The connection to build.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_actuator(self, actuator: Actuator):
        """
        Builds an actuator in the simulator.

        :param actuator: The actuator to build.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_camera(self, camera: MultiSimCamera):
        """
        Builds a camera in the simulator.

        :param camera: The camera to build.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_light(self, light: MultiSimLight):
        """
        Builds a light in the simulator.

        :param light: The light to build.
        """
        raise NotImplementedError

    @property
    def asset_folder_path(self) -> str:
        """
        The default file path to save the world to.
        """
        return self._asset_folder_path

    @property
    def world(self) -> World:
        return self._world


@dataclass
class WavefrontCopy:
    """
    An .obj copy of a mesh file MuJoCo cannot read.
    """

    mesh_file_path: str
    """
    The copy.
    """

    texture_file_path: str | None
    """
    The image the copy is textured with, written beside it, or ``None`` for an
    untextured mesh.
    """


@dataclass
class MujocoBuilder(MultiSimBuilder):
    """
    A builder to build a world in the Mujoco simulator.
    """

    spec: mujoco.MjSpec = field(default=mujoco.MjSpec())

    minimum_mass: float = 1e-3
    """
    The mass, in kilograms, every body weighs at least, so that a frame without an
    inertial or a shape can still move.
    """

    minimum_inertia: float = 1e-6
    """
    The moment of inertia, in kilogram square metres, every body has at least about each
    of its principal axes.
    """

    planar_thickness_epsilon: float = 1e-4
    """
    Below this thickness (in meters, measured along a mesh's own best-fit-plane
    normal), a mesh is treated as near-planar for :meth:`_thicken_if_near_planar`.
    """

    _thickened_mesh_paths: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Memoizes :meth:`_thicken_if_near_planar` by input path, so a mesh referenced by
    many geoms (e.g. a screw reused across a whole object) is only loaded and checked
    once instead of once per reference.
    """

    _wavefront_copies: dict[str, WavefrontCopy] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Memoizes :meth:`_wavefront_copy_of` by input path, so a mesh referenced by many
    geoms is copied, and its texture registered, once.
    """

    def _start_build(self, file_path: str):
        self.spec = mujoco.MjSpec()
        self.spec.modelname = "scene"
        self.spec.compiler.degree = 0
        # Some descriptions, such as the PR2's, give links an inertia no rigid body can
        # have, which MuJoCo refuses to compile unless it may correct them.
        self.spec.compiler.balanceinertia = True
        # A frame with neither an inertial nor a shape still needs some mass to move.
        self.spec.compiler.boundmass = self.minimum_mass
        self.spec.compiler.boundinertia = self.minimum_inertia
        self._thickened_mesh_paths = {}

    def _end_build(self, file_path: str):
        self._build_contact_exclusions()
        self._build_equalities()
        self._build_tendons()
        self.spec.compile()
        self.spec.to_file(file_path)
        import xml.etree.ElementTree as ET

        tree = ET.parse(file_path)
        root = tree.getroot()
        for compiler_element in root.iter("compiler"):
            # A body's own inertial wins; its geoms only stand in for one it lacks.
            compiler_element.set("inertiafromgeom", "auto")
            compiler_element.set("balanceinertia", "true")
            compiler_element.set("boundmass", str(self.minimum_mass))
            compiler_element.set("boundinertia", str(self.minimum_inertia))
        for body_id, body_element in enumerate(root.findall(".//body")):
            body_spec = self.spec.bodies[body_id + 1]
            if numpy.isclose(body_spec.mass, 0.0):
                continue
            inertial_element = ET.SubElement(body_element, "inertial")
            inertial_element.set("mass", f"{body_spec.mass}")
            inertial_element.set(
                "diaginertia", " ".join(map(str, body_spec.inertia.tolist()))
            )
            inertial_element.set("pos", " ".join(map(str, body_spec.ipos.tolist())))
            inertial_element.set("quat", " ".join(map(str, body_spec.iquat.tolist())))
        for material_id, material_element in enumerate(root.findall(".//material")):
            material_spec = self.spec.materials[material_id]
            texture_name = material_spec.textures[0]
            if texture_name != "":
                material_element.set("texture", texture_name)
        keyframe_element = ET.SubElement(root, "keyframe")
        key_element = ET.SubElement(keyframe_element, "key")
        key_element.set("name", "home")
        key_element.set("time", "0")
        qpos = []
        for body in self.world.bodies_topologically_sorted:
            if body.parent_connection is None:
                continue
            qpos += MujocoConnectionJoints.of(
                body.parent_connection, self.world, self.element_names
            ).home_joint_positions()
        key_element.set("qpos", " ".join(map(str, qpos)))
        tree.write(file_path, encoding="utf-8", xml_declaration=True)

    def _build_body(self, body: Body):
        gravity_compensation = body.get_simulator_property_of_type(GravityCompensation)
        additional_properties = {}
        if gravity_compensation is not None:
            additional_properties["gravcomp"] = gravity_compensation.fraction
        self._build_mujoco_body(
            entity=body, additional_properties=additional_properties
        )

    def _build_region(self, region: Region):
        self._build_mujoco_body(entity=region)

    def _build_shape(
        self,
        parent: Body | Region,
        shape: Shape,
        is_visible: bool,
        is_collidable: bool,
    ):
        geom_props = MujocoGeomConverter.convert(
            shape, visible=is_visible, collidable=is_collidable
        )
        parent_body_spec = self.body_spec_of(parent)
        if geom_props["type"] == mujoco.mjtGeom.mjGEOM_MESH:
            if not self._parse_geom(geom_props=geom_props):
                logger.warning(
                    f"Mesh {shape.mesh} could not be parsed. Skipping geom {geom_props['name']}."
                )
                return
        else:
            texture_file_path = geom_props.pop("texture_file_path", None)
            texture_repeat = geom_props.pop("texture_repeat", (1.0, 1.0))
            texture_uniform = geom_props.pop("texture_uniform", False)
            if isinstance(texture_file_path, str):
                geom_props["material"] = self._register_texture_material(
                    texture_file_path=texture_file_path,
                    texture_repeat=texture_repeat,
                    texture_uniform=texture_uniform,
                )
        contact = shape.get_simulator_property_of_type(ContactParameters)
        if contact is not None:
            geom_props["friction"] = contact.friction.to_list()
            if contact.stiffness is not None:
                geom_props["solref"] = contact.stiffness.to_list()
            if contact.impedance is not None:
                geom_props["solimp"] = contact.impedance.to_list()
        mujoco_geom = shape.get_simulator_property_of_type(MujocoGeom)
        if mujoco_geom is not None:
            mujoco_geom_props = mujoco_geom.to_dict()
            if not is_collidable:
                # a visual-only geom keeps the contact masks the converter cleared
                mujoco_geom_props.pop("contype")
                mujoco_geom_props.pop("conaffinity")
            geom_props.update(mujoco_geom_props)
        geom_spec = parent_body_spec.add_geom(**geom_props)
        if geom_spec.type == mujoco.mjtGeom.mjGEOM_BOX and geom_spec.size[2] == 0:
            geom_spec.type = mujoco.mjtGeom.mjGEOM_PLANE
            geom_spec.size = [0, 0, 0.05]
        if geom_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=geom_props["name"],
                entity_type=mujoco.mjtObj.mjOBJ_GEOM,
                action="add",
            )

    def _create_stl_from_dae_mesh(
        self, original_mesh_file_path: str, stl_file_path: str
    ):
        """
        Creates an .stl mesh at the location specified by stl_file_path from the original .dae mesh.

        :param original_mesh_file_path: filepath to the original .dae mesh
        :param stl_file_path: filepath to save the new .stl mesh to
        """
        logger.info(
            f"Converting Collada mesh to STL for MuJoCo: {original_mesh_file_path}"
        )
        trimesh_mesh = trimesh.load(original_mesh_file_path, force="mesh")

        trimesh_mesh.export(stl_file_path)

    def _wavefront_copy_of(self, mesh_file_path: str) -> WavefrontCopy:
        """
        Writes an .obj copy of a mesh file MuJoCo cannot read into the asset folder,
        with its material and texture image beside it.

        :param mesh_file_path: The mesh file to copy.
        :return: The copy.
        """
        if mesh_file_path in self._wavefront_copies:
            return self._wavefront_copies[mesh_file_path]
        copy_file_path = Path(
            self._asset_file_path_for(mesh_file_path, suffix=f".{MeshFileFormat.OBJ}")
        )
        text, resources = trimesh.exchange.obj.export_obj(
            trimesh.load(mesh_file_path, force="mesh"),
            return_texture=True,
            mtl_name=copy_file_path.with_suffix(".mtl").name,
        )
        copy_file_path.write_text(text)
        for resource_name, content in resources.items():
            (copy_file_path.parent / resource_name).write_bytes(content)
        images = [
            str(copy_file_path.parent / resource_name)
            for resource_name in resources
            if not resource_name.endswith(".mtl")
        ]
        copy = WavefrontCopy(
            mesh_file_path=str(copy_file_path),
            texture_file_path=images[0] if images else None,
        )
        self._wavefront_copies[mesh_file_path] = copy
        return copy

    def _asset_file_path_for(self, source_file_path: str, suffix: str) -> str:
        """
        The path in the asset folder that a file derived from a source mesh is written
        to, distinct for every source file so that meshes sharing a file name never
        overwrite or reuse each other's derived files.

        :param source_file_path: The mesh the file is derived from.
        :param suffix: The text replacing the source file's extension, such as
            ``".stl"``.
        :return: The path, whose directory exists.
        """
        source_without_extension, _ = os.path.splitext(
            os.path.abspath(source_file_path)
        )
        asset_file_path = os.path.join(
            self.asset_folder_path, source_without_extension.lstrip(os.sep) + suffix
        )
        os.makedirs(os.path.dirname(asset_file_path), exist_ok=True)
        return asset_file_path

    def _mesh_name_for(self, mesh_file_path: str, mesh_scale: list[float]) -> str:
        """
        The name of the mesh asset for a mesh file at a scale, adding the asset to the
        spec if no geom used this file at this scale yet.

        :param mesh_file_path: The mesh file MuJoCo loads.
        :param mesh_scale: The scale along x, y and z.
        :return: The name, unique to this file and scale.
        """
        for mesh in self.spec.meshes:
            if mesh.file == mesh_file_path and numpy.allclose(mesh.scale, mesh_scale):
                return mesh.name
        mesh_name = os.path.splitext(os.path.basename(mesh_file_path))[0]
        if not numpy.allclose(mesh_scale, [1.0, 1.0, 1.0]):
            mesh_name += f"_{'_'.join(map(str, mesh_scale))}"
        if mesh_name in [mesh.name for mesh in self.spec.meshes]:
            mesh_name += f"_{len(self.spec.meshes)}"
        mesh = self.spec.add_mesh(name=mesh_name)
        mesh.file = mesh_file_path
        mesh.scale = mesh_scale
        return mesh_name

    def _thicken_if_near_planar(self, mesh_file_path: str) -> str:
        """
        MuJoCo cannot compute a convex hull for a mesh whose vertices are (near-)
        exactly coplanar - a flat, effectively two-dimensional panel with no enclosed
        volume - and raises "mesh ... has coplanar vertices, cannot compute convex
        hull" rather than compiling. Real CAD furniture sometimes ships geometry this
        way (a door or cover modelled as a single-sided sheet).

        MuJoCo hulls a mesh's vertex positions, not its exact topology, so the fix does
        not need to close the mesh into a topological solid: duplicating every vertex
        and offsetting the two copies apart by a small epsilon along the panel's own
        normal already gives MuJoCo a non-degenerate point cloud to hull, invisibly
        (the offset is a fraction of a millimeter for real furniture-scale meshes).

        The offset is scaled to the mesh's own extent (floored at
        ``planar_thickness_epsilon``): STL's float32 export loses precision at large
        enough coordinate magnitudes, so a mesh authored at a much larger scale than
        typical furniture (seen on one real ArtVIP object, tens of thousands of units
        across) needs a correspondingly larger absolute offset to remain distinguishable
        from zero once exported.

        :param mesh_file_path: The mesh file to check.
        :return: ``mesh_file_path`` unchanged if it already has meaningful thickness,
            otherwise the path to a thickened copy written into the asset folder.
        """
        if mesh_file_path in self._thickened_mesh_paths:
            return self._thickened_mesh_paths[mesh_file_path]

        trimesh_mesh = trimesh.load(mesh_file_path, force="mesh")
        centered_vertices = trimesh_mesh.vertices - trimesh_mesh.vertices.mean(axis=0)
        _, _, principal_axes = numpy.linalg.svd(centered_vertices, full_matrices=False)
        normal = principal_axes[-1]
        projections = centered_vertices @ normal
        thickness = projections.max() - projections.min()
        if thickness > self.planar_thickness_epsilon:
            self._thickened_mesh_paths[mesh_file_path] = mesh_file_path
            return mesh_file_path

        thickened_file_path = self._asset_file_path_for(
            mesh_file_path, suffix="_thickened.stl"
        )
        # Always (re)written rather than reused when already present on disk: two
        # different source meshes can share a basename (e.g. two objects each
        # containing a "cover.stl"), and an existence check keyed only on that name
        # would silently reuse a stale, geometrically wrong thickened mesh left over
        # from a previous build that wrote into the same asset folder.
        extent = numpy.linalg.norm(
            trimesh_mesh.vertices.max(axis=0) - trimesh_mesh.vertices.min(axis=0)
        )
        offset_magnitude = max(self.planar_thickness_epsilon, extent * 1e-5)
        offset = normal * (offset_magnitude / 2.0)
        vertex_count = len(trimesh_mesh.vertices)
        thickened = trimesh.Trimesh(
            vertices=numpy.concatenate(
                [trimesh_mesh.vertices + offset, trimesh_mesh.vertices - offset]
            ),
            faces=numpy.concatenate(
                [trimesh_mesh.faces, trimesh_mesh.faces[:, ::-1] + vertex_count]
            ),
            process=False,
        )
        thickened.export(thickened_file_path)
        self._thickened_mesh_paths[mesh_file_path] = thickened_file_path
        return thickened_file_path

    def _parse_geom(self, geom_props: dict[str, Any]) -> bool:
        """
        Parses the geometry properties for a mesh geom. Adds the mesh to the spec if it doesn't exist.

        :param geom_props: The geometry properties to parse.
        :return: True if the mesh was parsed successfully, False otherwise.
        """
        mesh_entity = geom_props.pop("mesh")
        if isinstance(mesh_entity, Mesh):
            mesh_file_path = str(mesh_entity.local_file)
        else:
            raise NotImplementedError(
                f"Mesh type {type(mesh_entity)} not supported in Mujoco."
            )
        mesh_file_format = Path(mesh_file_path).suffix.removeprefix(".").lower()
        if mesh_file_format == MeshFileFormat.DAE:
            stl_file_path = self._asset_file_path_for(
                mesh_file_path, suffix=f".{MeshFileFormat.STL}"
            )
            # create a .stl mesh from the original .dae mesh, as a replacement. If it not already exists.
            if not os.path.exists(stl_file_path):
                self._create_stl_from_dae_mesh(
                    original_mesh_file_path=mesh_file_path, stl_file_path=stl_file_path
                )
            mesh_file_path = stl_file_path
        texture_file_path = geom_props.pop("texture_file_path", None)
        if mesh_file_format == MeshFileFormat.GLB:
            wavefront_copy = self._wavefront_copy_of(mesh_file_path)
            mesh_file_path = wavefront_copy.mesh_file_path
            texture_file_path = wavefront_copy.texture_file_path

        mesh_file_path = self._thicken_if_near_planar(mesh_file_path)

        mesh_scale = [mesh_entity.scale.x, mesh_entity.scale.y, mesh_entity.scale.z]
        geom_props["meshname"] = self._mesh_name_for(mesh_file_path, mesh_scale)
        if isinstance(texture_file_path, str):
            geom_props["material"] = self._register_texture_material(
                texture_file_path=texture_file_path
            )
        return True

    def _register_texture_material(
        self,
        texture_file_path: str,
        texture_repeat: tuple[float, float] = (1.0, 1.0),
        texture_uniform: bool = False,
    ) -> str:
        """
        Registers a texture and a material referencing it in the spec, unless a texture or
        material of the same derived name is already registered.

        RoboCasa's asset pipeline reuses generic texture basenames (e.g. "T_BC001.png")
        across many unrelated fixtures' own distinct texture files, so the basename alone is
        not a valid dedup/uniqueness key: two different fixtures' textures with the same
        basename would otherwise collide onto whichever one was registered first. Suffixing
        with a hash of the full path keeps the same file deduplicated (reused) while keeping
        different files (even same basename) distinct.

        :param texture_file_path: The texture image's file path.
        :param texture_repeat: How many times the texture tiles across the surface.
        :param texture_uniform: Whether the texture is scaled uniformly across the surface.
        :return: The name of the material referencing the texture, to set on a geom's
            "material" property. Returned even if the texture file does not exist on disk (the
            geom is still given a material name, just one with no actual texture registered).
        """
        path_hash = hashlib.md5(
            os.path.abspath(texture_file_path).encode()
        ).hexdigest()[:8]
        texture_name = (
            f"{os.path.splitext(os.path.basename(texture_file_path))[0]}_{path_hash}"
        )
        material_name = texture_name
        if material_name.startswith("T_"):
            material_name = material_name[2:]
        material_name = f"M_{material_name}"

        texture_already_registered = texture_name in [
            self.spec.textures[i].name for i in range(len(self.spec.textures))
        ]
        material_already_registered = material_name in [
            self.spec.materials[i].name for i in range(len(self.spec.materials))
        ]
        if (
            not texture_already_registered
            and not material_already_registered
            and os.path.exists(texture_file_path)
        ):
            self.spec.add_texture(
                name=texture_name,
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=texture_file_path,
            )
            material = self.spec.add_material(name=material_name)
            material.textures[0] = texture_name
            material.texrepeat = list(texture_repeat)
            material.texuniform = texture_uniform
        return material_name

    def _build_connection(self, connection: Connection):
        MujocoConnectionJoints.of(connection, self.world, self.element_names).build(
            self
        )

    def _build_actuator(self, actuator: Actuator):
        [degree_of_freedom] = actuator.dofs
        self._add_actuator(
            actuator,
            self.element_names.of(actuator),
            self._transmission_for_degree_of_freedom(degree_of_freedom),
        )

    def build_actuator_on_joint(
        self, actuator: Actuator, name: str, joint_name: str
    ) -> None:
        """
        Add a MuJoCo actuator working like one of the world's actuators on another
        joint.

        :param actuator: The actuator of the world whose kind and gains it takes.
        :param name: The name of the MuJoCo actuator.
        :param joint_name: The joint it drives.
        """
        self._add_actuator(
            actuator,
            name,
            {
                MujocoActuatorProperty.TARGET: joint_name,
                MujocoActuatorProperty.TRANSMISSION_TYPE: mujoco.mjtTrn.mjTRN_JOINT,
            },
        )

    def _add_actuator(
        self, actuator: Actuator, name: str, transmission: dict[str, Any]
    ) -> None:
        """
        :param actuator: The actuator of the world whose kind and gains the MuJoCo
            actuator takes.
        :param name: The name of the MuJoCo actuator.
        :param transmission: What the MuJoCo actuator drives.
        """
        actuator_props = MujocoActuatorConverter.convert(actuator)
        actuator_props.pop("dof_names")
        actuator_props["name"] = name
        actuator_props.update(transmission)
        actuator_spec = self.spec.add_actuator(**actuator_props)
        if actuator_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=name,
                entity_type=mujoco.mjtObj.mjOBJ_ACTUATOR,
                action="add",
            )

    def _transmission_for_degree_of_freedom(
        self, degree_of_freedom: DegreeOfFreedom
    ) -> dict[str, Any]:
        """
        :param degree_of_freedom: The degree of freedom an actuator drives.
        :return: The actuator's transmission properties: the joint moving that degree
            of freedom, or the tendon of its name if no joint does.
        """
        connection = next(
            (
                candidate
                for candidate in self.world.connections
                if any(dof.id == degree_of_freedom.id for dof in candidate.dofs)
            ),
            None,
        )
        if connection is None:
            return {
                MujocoActuatorProperty.TARGET: degree_of_freedom.name.name,
                MujocoActuatorProperty.TRANSMISSION_TYPE: mujoco.mjtTrn.mjTRN_TENDON,
            }
        joint_name = MujocoConnectionJoints.of(
            connection, self.world, self.element_names
        ).joint_name_of(degree_of_freedom)
        joint_spec = self._find_entity(
            entity_type=mujoco.mjtObj.mjOBJ_JOINT, entity_name=joint_name
        )
        if joint_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=joint_name,
                entity_type=mujoco.mjtObj.mjOBJ_JOINT,
            )
        return {
            MujocoActuatorProperty.TARGET: joint_spec.name,
            MujocoActuatorProperty.TRANSMISSION_TYPE: mujoco.mjtTrn.mjTRN_JOINT,
        }

    def _build_camera(self, camera: MultiSimCamera):
        camera_name = camera.name
        camera_props = MujocoCameraConverter.convert(camera)
        camera_props.pop("body")
        camera_spec = self.body_spec_of(camera.body).add_camera(**camera_props)
        if camera_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=camera_name,
                entity_type=mujoco.mjtObj.mjOBJ_CAMERA,
                action="add",
            )

    def _build_light(self, light: MultiSimLight):
        """
        Builds a light in the Mujoco spec, attached to its parent body.

        :param light: The light to build.
        """
        light_name = light.name
        light_props = MujocoLightConverter.convert(light)
        light_props.pop("body")
        light_spec = self.body_spec_of(light.body).add_light(**light_props)
        if light_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=light_name,
                entity_type=mujoco.mjtObj.mjOBJ_LIGHT,
                action="add",
            )

    def _build_mujoco_body(
        self,
        entity: KinematicStructureEntity,
        additional_properties: dict[str, Any] | None = None,
    ):
        """
        Builds a kinematic structure entity as a body in the Mujoco spec: MuJoCo has no
        regions, so a region is a body like any other.

        :param entity: The entity to build.
        :param additional_properties: Properties only one kind of entity carries, such
            as a body's gravity compensation, merged in by the caller that knows its
            kind.
        """
        body_name = self.element_names.of(entity)
        if entity is self.world.root:
            return
        body_props = MujocoKinematicStructureEntityConverter.convert(entity)
        body_props["name"] = body_name
        body_props.update(additional_properties or {})
        mujoco_body = entity.get_simulator_property_of_type(MujocoBody)
        if mujoco_body is not None:
            body_props["mocap"] = mujoco_body.motion_capture
        body_spec = self.body_spec_of(entity.parent_connection.parent).add_body(
            **body_props
        )
        if body_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=body_name,
                entity_type=mujoco.mjtObj.mjOBJ_BODY,
                action="add",
            )

    def _build_contact_exclusions(self):
        """
        Tell MuJoCo which pairs of a robot's own bodies never to check for contact: the
        pairs the world's own collision rules allow to collide.

        A description's neighbouring links overlap where they meet, so two links left
        to collide hold the joint between them still with the very contact the rules
        exist to ignore, and no servo can drive it. Only pairs within one robot are
        excluded: a rule excusing two other bodies says the planner need not check
        them, not that they pass through each other.
        """
        body_to_robot = self.world.robot_body_to_robot_mapping
        excluded: set[tuple[str, str]] = set()
        for rule in self.world.collision_manager.ignore_collision_rules:
            rule.update(self.world)
            for collision_check in rule.allowed_collision_pairs:
                self._exclude_robot_contact(
                    collision_check.body_a,
                    collision_check.body_b,
                    body_to_robot,
                    excluded,
                )
            for body in rule.allowed_collision_bodies:
                for other in self.world.bodies_with_collision:
                    self._exclude_robot_contact(body, other, body_to_robot, excluded)

    def _exclude_robot_contact(
        self,
        body_a: Body,
        body_b: Body,
        body_to_robot: dict[Body, Any],
        excluded: set[tuple[str, str]],
    ) -> None:
        """
        Exclude one pair from contact checking if both bodies belong to the same robot
        and the pair was not excluded yet.

        :param body_a: One of the two bodies.
        :param body_b: The other.
        :param body_to_robot: Which robot each body belongs to, if any.
        :param excluded: The pairs excluded so far, extended in place.
        """
        if body_a is body_b:
            return
        robot = body_to_robot.get(body_a)
        if robot is None or body_to_robot.get(body_b) is not robot:
            return
        pair = tuple(
            sorted((self.body_spec_of(body_a).name, self.body_spec_of(body_b).name))
        )
        if pair in excluded:
            return
        excluded.add(pair)
        self.spec.add_exclude(name="_".join(pair), bodyname1=pair[0], bodyname2=pair[1])

    def _build_equalities(self):
        """
        Builds all equalities in the Mujoco spec.
        """
        for mujoco_equality in self.world.simulator_additional_properties:
            if isinstance(mujoco_equality, MujocoEquality):
                equality = self.spec.add_equality()
                equality.type = mujoco_equality.type
                equality.objtype = mujoco_equality.object_type
                equality.name1 = mujoco_equality.name_1
                equality.name2 = mujoco_equality.name_2
                equality.data = mujoco_equality.data

    def _build_tendons(self):
        """
        Builds all tendons in the Mujoco spec.
        """
        for mujoco_tendon in self.world.simulator_additional_properties:
            if isinstance(mujoco_tendon, MujocoTendon):
                tendon: mujoco.MjsTendon = self.spec.add_tendon()
                tendon.name = mujoco_tendon.name
                tendon.actfrclimited = mujoco_tendon.actuator_force_limited
                tendon.actfrcrange = mujoco_tendon.actuator_force_range
                tendon.armature = mujoco_tendon.armature
                tendon.damping = (
                    mujoco_tendon.damping[0]
                    if mujoco.mj_version() < 3007000
                    else mujoco_tendon.damping
                )
                tendon.frictionloss = mujoco_tendon.frictionloss
                tendon.group = mujoco_tendon.group
                tendon.limited = mujoco_tendon.limited
                tendon.margin = mujoco_tendon.margin
                tendon.material = mujoco_tendon.material
                tendon.range = mujoco_tendon.range
                tendon.rgba = mujoco_tendon.rgba.to_rgba()
                tendon.solimp_friction = mujoco_tendon.solver_impedance_friction
                tendon.solimp_limit = mujoco_tendon.solver_impedance_limit
                tendon.solref_friction = mujoco_tendon.solver_reference_friction
                tendon.solref_limit = mujoco_tendon.solver_reference_limit
                tendon.springlength = mujoco_tendon.spring_length
                tendon.stiffness = (
                    mujoco_tendon.stiffness[0]
                    if mujoco.mj_version() < 3007000
                    else mujoco_tendon.stiffness
                )
                tendon.width = mujoco_tendon.width
                for joint_name, joint_coef in mujoco_tendon.joints.items():
                    tendon.wrap_joint(joint_name, joint_coef)

    def body_spec_of(self, entity: KinematicStructureEntity) -> mujoco.MjsBody:
        """
        :param entity: An entity already built into the model.
        :return: The body the entity is built as.
        """
        body_name = self.element_names.of(entity)
        body_spec = self._find_entity(
            entity_type=mujoco.mjtObj.mjOBJ_BODY, entity_name=body_name
        )
        if body_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=body_name, entity_type=mujoco.mjtObj.mjOBJ_BODY
            )
        return body_spec

    def _find_entity(
        self,
        entity_type: mujoco.mjtObj,
        entity_name: str,
    ) -> mujoco.MjsBody | mujoco.MjsGeom | mujoco.MjsJoint | mujoco.MjsSite | None:
        """
        Finds an entity in the Mujoco spec by its type and name.

        :param entity_type: The type of the entity.
        :param entity_name: The name of the entity.
        :return: The entity if found, None otherwise.
        """
        entity_type_str = entity_type.name.replace("mjOBJ_", "").lower()
        if mujoco.mj_version() >= 330:
            return self.spec.__getattribute__(entity_type_str)(entity_name)
        else:
            return self.spec.__getattribute__(f"find_{entity_type_str}")(entity_name)


@dataclass
class EntitySpawner(ABC):
    """
    A spawner to spawn a WorldEntity object in the Multiverse simulator.
    """

    entity_type: ClassVar[type[Any]] = Any
    """
    The type of the entity to spawn.
    """

    element_names: SimulatorModelNames = field(kw_only=True)
    """
    The name every entity is spawned under, shared with the model it is spawned into.
    """

    @classmethod
    def spawn(
        cls,
        simulator: BaseSimulator,
        entity: entity_type,  # type: ignore
        element_names: SimulatorModelNames,
    ) -> bool:
        """
        Spawns a WorldEntity object in the Multiverse simulator.

        :param simulator: The Multiverse simulator to spawn the entity in.
        :param entity: The WorldEntity object to spawn.
        :param element_names: The name every entity is spawned under, shared with the
            model it is spawned into.

        :return: True if the entity was spawned successfully, False otherwise.
        """
        for subclass in recursive_subclasses(cls):
            if (
                not inspect.isabstract(subclass)
                and not inspect.isabstract(subclass.entity_type)
                and type(entity) is subclass.entity_type
            ):
                return subclass(element_names=element_names)._spawn(simulator, entity)
        raise NotImplementedError(f"No converter found for entity type {type(entity)}.")

    @abstractmethod
    def _spawn(self, simulator: BaseSimulator, entity: Any) -> bool:
        """
        The actual spawning method to be implemented by subclasses.

        :param simulator: The Multiverse simulator to spawn the entity in.
        :param entity: The WorldEntity object to spawn.
        :return: True if the entity was spawned successfully, False otherwise.
        """
        raise NotImplementedError


@dataclass
class KinematicStructureEntitySpawner(EntitySpawner):
    """
    A spawner to spawn a KinematicStructureEntity object in the Multiverse simulator.
    """

    entity_type: ClassVar[type[KinematicStructureEntity]] = KinematicStructureEntity
    """
    The type of the entity to spawn.
    """

    def _spawn(
        self, simulator: BaseSimulator, entity: KinematicStructureEntity
    ) -> bool:
        """
        Spawns a KinematicStructureEntity object in the Multiverse simulator including its shapes.

        :param simulator: The Multiverse simulator to spawn the entity in.
        :param entity: The KinematicStructureEntity object to spawn.

        :return: True if the entity and its shapes were spawned successfully, False otherwise.
        """
        return self._spawn_kinematic_structure_entity(
            simulator, entity
        ) and self._spawn_shapes(simulator, entity)

    @abstractmethod
    def _spawn_kinematic_structure_entity(
        self, simulator: BaseSimulator, entity: KinematicStructureEntity
    ) -> bool:
        """
        Spawns a KinematicStructureEntity object in the Multiverse simulator.

        :param simulator: The Multiverse simulator to spawn the entity in.
        :param entity: The KinematicStructureEntity object to spawn.

        :return: True if the entity was spawned successfully, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def _spawn_shapes(
        self, simulator: BaseSimulator, entity: KinematicStructureEntity
    ) -> bool:
        """
        Spawns the shapes of a KinematicStructureEntity object in the Multiverse simulator.

        :param simulator: The Multiverse simulator to spawn the shapes in.
        :param entity: The KinematicStructureEntity object whose shapes to spawn.

        :return: True if all shapes were spawned successfully, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def _spawn_shape(
        self,
        parent: Body | Region,
        simulator: BaseSimulator,
        shape: Shape,
        visible: bool,
        collidable: bool,
    ) -> bool:
        """
        Spawns a shape in the Multiverse simulator and attaches it to its parent body or region.

        :param parent: The parent body or region to attach the shape to.
        :param simulator: The Multiverse simulator to spawn the shape in.
        :param shape: The shape to spawn.
        :param visible: Whether the shape is visible.
        :param collidable: Whether the shape is collidable.

        :return: True if the shape was spawned successfully, False otherwise.
        """
        raise NotImplementedError


@dataclass
class BodySpawner(KinematicStructureEntitySpawner, ABC):
    """
    A spawner to spawn a Body object in the Multiverse simulator.
    """

    entity_type: ClassVar[type[Body]] = Body
    """
    The type of the entity to spawn.
    """

    def _spawn_shapes(self, simulator: BaseSimulator, parent: Body) -> bool:
        return all(
            self._spawn_shape(
                parent=parent,
                simulator=simulator,
                shape=shape,
                visible=shape in parent.visual or not parent.visual,
                collidable=shape in parent.collision,
            )
            for shape in {
                id(s): s for s in parent.visual.shapes + parent.collision.shapes
            }.values()
        )


@dataclass
class RegionSpawner(KinematicStructureEntitySpawner, ABC):
    """
    A spawner to spawn a Region object in the Multiverse simulator.
    """

    entity_type: ClassVar[type[Region]] = Region
    """
    The type of the entity to spawn.
    """

    def _spawn_shapes(self, simulator: BaseSimulator, parent: Region) -> bool:
        return all(
            self._spawn_shape(
                parent=parent,
                simulator=simulator,
                shape=shape,
                visible=True,
                collidable=False,
            )
            for shape in parent.area
        )


@dataclass
class ConnectionSpawner(EntitySpawner):
    """
    A spawner to spawn a Connection object in the simulator.
    """

    entity_type: ClassVar[type[Connection]] = Connection
    """
    The type of the entity to spawn.
    """

    def _spawn(self, simulator: BaseSimulator, entity: Connection) -> bool:
        """
        Spawns a Connection object in the simulator, including its dof and its child body.

        :param simulator: The simulator to spawn the Connection in.
        :param entity: The Connection object to spawn.

        :return: True if the Connection was spawned successfully, False otherwise.
        """
        return self._spawn_connection(simulator, entity)

    @abstractmethod
    def _spawn_connection(
        self, simulator: BaseSimulator, connection: Connection
    ) -> bool:
        """
        Spawns a Connection object in the simulator.

        :param simulator: The simulator to spawn the Connection in.
        :param connection: The Connection object to spawn.

        :return: True if the Connection was spawned successfully, False otherwise.
        """
        raise NotImplementedError


@dataclass
class FixedConnectionSpawner(ConnectionSpawner, ABC):
    """
    A spawner to spawn a FixedConnection object in the simulator.
    """

    entity_type: ClassVar[type[Connection]] = FixedConnection
    """
    The type of the entity to spawn.
    """


@dataclass
class Connection1DOFSpawner(ConnectionSpawner, ABC):
    """
    A spawner to spawn an ActiveConnection1DOF object in the simulator.
    """

    entity_type: ClassVar[type[Connection]] = ActiveConnection1DOF
    """
    The type of the entity to spawn.
    """


@dataclass
class ConnectionPrismaticSpawner(ConnectionSpawner, ABC):
    """
    A spawner to spawn a PrismaticConnection object in the simulator.
    """

    entity_type: ClassVar[type[Connection]] = PrismaticConnection
    """
    The type of the entity to spawn.
    """


@dataclass
class ConnectionRevoluteSpawner(ConnectionSpawner, ABC):
    """
    A spawner to spawn a RevoluteConnection object in the simulator.
    """

    entity_type: ClassVar[type[Connection]] = RevoluteConnection
    """
    The type of the entity to spawn.
    """


@dataclass
class Connection6DOFSpawner(ConnectionSpawner, ABC):
    """
    A spawner to spawn a Connection6DoF object in the simulator.
    """

    entity_type: ClassVar[type[Connection]] = Connection6DoF
    """
    The type of the entity to spawn.
    """


@dataclass
class ActuatorSpawner(EntitySpawner):
    """
    A spawner to spawn an Actuator object in the simulator.
    """

    entity_type: ClassVar[type[Actuator]] = Actuator
    """
    The type of the entity to spawn.
    """

    def _spawn(self, simulator: BaseSimulator, entity: Actuator) -> bool:
        """
        Spawns an Actuator object in the simulator, including its dof.

        :param simulator: The simulator to spawn the entity in.
        :param entity: The Actuator object to spawn.

        :return: True if the entity is spawned successfully, False otherwise.
        """
        return self._spawn_actuator(simulator, entity)

    @abstractmethod
    def _spawn_actuator(self, simulator: BaseSimulator, actuator: Actuator) -> bool:
        """
        Spawns an Actuator object in the simulator.

        :param simulator: The simulator to spawn the entity in.
        :param actuator: The Actuator object to spawn.

        :return: True if the Actuator was spawned successfully, False otherwise.
        """
        raise NotImplementedError


@dataclass
class MujocoEntitySpawner(EntitySpawner, ABC):
    """
    A spawner to spawn a WorldEntity object in the Mujoco simulator.
    """

    def _add_to_simulator(
        self,
        simulator: MujocoSimulator,
        entity: KinematicStructureEntity | Connection | Actuator,
        element_type: MujocoElementType,
        properties: dict[str, Any],
        parent: KinematicStructureEntity | None = None,
    ) -> bool:
        """
        Add an entity to the simulator's model under the name the model knows it by.

        :param simulator: The simulator to add the entity to.
        :param entity: The entity to add.
        :param element_type: The kind of element the entity is added as.
        :param properties: The properties of the element, apart from its name.
        :param parent: The entity whose body the element is added to, if it is added
            to one.
        :return: Whether the simulator added the element.
        """
        result = simulator.add_entity(
            entity_name=self.element_names.of(entity),
            entity_type=element_type,
            entity_properties=properties,
            parent_name=None if parent is None else self.element_names.of(parent),
        )
        return (
            result.type
            == SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        )


@dataclass
class MujocoKinematicStructureEntitySpawner(
    MujocoEntitySpawner, KinematicStructureEntitySpawner, ABC
):
    """
    A spawner to spawn a KinematicStructureEntity object in the Mujoco simulator.
    """

    def _spawn_kinematic_structure_entity(
        self, simulator: MujocoSimulator, entity: KinematicStructureEntity
    ) -> bool:
        kinematic_structure_entity_props = (
            MujocoKinematicStructureEntityConverter.convert(entity)
        )
        del kinematic_structure_entity_props["name"]
        return self._add_to_simulator(
            simulator,
            entity,
            MujocoElementType.BODY,
            kinematic_structure_entity_props,
            parent=entity.parent_connection.parent,
        )

    def _spawn_shape(
        self,
        parent: Body,
        simulator: MujocoSimulator,
        shape: Shape,
        visible: bool,
        collidable: bool,
    ) -> bool:
        shape_props = MujocoGeomConverter.convert(
            shape, visible=visible, collidable=collidable
        )
        shape_name = shape_props.pop("name")
        result = simulator.add_entity(
            entity_name=shape_name,
            entity_type=MujocoElementType.GEOM,
            entity_properties=shape_props,
            parent_name=self.element_names.of(parent),
        )
        return (
            result.type
            == SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        )


@dataclass
class MujocoBodySpawner(MujocoKinematicStructureEntitySpawner, BodySpawner):
    """
    A spawner to spawn a Body object in the Mujoco simulator.
    """

    ...


@dataclass
class MujocoRegionSpawner(MujocoKinematicStructureEntitySpawner, RegionSpawner):
    """
    A spawner to spawn a Region object in the Mujoco simulator.
    """

    ...


@dataclass
class MujocoConnectionSpawner(MujocoEntitySpawner, ConnectionSpawner):
    """
    A spawner to spawn a Connection object in the Mujoco simulator.
    """

    mujoco_joint_converter: ClassVar[type[MujocoJointConverter]] = Any

    def _spawn_connection(
        self, simulator: MujocoSimulator, connection: Connection
    ) -> bool:
        joint_props = Mujoco1DOFJointConverter.convert(connection)
        del joint_props["name"]
        return self._add_to_simulator(
            simulator,
            connection,
            MujocoElementType.JOINT,
            joint_props,
            parent=connection.child,
        )


@dataclass
class MujocoFixedConnectionSpawner(MujocoEntitySpawner, FixedConnectionSpawner):
    """
    This spawner does nothing. FixedConnections are implicitly created in Mujoco.
    """

    def _spawn_connection(
        self, simulator: MujocoSimulator, connection: Connection
    ) -> bool:
        return True


@dataclass
class MujocoPrismaticJointSpawner(MujocoConnectionSpawner, ConnectionPrismaticSpawner):
    """
    A spawner to spawn a PrismaticConnection object in the MuJoCo simulator.
    """

    mujoco_joint_converter: ClassVar[type[MujocoJointConverter]] = (
        MujocoPrismaticJointConverter
    )


@dataclass
class MujocoRevoluteJointSpawner(MujocoConnectionSpawner, ConnectionRevoluteSpawner):
    """
    A spawner to spawn a RevoluteConnection object in the MuJoCo simulator.
    """

    mujoco_joint_converter: ClassVar[type[MujocoJointConverter]] = (
        MujocoRevoluteJointConverter
    )


@dataclass
class MujocoFreejointSpawner(MujocoEntitySpawner, Connection6DOFSpawner):
    """
    A spawner to spawn a Connection6DoF object in the MuJoCo simulator.
    """

    def _spawn_connection(
        self, simulator: MujocoSimulator, connection: Connection
    ) -> bool:
        return self._add_to_simulator(
            simulator,
            connection,
            MujocoElementType.JOINT,
            {"type": mujoco.mjtJoint.mjJNT_FREE},
            parent=connection.child,
        )


@dataclass
class MujocoActuatorSpawner(MujocoEntitySpawner, ActuatorSpawner):
    """
    A spawner to spawn a MujocoActuator object in the MuJoCo simulator.
    """

    def _spawn_actuator(self, simulator: MujocoSimulator, actuator: Actuator) -> bool:
        actuator_props = MujocoActuatorConverter.convert(actuator)
        del actuator_props["name"]
        del actuator_props["dof_names"]
        [degree_of_freedom] = actuator.dofs
        connection = next(
            (
                candidate
                for candidate in actuator._world.connections
                if degree_of_freedom in candidate.dofs
            ),
            None,
        )
        if connection is None:
            raise MujocoEntityNotFoundError(
                entity_name=degree_of_freedom.name.name,
                entity_type=mujoco.mjtObj.mjOBJ_JOINT,
            )
        joint_name = MujocoConnectionJoints.of(
            connection, actuator._world, self.element_names
        ).joint_name_of(degree_of_freedom)
        joint_spec = simulator.get_joint(joint_name=joint_name).result
        if joint_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=joint_name,
                entity_type=mujoco.mjtObj.mjOBJ_JOINT,
            )
        actuator_props[MujocoActuatorProperty.TARGET] = joint_spec.name
        actuator_props[MujocoActuatorProperty.TRANSMISSION_TYPE] = (
            mujoco.mjtTrn.mjTRN_JOINT
        )
        return self._add_to_simulator(
            simulator, actuator, MujocoElementType.ACTUATOR, actuator_props
        )


@dataclass(eq=False)
class _MultiSimStateCallback(StateChangeCallback):
    """
    Sibling callback owned by a :class:`MultiSimSynchronizer`. Forwards
    world-state-change notifications to the synchronizer so it can push the
    new world state into its simulator.
    """

    synchronizer: MultiSimSynchronizer = field(kw_only=True)

    def on_state_change(self, **kwargs):
        self.synchronizer._on_state_change()


@dataclass(eq=False)
class MultiSimSynchronizer(ModelChangeCallback, ABC):
    """
    A callback to synchronize the world model with the simulator.

    Listens to world *model* changes (entity/connection/actuator additions)
    via the :class:`ModelChangeCallback` base and spawns the matching entities
    in the simulator. In addition, a sibling :class:`_MultiSimStateCallback`
    is created in ``__post_init__`` that listens to world *state* changes and
    routes them to :meth:`_on_state_change` so concrete synchronizers can
    push the new state into the simulator (the *world → sim* direction).
    """

    simulator: BaseSimulator = field(kw_only=True)
    """
    The simulator to synchronize with the world.
    """

    entity_converter: type[EntityConverter] = NoneType
    """
    The converter to convert WorldEntity, Shape, and Connection objects to dictionaries of properties for the simulator.
    """

    entity_spawner: type[EntitySpawner] = NoneType
    """
    The spawner to spawn WorldEntity, Shape, and Connection objects in the simulator.
    """

    element_names: SimulatorModelNames = field(kw_only=True)
    """
    The name every entity of the world is built into the simulator's model under.
    """

    _state_callback: _MultiSimStateCallback | None = field(
        init=False, default=None, repr=False
    )
    """
    Sibling state-change callback registered in ``world.state.state_change_callbacks``.
    """

    def __post_init__(self):
        super().__post_init__()
        self._state_callback = _MultiSimStateCallback(
            _world=self._world,
            synchronizer=self,
        )

    def on_model_change(self, **kwargs):
        for modification in self._world._model_manager.model_modification_blocks[-1]:
            if isinstance(modification, AddKinematicStructureEntityModification):
                entity = modification.kinematic_structure_entity
                self.entity_spawner.spawn(
                    simulator=self.simulator,
                    entity=entity,
                    element_names=self.element_names,
                )
            elif isinstance(modification, AddConnectionModification):
                connection = modification.connection
                self.entity_spawner.spawn(
                    simulator=self.simulator,
                    entity=connection,
                    element_names=self.element_names,
                )
            elif isinstance(modification, AddActuatorModification):
                entity = modification.actuator
                self.entity_spawner.spawn(
                    simulator=self.simulator,
                    entity=entity,
                    element_names=self.element_names,
                )

    def stop(self):
        if self._state_callback is not None:
            self._state_callback.stop()
            self._state_callback = None
        super().stop()

    @abstractmethod
    def _on_state_change(self) -> None:
        """
        Push the current ``world.state`` into the backing simulator.

        Called from the thread that mutated ``world.state`` (typically the
        user thread). Concrete subclasses implement the simulator-specific
        write path.
        """
        raise NotImplementedError


@dataclass
class JointBackedConnection:
    """
    A connection paired with the MuJoCo joint that backs it.

    Resolving a connection to its joint costs a name lookup, so the two sync
    directions carry the resolved address alongside the connection rather than
    each looking it up again.
    """

    connection: Connection
    """
    The connection in the world.
    """

    qpos_address: int
    """
    Index at which this joint's values start in ``_mj_data.qpos``.
    """


@dataclass(eq=False)
class MujocoSynchronizer(MultiSimSynchronizer):
    simulator: MujocoSimulator
    entity_converter: type[EntityConverter] = field(default=MujocoConverter)
    entity_spawner: type[EntitySpawner] = field(default=MujocoEntitySpawner)

    UNTHROTTLED_SYNC_RATE_HZ: ClassVar[float] = float("inf")
    """
    Assign this to :attr:`sync_rate_hz` to sync on every single call, i.e. to not throttle the
    *sim → world* direction at all: since ``1 / sync_rate_hz`` is then exactly ``0.0``, the
    "less than 1 / sync_rate_hz seconds elapsed" skip condition in :meth:`_sim_to_world` can
    never trigger. Distinct from ``sync_rate_hz <= 0``, which disables that direction entirely.
    """

    sync_rate_hz: float = 30
    """
    Throttle (in wall-clock Hz) for the *sim → world* direction: how often
    :meth:`_sim_to_world` is allowed to pull ``_mj_data.qpos`` back into
    ``world.state`` from the physics thread. ``_sim_to_world`` is invoked
    after every ``mj_step``, but a call is skipped if less than
    ``1 / sync_rate_hz`` seconds have elapsed since the last successful sync.

    Set ``<= 0`` to **disable the sim → world direction entirely**: no qpos
    values are pulled back, so ``world.state`` will not reflect joint motion
    produced by the simulator (gravity, contacts, actuator dynamics, etc.).
    The opposite *world → sim* direction (driven by ``_on_state_change``) is
    independent of this setting and continues to push ``world.state`` changes
    into MuJoCo regardless. Set to :attr:`UNTHROTTLED_SYNC_RATE_HZ` for the
    opposite extreme: sync on every call, with no throttling at all.
    """

    _last_sync_time: float = field(init=False, default=0.0, repr=False)

    _sent_controls: dict[str, float] = field(
        init=False, default_factory=dict, repr=False
    )
    """
    The control input last handed to each actuator, by the actuator's name, so that
    only a changed one is handed over again.
    """

    def __post_init__(self):
        super().__post_init__()
        self.simulator.read_data_from_simulator = self._sim_to_world

    def _resolve_qpos_address(self, connection: Connection) -> int | None:
        """
        Resolve the qpos address for the MuJoCo joint backing ``connection``,
        or ``None`` if the joint is not present in the model.
        """
        mj_model = self.simulator._mj_model
        joint_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            self.element_names.of(connection),
        )
        if joint_id == -1:
            return None
        return mj_model.jnt_qposadr[joint_id]

    @staticmethod
    def _make_pose_matrix(
        xyz: numpy.ndarray, rotation_matrix: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Build a 4×4 homogeneous pose matrix from a translation and a 3×3
        rotation matrix.
        """
        pose = numpy.eye(4)
        pose[:3, 3] = xyz
        pose[:3, :3] = rotation_matrix
        return pose

    @staticmethod
    def _decompose_pose_matrix(
        pose: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Decompose a 4×4 homogeneous pose matrix into ``(xyz, quat_xyzw)``.
        """
        xyz = pose[:3, 3]
        quat_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        return xyz, quat_xyzw

    def _joint_backed_connections(self) -> Iterator[JointBackedConnection]:
        """
        Yield every connection that a MuJoCo joint can be synced with, paired
        with the qpos address of that joint.

        Fixed connections carry no DoFs, and a connection that does not resolve
        to a joint is not in the compiled model, so neither has anything to
        sync. Both sync directions walk the same set, so they share this.
        """
        for connection in self._world.connections:
            if isinstance(connection, FixedConnection):
                continue
            qpos_address = self._resolve_qpos_address(connection)
            if qpos_address is None:
                continue
            yield JointBackedConnection(
                connection=connection, qpos_address=qpos_address
            )

    @staticmethod
    def _warn_unsupported_connection(direction: str, connection: Connection) -> None:
        """
        Report a connection that has a MuJoCo joint but no sync implementation.

        :param direction: Which way the sync was going, for the message.
        :param connection: The connection that could not be synced.
        """
        logger.warning(
            "%s sync: unsupported connection type %s for joint %s; skipping",
            direction,
            type(connection).__name__,
            connection.name.name,
        )

    def _read_connections_from_qpos(self) -> bool:
        """
        Copy ``_mj_data.qpos`` into ``world.state`` for every joint-backed
        connection.

        Held under ``_model_lock`` so the whole pull sees one coherent
        post-step state rather than a mixture of poses from either side of an
        ``mj_step`` running on the physics thread.

        :return: Whether any connection was read.
        """
        changed = False
        actuator_name_by_dof_id = self._actuator_names_by_degree_of_freedom()
        with self.simulator._model_lock:
            for joint_backed in self._joint_backed_connections():
                connection = joint_backed.connection
                match connection:
                    case Connection6DoF():
                        self._read_6dof_from_qpos(connection, joint_backed.qpos_address)
                        changed = True
                    case ActiveConnection1DOF():
                        # An actuated joint's world position is the set point it was
                        # commanded to, which the position it has reached so far must
                        # not overwrite.
                        if connection.raw_dof.id in actuator_name_by_dof_id:
                            continue
                        self._read_1dof_from_qpos(connection, joint_backed.qpos_address)
                        changed = True
                    case _:
                        self._warn_unsupported_connection("sim→world", connection)
            changed = self._read_measured_positions() or changed
        return changed

    def _read_measured_positions(self) -> bool:
        """
        Copy into ``world.state`` the positions the simulation measures beyond those of
        the joints named after their connections, such as a drive's odometry.

        :return: Whether any position was read.
        """
        changed = False
        state = self._world.state
        for connection in self._world.connections:
            measured_positions = MujocoConnectionJoints.of(
                connection, self._world, self.element_names
            ).measured_positions(self.simulator)
            for degree_of_freedom, position in measured_positions.items():
                state[degree_of_freedom.id].position = position
                changed = True
        return changed

    def _write_connections_to_qpos(
        self, positions: numpy.ndarray, previous_positions: numpy.ndarray
    ) -> None:
        """
        Push ``world.state`` into ``_mj_data.qpos`` for every joint-backed
        connection whose DoF values differ from ``previous_positions``.

        ``_model_lock`` is what serialises access to ``_mj_data`` against the
        physics thread, and holding it here is not just about torn reads: the
        model integrates with RK4, and ``mj_step`` writes the integrated qpos
        back from state it saved at the top of the step, so a write that lands
        mid-step is overwritten and vanishes. The body then simply carries on
        from its old pose, with nothing raised anywhere. Acquired inside
        ``_world_lock`` to match the order ``modify_world`` already
        establishes.

        :param positions: The current ``world.state`` positions.
        :param previous_positions: The positions as of the last notification,
            used to find what changed. Must be the same length as ``positions``.
        """
        state_index = self._world.state._index
        actuator_name_by_dof_id = self._actuator_names_by_degree_of_freedom()
        with self.simulator._model_lock:
            for joint_backed in self._joint_backed_connections():
                connection = joint_backed.connection
                match connection:
                    case Connection6DoF():
                        self._write_6dof_to_qpos(
                            connection,
                            joint_backed.qpos_address,
                            positions,
                            previous_positions,
                            state_index,
                        )
                    case ActiveConnection1DOF():
                        # An actuated joint is commanded by
                        # :meth:`_command_actuators` instead.
                        if connection.raw_dof.id in actuator_name_by_dof_id:
                            continue
                        self._write_1dof_to_qpos(
                            connection,
                            joint_backed.qpos_address,
                            positions,
                            previous_positions,
                            state_index,
                        )
                    case _:
                        self._warn_unsupported_connection("world→sim", connection)
            self._command_actuators()

    def _actuator_names_by_degree_of_freedom(self) -> dict[Any, str]:
        """
        :return: The name of the actuator driving each actuated degree of freedom, by
            the degree of freedom's id.
        """
        return {
            degree_of_freedom.id: self.element_names.of(actuator)
            for actuator in self._world.actuators
            for degree_of_freedom in actuator.dofs
        }

    def _actuator_controls(self) -> dict[str, float]:
        """
        :return: The control input of every actuator, by its name, as the realisation
            of the connection moving its degree of freedom works it out from the
            actuator's set point, and of every actuator a realisation builds itself.
        """
        connection_by_dof_id = {
            degree_of_freedom.id: connection
            for connection in self._world.connections
            for degree_of_freedom in connection.dofs
        }
        controls = {}
        for actuator in self._world.actuators:
            [degree_of_freedom] = actuator.dofs
            connection = connection_by_dof_id.get(degree_of_freedom.id)
            controls[self.element_names.of(actuator)] = (
                actuator.set_point
                if connection is None
                else MujocoConnectionJoints.of(
                    connection, self._world, self.element_names
                ).control_of(actuator)
            )
        for connection in self._world.connections:
            controls.update(
                MujocoConnectionJoints.of(
                    connection, self._world, self.element_names
                ).own_actuator_controls()
            )
        return controls

    def _command_actuators(self) -> None:
        """
        Hand every actuator whose control input changed its new one, instead of writing
        the world's positions into ``qpos``: a position servo's joint then reaches its
        set point through the actuator's own dynamics, as fast and as hard as its gains
        allow, rather than being teleported there, and a velocity servo's joint moves
        at its set point.
        """
        for actuator_name, control in self._actuator_controls().items():
            if self._sent_controls.get(actuator_name) == control:
                continue
            self.simulator.set_actuator_control(
                actuator_name=actuator_name, value=control
            )
            self._sent_controls[actuator_name] = control

    def command_actuators_from_world_state(self) -> None:
        """
        Hand every actuator the control input the world currently commands it to.

        A freshly reset simulation leaves every control input at zero, which would
        send a position servo's joint rushing towards the origin the moment the
        physics starts.
        """
        self._sent_controls.clear()
        self._command_actuators()

    def _read_6dof_from_qpos(
        self, connection: Connection6DoF, qpos_address: int
    ) -> None:
        """
        Copy a 6DoF MuJoCo free-joint qpos block into ``world.state`` for
        ``connection``.

        :param connection: The 6DoF connection whose DoFs are written.
        :param qpos_address: Index of the free joint's 7-value qpos block.
        """
        mj_data = self.simulator._mj_data
        state = self._world.state

        xyz = mj_data.qpos[qpos_address : qpos_address + 3]
        qwxyz = mj_data.qpos[qpos_address + 3 : qpos_address + 7]

        mj_T_world = self._make_pose_matrix(
            xyz,
            Rotation.from_quat(qwxyz.tolist(), scalar_first=True).as_matrix(),
        )
        parent_T_conn = connection.parent_T_connection_expression.to_np()
        conn_T_child = inverse_frame(parent_T_conn) @ mj_T_world
        dof_xyz, dof_quat_xyzw = self._decompose_pose_matrix(conn_T_child)

        state[connection.x.id].position = float(dof_xyz[0])
        state[connection.y.id].position = float(dof_xyz[1])
        state[connection.z.id].position = float(dof_xyz[2])
        state[connection.qw.id].position = float(dof_quat_xyzw[3])
        state[connection.qx.id].position = float(dof_quat_xyzw[0])
        state[connection.qy.id].position = float(dof_quat_xyzw[1])
        state[connection.qz.id].position = float(dof_quat_xyzw[2])

    def _read_1dof_from_qpos(
        self, connection: ActiveConnection1DOF, qpos_address: int
    ) -> None:
        """
        Copy a single MuJoCo qpos slot into ``world.state`` for ``connection``.

        :param connection: The 1DoF connection whose DoF is written.
        :param qpos_address: Index of the joint's single qpos slot.
        """
        self._world.state[connection.raw_dof.id].position = float(
            self.simulator._mj_data.qpos[qpos_address]
        )

    def _sim_to_world(self) -> None:
        """
        Copy ``_mj_data.qpos`` back into ``world.state``.

        Called by :meth:`physics_simulators.base_simulator.BaseSimulator.step`
        on the physics thread after every ``mj_step``. Throttled to
        ``sync_rate_hz`` wall-clock Hz. The sibling state-change callback is
        paused across the write so our own ``notify_state_change`` does not
        echo back into :meth:`_on_state_change`.

        The whole pull runs under ``World._world_lock``. ``WorldState`` takes
        that lock per accessor, so holding it only for the individual writes
        would let a writer thread land in the middle of this pull: the pose it
        just wrote gets overwritten here with the pre-write qpos values, and
        the notification that would have pushed that pose into MuJoCo is
        swallowed by the ``pause()`` below. The write is then lost without a
        trace, because the closing ``update_previous_world_state`` rebases the
        diff baseline onto what this pull just wrote. ``_world_lock`` is the
        outermost lock in the codebase -- ``modify_world`` takes it before the
        simulator's ``_model_lock`` -- so acquiring it here preserves that
        order.
        """
        if self.sync_rate_hz <= 0:
            return
        now = time.time()
        if now - self._last_sync_time < 1.0 / self.sync_rate_hz:
            return
        self._last_sync_time = now

        with self._world._world_lock:
            self._state_callback.pause()
            try:
                if self._read_connections_from_qpos():
                    self._world.notify_state_change()
                    self._state_callback.update_previous_world_state()
            finally:
                # Always resume: a callback left paused by an exception would
                # silently disable the world -> sim direction for the rest of
                # the run.
                self._state_callback.resume()

    def _write_6dof_to_qpos(
        self,
        connection: Connection6DoF,
        qpos_address: int,
        positions: numpy.ndarray,
        previous_positions: numpy.ndarray,
        state_index: dict[Any, int],
    ) -> None:
        """
        Push the 6DoF world state for ``connection`` into the MuJoCo qpos
        block at ``qpos_address``. No-op if the DoF values match the previous
        snapshot within tolerance.

        :param connection: The 6DoF connection whose pose is pushed.
        :param qpos_address: Index of the free joint's 7-value qpos block.
        :param positions: The current ``world.state`` positions.
        :param previous_positions: The positions as of the last notification,
            compared against ``positions`` to decide whether to write.
        :param state_index: Maps a DoF id to its column in those two arrays.
        """
        ix = state_index[connection.x.id]
        iy = state_index[connection.y.id]
        iz = state_index[connection.z.id]
        iqx = state_index[connection.qx.id]
        iqy = state_index[connection.qy.id]
        iqz = state_index[connection.qz.id]
        iqw = state_index[connection.qw.id]

        dof_indices = [ix, iy, iz, iqx, iqy, iqz, iqw]
        if numpy.allclose(
            positions[dof_indices],
            previous_positions[dof_indices],
            atol=1e-4,
            rtol=1e-4,
        ):
            return

        conn_T_child = self._make_pose_matrix(
            numpy.array([positions[ix], positions[iy], positions[iz]]),
            Rotation.from_quat(
                [positions[iqx], positions[iqy], positions[iqz], positions[iqw]]
            ).as_matrix(),
        )
        parent_T_conn = connection.parent_T_connection_expression.to_np()
        mj_xyz, mj_quat_xyzw = self._decompose_pose_matrix(parent_T_conn @ conn_T_child)

        mj_data = self.simulator._mj_data
        mj_data.qpos[qpos_address + 0] = mj_xyz[0]
        mj_data.qpos[qpos_address + 1] = mj_xyz[1]
        mj_data.qpos[qpos_address + 2] = mj_xyz[2]
        mj_data.qpos[qpos_address + 3] = mj_quat_xyzw[3]
        mj_data.qpos[qpos_address + 4] = mj_quat_xyzw[0]
        mj_data.qpos[qpos_address + 5] = mj_quat_xyzw[1]
        mj_data.qpos[qpos_address + 6] = mj_quat_xyzw[2]

    def _write_1dof_to_qpos(
        self,
        connection: ActiveConnection1DOF,
        qpos_address: int,
        positions: numpy.ndarray,
        previous_positions: numpy.ndarray,
        state_index: dict[Any, int],
    ) -> None:
        """
        Push the 1DoF world state for ``connection`` into the MuJoCo qpos slot
        at ``qpos_address``. No-op if the DoF value is unchanged.

        :param connection: The 1DoF connection whose value is pushed.
        :param qpos_address: Index of the joint's single qpos slot.
        :param positions: The current ``world.state`` positions.
        :param previous_positions: The positions as of the last notification,
            compared against ``positions`` to decide whether to write.
        :param state_index: Maps a DoF id to its column in those two arrays.
        """
        idx = state_index[connection.raw_dof.id]
        if positions[idx] == previous_positions[idx]:
            return
        self.simulator._mj_data.qpos[qpos_address] = positions[idx]

    def _on_state_change(self) -> None:
        """
        Push ``world.state`` into ``_mj_data.qpos`` for every connection whose
        DoF values changed since the last notification. Only non-fixed
        connections that resolve to a MuJoCo joint are pushed.

        Runs under ``World._world_lock`` so the diff against
        ``previous_world_state_data`` and the qpos writes it produces cannot be
        interleaved with :meth:`_sim_to_world` on the physics thread. Callers
        that mutate ``world.state`` already hold the lock, and it is reentrant,
        so this is free on that path.
        """
        with self._world._world_lock:
            positions = self._world.state.positions
            previous_positions = self._state_callback.previous_world_state_data

            if len(positions) != len(previous_positions):
                # Model shape changed since the last notification (e.g. a spawn
                # just added DoFs). The spawner already wrote the initial qpos
                # for the new entities; just rebase the diff and return.
                self._state_callback.update_previous_world_state()
                return

            self._write_connections_to_qpos(positions, previous_positions)
            self._state_callback.update_previous_world_state()

    def stop(self):
        if "read_data_from_simulator" in self.simulator.__dict__:
            del self.simulator.read_data_from_simulator
        super().stop()


class MultiSim(ABC):
    """
    Class to handle the simulation of a world using the simulator.
    """

    simulator_class: ClassVar[type[BaseSimulator]]
    """
    The class of the simulator to use.
    """

    synchronizer_class: ClassVar[type[MultiSimSynchronizer]]
    """
    The class of the MultiSimSynchronizer to use.
    """

    builder_class: ClassVar[type[MultiSimBuilder]]
    """
    The class of the MultiSimBuilder to use.
    """

    world: World
    """
    The simulated world.
    """

    simulator: BaseSimulator
    """
    The simulator instance.
    """

    synchronizer: MultiSimSynchronizer
    """
    The MultiSimSynchronizer instance.
    """

    file_path: Path
    """
    The file the world is saved to for the simulator to load.
    """

    _scene_directory: tempfile.TemporaryDirectory | None
    """
    The temporary directory holding the scene file and its assets when no file path was
    given, removed together with this simulation; ``None`` otherwise.
    """

    def __init__(
        self,
        world: World,
        headless: bool = False,
        step_size: float = 1e-3,
        file_path: Path | None = None,
        **kwargs,
    ):
        """
        Initializes the MultiSim class.

        :param world: The world to simulate.
        :param headless: Whether to run the simulation in headless mode.
        :param step_size: The step size for the simulation.
        :param file_path: The file to save the world to. If not given, every simulation
            saves its world to its own temporary directory, so that simulations built at
            the same time, e.g. by parallel test workers, never overwrite each other's
            scene.
        """
        self.world = world
        if file_path is None:
            self._scene_directory = tempfile.TemporaryDirectory(
                prefix=f"{type(self).__name__.lower()}_"
            )
            file_path = Path(self._scene_directory.name) / "scene.xml"
        else:
            self._scene_directory = None
        self.file_path = Path(file_path)
        builder = self.builder_class()
        builder.build_world(world=world, file_path=str(self.file_path))
        self.simulator = self.simulator_class(
            file_path=str(self.file_path),
            _headless=headless,
            _step_size=step_size,
            config=kwargs,
        )
        self.synchronizer = self.synchronizer_class(
            _world=world,
            simulator=self.simulator,
            element_names=builder.element_names,
        )

    def start_simulation(self, constraints: SimulatorConstraints | None = None):
        """
        Starts the simulation. This will start one physics simulation thread and render it at 60Hz.

        :param constraints: The constraints to apply to the simulation.
        """
        assert self.simulator.state != SimulatorState.RUNNING, (
            "Simulation is already running."
        )
        self.simulator.start(constraints=constraints)

    def stop_simulation(self):
        """
        Stops the simulation. This will stop the physics simulation and the rendering.
        """
        self.synchronizer.stop()
        self.simulator.stop()

    def pause_simulation(self):
        """
        Pauses the simulation. This will pause the physics simulation but not the rendering.
        """
        if self.simulator.state != SimulatorState.PAUSED:
            self.simulator.pause()

    def unpause_simulation(self):
        """
        Unpauses the simulation. This will unpause the physics simulation.
        """
        if self.simulator.state == SimulatorState.PAUSED:
            self.simulator.unpause()

    def reset_simulation(self):
        """
        Resets the simulation. This will reset the physics simulation to the initial state.
        """
        self.simulator.reset()

    def is_running(self):
        return self.simulator.state == SimulatorState.RUNNING


class MujocoSim(MultiSim):
    simulator_class: ClassVar[type[BaseSimulator]] = MujocoSimulator
    synchronizer_class: ClassVar[type[MultiSimSynchronizer]] = MujocoSynchronizer
    builder_class: ClassVar[type[MultiSimBuilder]] = MujocoBuilder
    simulator: MujocoSimulator
    synchronizer: MujocoSynchronizer

    def start_stepped_simulation(self) -> None:
        """
        Start the simulation without a physics thread: the physics advances only in
        :meth:`step_simulation`, so a controller can drive the world between steps, in
        lockstep with the physics, and reads the world back after every step rather
        than at the synchronizer's throttled rate.

        Every servo is handed the position its joint currently holds in the world, so
        nothing rushes towards zero the moment the physics starts.

        :raises SimulationAlreadyRunningError: If the simulation is already running.
        """
        if self.simulator.state == SimulatorState.RUNNING:
            raise SimulationAlreadyRunningError(self.world.root.name.name)
        self.synchronizer.sync_rate_hz = MujocoSynchronizer.UNTHROTTLED_SYNC_RATE_HZ
        self.simulator.start(simulate_in_thread=False, render_in_thread=False)
        self.synchronizer.command_actuators_from_world_state()

    def step_simulation(self, duration: timedelta) -> None:
        """
        Advance the physics of a simulation started with
        :meth:`start_stepped_simulation` and refresh the viewer.

        :param duration: How much simulated time to advance.
        :raises SimulationNotStartedError: If the simulation is not running.
        """
        if self.simulator.state != SimulatorState.RUNNING:
            raise SimulationNotStartedError(self.world.root.name.name)
        for _ in range(round(duration.total_seconds() / self.simulator.step_size)):
            self.simulator.step()
        self.simulator.render()

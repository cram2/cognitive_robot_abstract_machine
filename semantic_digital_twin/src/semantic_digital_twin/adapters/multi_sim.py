import logging
import inspect
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from types import NoneType
from typing_extensions import Dict, List, Any, ClassVar, Type, Optional, Union, Self

import numpy
from mujoco_connector import MultiverseMujocoConnector
import mujoco
from multiverse_simulator import (
    MultiverseSimulator,
    MultiverseSimulatorState,
    MultiverseViewer,
    MultiverseAttribute,
    MultiverseCallbackResult,
)
from krrood.utils import recursive_subclasses
from scipy.spatial.transform import Rotation
from trimesh.visual import TextureVisuals

from ..callbacks.callback import ModelChangeCallback
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Quaternion,
)
from ..world import World
from ..world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    ActiveConnection1DOF,
    FixedConnection,
    Connection6DoF,
)
from ..world_description.geometry import (
    Box,
    Cylinder,
    Sphere,
    Shape,
    FileMesh,
    TriangleMesh,
    Mesh,
)
from ..world_description.world_entity import (
    Region,
    Body,
    KinematicStructureEntity,
    Connection,
    WorldEntity,
    Actuator,
    SemanticAnnotation,
)
from ..world_description.world_modification import (
    AddKinematicStructureEntityModification,
    AddActuatorModification,
)

logger = logging.getLogger(__name__)


def cas_pose_to_list(pose: HomogeneousTransformationMatrix) -> List[float]:
    """
    Converts a CAS TransformationMatrix to a list of 7 floats (position + quaternion).

    :param pose: The CAS TransformationMatrix to convert.
    :return: A list of 7 floats ([px, py, pz, qw, qx, qy, qz]) representing the position and quaternion.
    """
    pose = pose.evaluate()
    pos = pose[:3, 3]
    rotation_matrix = pose[:3, :3]
    quat = Rotation.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    return [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]


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


class MultiSimError(Exception):
    """Base class for all MultiSim-related exceptions."""


@dataclass(eq=False)
class MultiSimCamera(SemanticAnnotation):
    """Semantic annotation declaring that a Body is a MultiSimCamera."""

    body: Body = field(kw_only=True)
    """
    The body which is the camera
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

    diagonal_inertia: List[float]
    """
    The diagonal inertia tensor in the form [Ixx, Iyy, Izz].
    """

    def __post_init__(self):
        assert self.mass > 0, "Mass must be positive."
        assert len(self.diagonal_inertia) == 3, "Diagonal inertia must have 3 elements."
        assert all(
            i >= 0 for i in self.diagonal_inertia
        ), "Inertia values must be non-negative."

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


class EntityConverter(ABC):
    """
    A converter to convert an entity object (WorldEntity, Shape, Connection) to a dictionary of properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Any]] = Any
    """
    The type of the entity to convert.
    """

    name_str: str = "name"
    """
    The key for the name property in the output dictionary.
    """

    @classmethod
    def convert(cls, entity: entity_type, **kwargs) -> Dict[str, Any]:  # type: ignore
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

    def _convert(self, entity: entity_type, **kwargs) -> Dict[str, Any]:  # type: ignore
        """
        The actual conversion method to be implemented by subclasses.

        :param entity: The object to convert.
        :return: A dictionary of properties, by default containing the name.
        """
        return {
            self.name_str: (
                entity.name.name
                if hasattr(entity, "name") and isinstance(entity.name, PrefixedName)
                else f"{type(entity).__name__.lower()}_{id(entity)}"
            )
        }

    @abstractmethod
    def _post_convert(
        self, entity: entity_type, entity_props: Dict[str, Any], **kwargs  # type: ignore
    ) -> Dict[str, Any]:
        """
        Post-processes the converted entity properties. This method can be overridden by subclasses to update the properties after conversion.

        :param entity: The object that was converted.
        :param entity_props: The dictionary of properties that was converted.
        :return: The updated dictionary of properties.
        """
        raise NotImplementedError


class KinematicStructureEntityConverter(EntityConverter, ABC):
    """
    Converts a KinematicStructureEntity object to a dictionary of body properties for Multiverse simulator.
    For inheriting classes, the following string attributes must be defined:
    - pos_str: The key for the position property in the output dictionary.
    - quat_str: The key for the quaternion property in the output dictionary.
    """

    entity_type: ClassVar[Type[KinematicStructureEntity]] = KinematicStructureEntity
    pos_str: str
    quat_str: str

    def _convert(self, entity: entity_type, **kwargs) -> Dict[str, Any]:
        """
        Converts a KinematicStructureEntity object to a dictionary of body properties for Multiverse simulator.

        :param entity: The KinematicStructureEntity object to convert.
        :return: A dictionary of body properties, by default containing position and quaternion.
        """

        kinematic_structure_entity_props = EntityConverter._convert(self, entity)
        px, py, pz, qw, qx, qy, qz = cas_pose_to_list(
            entity.parent_connection.origin_expression
        )
        kinematic_structure_entity_pos = [px, py, pz]
        kinematic_structure_entity_quat = [qw, qx, qy, qz]
        kinematic_structure_entity_props.update(
            {
                self.pos_str: kinematic_structure_entity_pos,
                self.quat_str: kinematic_structure_entity_quat,
            }
        )
        return kinematic_structure_entity_props


class BodyConverter(KinematicStructureEntityConverter, ABC):
    """
    Converts a Body object to a dictionary of body properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[WorldEntity]] = Body
    """
    The type of the entity to convert.
    """

    # Attributes for specifying body properties in the Mujoco simulator.

    mass_str: str
    """
    The key for the mass property in the output dictionary.
    """

    inertia_pos_str: str
    """
    The key for the inertia position property in the output dictionary.
    """

    inertia_quat_str: str
    """
    The key for the inertia quaternion property in the output dictionary.
    """

    diagonal_inertia_str: str
    """
    The key for the diagonal inertia tensor property in the output dictionary.
    """

    def _convert(self, entity: Body, **kwargs) -> Dict[str, Any]:
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
            inertia_quat = principal_axes.to_rotation_matrix().to_quaternion().to_np()
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


class RegionConverter(KinematicStructureEntityConverter, ABC):
    """
    Converts a Region object to a dictionary of region properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[WorldEntity]] = Region
    """
    The type of the entity to convert.
    """


class ShapeConverter(EntityConverter, ABC):
    """
    Converts a Shape object to a dictionary of shape properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Shape]] = Shape
    """
    The type of the entity to convert.
    """

    pos_str: str
    """
    The key for the shape position property in the output dictionary.
    """

    quat_str: str
    """
    The key for the shape quaternion property in the output dictionary.
    """

    rgba_str: str
    """
    The key for the shape RGBA color property in the output dictionary.
    """

    def _convert(self, entity: Shape, **kwargs) -> Dict[str, Any]:
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
        return geom_props


class BoxConverter(ShapeConverter, ABC):
    """
    Converts a Box object to a dictionary of box properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Box]] = Box


class SphereConverter(ShapeConverter, ABC):
    """
    Converts a Sphere object to a dictionary of sphere properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Sphere]] = Sphere


class CylinderConverter(ShapeConverter, ABC):
    """
    Converts a Cylinder object to a dictionary of cylinder properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Cylinder]] = Cylinder


class MeshConverter(ShapeConverter, ABC):
    """
    Converts a Mesh object to a dictionary of mesh properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[FileMesh]] = FileMesh


class TriangleMeshConverter(ShapeConverter, ABC):
    """
    Converts a Mesh object to a dictionary of mesh properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[TriangleMesh]] = TriangleMesh


class ConnectionConverter(EntityConverter, ABC):
    """
    Converts a Connection object to a dictionary of joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Connection]] = Connection
    """
    The type of the entity to convert.
    """

    def _convert(self, entity: Connection, **kwargs) -> Dict[str, Any]:
        """
        Converts a Connection object to a dictionary of joint properties for Multiverse simulator.

        :param entity: The Connection object to convert.
        :return: A dictionary of joint properties.
        """
        return EntityConverter._convert(self, entity)


class Connection1DOFConverter(ConnectionConverter, ABC):
    """
    Converts an ActiveConnection1DOF object to a dictionary of joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[ActiveConnection1DOF]] = ActiveConnection1DOF
    """
    The type of the entity to convert.
    """

    axis_str: str
    """
    The key for the joint axis property in the output dictionary.
    """

    range_str: str
    """
    The key for the joint range property in the output dictionary.
    """

    pos_str: str
    """
    The key for the joint position property in the output dictionary.
    """

    quat_str: str
    """
    The key for the joint quaternion property in the output dictionary.
    """

    armature_str: str
    """
    The key for the joint armature property in the output dictionary.
    """

    dry_friction_str: str
    """
    The key for the joint dry friction property in the output dictionary.
    """

    damping_str: str
    """
    The key for the joint damping property in the output dictionary.
    """

    def _convert(self, entity: ActiveConnection1DOF, **kwargs) -> Dict[str, Any]:
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
        joint_props.update(
            {
                self.pos_str: joint_pos,
                self.quat_str: joint_quat,
                self.axis_str: entity.axis.to_np().tolist()[:3],
                self.range_str: [dof.lower_limits.position, dof.upper_limits.position],
                self.armature_str: entity.dynamics.armature,
                self.dry_friction_str: entity.dynamics.dry_friction,
                self.damping_str: entity.dynamics.damping,
            }
        )
        if dof.name.name != joint_props["name"]:
            joint_props["equality_joint"] = {
                "joint": dof.name.name,
                "data": [entity.offset, entity.multiplier, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            }
        return joint_props


class ConnectionRevoluteConverter(Connection1DOFConverter, ABC):
    """
    Converts a RevoluteConnection object to a dictionary of revolute joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[RevoluteConnection]] = RevoluteConnection
    """
    The type of the entity to convert.
    """


class ConnectionPrismaticConverter(Connection1DOFConverter, ABC):
    """
    Converts a PrismaticConnection object to a dictionary of prismatic joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[PrismaticConnection]] = PrismaticConnection
    """
    The type of the entity to convert.
    """


class Connection6DOFConverter(ConnectionConverter, ABC):
    """
    Converts a Connection6DoF object to a dictionary of 6DoF joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Connection6DoF]] = Connection6DoF
    """
    The type of the entity to convert.
    """

    def _convert(self, entity: Connection6DoF, **kwargs) -> Dict[str, Any]:
        """
        Converts a Connection6DoF object to a dictionary of joint properties for Multiverse simulator.

        :param entity: The Connection6DoF object to convert.
        :return: A dictionary of joint properties.
        """
        joint_props = ConnectionConverter._convert(self, entity)
        assert len(entity.dofs) == 7, "Connection6DoF must have exactly six DOFs."
        return joint_props


class ActuatorConverter(EntityConverter, ABC):
    """
    Converts an Actuator object to a dictionary of actuator properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Actuator]] = Actuator
    """
    The type of the entity to convert.
    """

    def _convert(self, entity: Actuator, **kwargs) -> Dict[str, Any]:
        """
        Converts an Actuator object to a dictionary of joint properties for Multiverse simulator.

        :param entity: The Actuator object to convert.
        :return: A dictionary of actuator properties, by default containing list of DOF names.
        """
        actuator_props = EntityConverter._convert(self, entity)
        actuator_props["dof_names"] = [dof.name.name for dof in entity.dofs]
        return actuator_props


class CameraConverter(EntityConverter, ABC):
    """
    Converts an Camera object to a dictionary of actuator properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[MultiSimCamera]] = MultiSimCamera
    """
    The type of the entity to convert.
    """

    def _convert(self, entity: MultiSimCamera, **kwargs) -> Dict[str, Any]:
        """
        Converts a Camera object to a dictionary of camera properties for Multiverse simulator.

        :param entity: The Camera object to convert.
        :return: A dictionary of camer properties, by default containing list of DOF names.
        """
        camera_props = EntityConverter._convert(self, entity)
        camera_props["body"] = entity.body.name.name
        return camera_props


class MujocoError(MultiSimError):
    """Base class for all MuJoCo-related exceptions."""


class MujocoEntityNotFoundError(MujocoError):
    """Raised when a MuJoCo entity of a given type and name cannot be found."""

    def __init__(
        self, entity_name: str, entity_type: mujoco.mjtObj, action: str = "find"
    ):
        message = f"Failed to {action}: type={entity_type}, name='{entity_name}'"
        super().__init__(message)


@dataclass(eq=False)
class MujocoActuator(Actuator):
    """
    Represents a MuJoCo-specific actuator in the world model.
    For more information, see: https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general
    """

    activation_limited: mujoco.mjtLimited = mujoco.mjtLimited.mjLIMITED_AUTO
    """
    If mujoco.mjtLimited.mjLIMITED_TRUE, the internal state (activation) associated with this actuator is automatically clamped to actrange at runtime. 
    If mujoco.mjtLimited.mjLIMITED_FALSE, activation clamping is disabled. 
    If mujoco.mjtLimited.mjLIMITED_AUTO and autolimits is set in compiler, activation clamping will automatically be set to mujoco.mjtLimited.mjLIMITED_TRUE if activation_range is defined without explicitly setting this attribute to mujoco.mjtLimited.mjLIMITED_TRUE. 
    """

    activation_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    """
    Range for clamping the activation state. The first value must be no greater than the second value.
    """

    ctrl_limited: mujoco.mjtLimited = mujoco.mjtLimited.mjLIMITED_AUTO
    """
    If mujoco.mjtLimited.mjLIMITED_TRUE, the control input to this actuator is automatically clamped to ctrl_range at runtime. 
    If mujoco.mjtLimited.mjLIMITED_FALSE, control input clamping is disabled. 
    If mujoco.mjtLimited.mjLIMITED_AUTO and autolimits is set in compiler, control clamping will automatically be set to mujoco.mjtLimited.mjLIMITED_TRUE if ctrl_range is defined without explicitly setting this attribute to mujoco.mjtLimited.mjLIMITED_TRUE.
    """

    ctrl_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    """
    The range of the control input.
    """

    force_limited: mujoco.mjtLimited = mujoco.mjtLimited.mjLIMITED_AUTO
    """
    If mujoco.mjtLimited.mjLIMITED_TRUE, the force output of this actuator is automatically clamped to force_range at runtime. 
    If mujoco.mjtLimited.mjLIMITED_FALSE, force clamping is disabled. 
    If mujoco.mjtLimited.mjLIMITED_AUTO and autolimits is set in compiler, force clamping will automatically be set to mujoco.mjtLimited.mjLIMITED_TRUE if force_range is defined without explicitly setting this attribute to mujoco.mjtLimited.mjLIMITED_TRUE.
    """

    force_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    """
    Range for clamping the force output. The first value must be no greater than the second value.
    """

    bias_parameters: List[float] = field(default_factory=lambda: [0.0] * 10)
    """
    Bias parameters. The affine bias type uses three parameters.
    """

    bias_type: mujoco.mjtBias = mujoco.mjtBias.mjBIAS_NONE
    """
    The keywords have the following meaning:
    mujoco.mjtBias.mjBIAS_NONE:     bias_term = 0
    mujoco.mjtBias.mjBIAS_AFFINE:   bias_term = biasprm[0] + biasprm[1]*length + biasprm[2]*velocity
    mujoco.mjtBias.mjBIAS_MUSCLE:   bias_term = mju_muscleBias(…)
    mujoco.mjtBias.mjBIAS_USER:     bias_term = mjcb_act_bias(…)
    """

    dynamics_parameters: List[float] = field(default_factory=lambda: [1.0] + [0.0] * 9)
    """
    Activation dynamics parameters.
    """

    dynamics_type: mujoco.mjtDyn = mujoco.mjtDyn.mjDYN_NONE
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

    gain_parameters: List[float] = field(default_factory=lambda: [0.0] * 10)
    """
    Gain parameters.
    """

    gain_type: mujoco.mjtGain = mujoco.mjtGain.mjGAIN_FIXED
    """
    The gain and bias together determine the output of the force generation mechanism, which is currently assumed to be affine.
    The keywords have the following meaning:
    mujoco.mjtGain.mjGAIN_FIXED:    gain_term = gainprm[0]
    mujoco.mjtGain.mjGAIN_AFFINE:   gain_term = gain_prm[0] + gain_prm[1]*length + gain_prm[2]*velocity
    mujoco.mjtGain.mjGAIN_MUSCLE:   gain_term = mju_muscleGain(…)
    mujoco.mjtGain.mjGAIN_USER:     gain_term = mjcb_act_gain(…)
    """

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["activation_limited"] = self.activation_limited.value
        result["activation_range"] = self.activation_range
        result["ctrl_limited"] = self.ctrl_limited.value
        result["ctrl_range"] = self.ctrl_range
        result["force_limited"] = self.force_limited.value
        result["force_range"] = self.force_range
        result["bias_parameters"] = self.bias_parameters
        result["bias_type"] = self.bias_type.value
        result["dynamics_parameters"] = self.dynamics_parameters
        result["dynamics_type"] = self.dynamics_type.value
        result["gain_parameters"] = self.gain_parameters
        result["gain_type"] = self.gain_type.value
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        actuator = super()._from_json(data, **kwargs)
        actuator.activation_limited = mujoco.mjtLimited(data["activation_limited"])
        actuator.activation_range = data["activation_range"]
        actuator.ctrl_limited = mujoco.mjtLimited(data["ctrl_limited"])
        actuator.ctrl_range = data["ctrl_range"]
        actuator.force_limited = mujoco.mjtLimited(data["force_limited"])
        actuator.force_range = data["force_range"]
        actuator.bias_parameters = data["bias_parameters"]
        actuator.bias_type = mujoco.mjtBias(data["bias_type"])
        actuator.dynamics_parameters = data["dynamics_parameters"]
        actuator.dynamics_type = mujoco.mjtDyn(data["dynamics_type"])
        actuator.gain_parameters = data["gain_parameters"]
        actuator.gain_type = mujoco.mjtGain(data["gain_type"])
        return actuator


@dataclass(eq=False)
class MujocoCamera(MultiSimCamera):
    """Semantic annotation declaring that a Body is a MujocoCamera."""

    mode: mujoco.mjtCamLight = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    orthographic: bool = False
    fovy: float = 45.0
    resolution: list = field(default_factory=lambda: [1, 1])
    focal_length: list = field(default_factory=lambda: [0, 0])
    focal_pixel: list = field(default_factory=lambda: [0, 0])
    principal_length: list = field(default_factory=lambda: [0, 0])
    principal_pixel: list = field(default_factory=lambda: [0, 0])
    sensor_size: list = field(default_factory=lambda: [0, 0])
    ipd: float = 0.068
    pos: list = field(default_factory=lambda: [0, 0, 0])
    quat: list = field(default_factory=lambda: [1, 0, 0, 0])


@dataclass(eq=False)
class MujocoEquality(SemanticAnnotation):
    """
    Semantic annotation declaring that two MuJoCo entities are constrained.
    """

    type: mujoco.mjtEq = field(kw_only=True)
    """
    The type of the equality constraint.
    """

    obj_type: mujoco.mjtObj = field(kw_only=True)
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

    data: List[float] = field(kw_only=True)
    """
    The data associated with the equality constraint.
    """


@dataclass(eq=False)
class MujocoMocapBody(SemanticAnnotation):
    """
    Semantic annotation declaring that a Body is a MujocoMocapBody.
    """

    body: Body = field(kw_only=True)
    """
    The body which is a MujocoMocapBody.
    """


class MujocoConverter(EntityConverter, ABC): ...


class MujocoKinematicStructureEntityConverter(
    MujocoConverter, KinematicStructureEntityConverter, ABC
):
    pos_str: str = "pos"
    quat_str: str = "quat"


class MujocoBodyConverter(MujocoKinematicStructureEntityConverter, BodyConverter):
    mass_str: str = "mass"
    inertia_pos_str: str = "ipos"
    inertia_quat_str: str = "iquat"
    diagonal_inertia_str: str = "inertia"

    def _post_convert(
        self, entity: Body, body_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        return body_props


class MujocoRegionConverter(MujocoKinematicStructureEntityConverter, RegionConverter):
    def _post_convert(
        self, entity: Region, region_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        return region_props


class MujocoGeomConverter(MujocoConverter, ShapeConverter, ABC):
    pos_str: str = "pos"
    quat_str: str = "quat"
    rgba_str: str = "rgba"
    type: mujoco.mjtGeom

    def _post_convert(
        self, entity: Shape, shape_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
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


class MujocoBoxConverter(MujocoGeomConverter, BoxConverter):
    type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_BOX

    def _post_convert(
        self, entity: Box, shape_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        shape_props.update(
            MujocoGeomConverter._post_convert(self, entity, shape_props, **kwargs)
        )
        shape_props.update(
            {"size": [entity.scale.x / 2, entity.scale.y / 2, entity.scale.z / 2]}
        )
        return shape_props


class MujocoSphereConverter(MujocoGeomConverter, SphereConverter):
    type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_SPHERE

    def _post_convert(
        self, entity: Sphere, shape_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        shape_props.update(
            MujocoGeomConverter._post_convert(self, entity, shape_props, **kwargs)
        )
        shape_props.update({"size": [entity.radius, entity.radius, entity.radius]})
        return shape_props


class MujocoCylinderConverter(MujocoGeomConverter, CylinderConverter):
    type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_CYLINDER

    def _post_convert(
        self, entity: Cylinder, shape_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        shape_props.update(
            MujocoGeomConverter._post_convert(self, entity, shape_props, **kwargs)
        )
        shape_props.update({"size": [entity.width / 2, entity.height, 0.0]})
        return shape_props


class MujocoMeshConverter(MujocoGeomConverter, MeshConverter):
    type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_MESH

    def _post_convert(
        self, entity: Mesh, shape_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        shape_props.update(
            MujocoGeomConverter._post_convert(self, entity, shape_props, **kwargs)
        )
        shape_props.update({"mesh": entity})
        if isinstance(entity.mesh.visual, TextureVisuals) and isinstance(
            entity.mesh.visual.material.name, str
        ):
            shape_props["texture_file_path"] = (
                entity.mesh.visual.material.image.filename
            )
        return shape_props


class MujocoJointConverter(ConnectionConverter, ABC):
    pos_str: str = "pos"
    quat_str: str = "quat"
    type: mujoco.mjtJoint
    armature_str: str = "armature"
    dry_friction_str: str = "frictionloss"
    damping_str: str = "damping"

    def _post_convert(
        self, entity: Connection, joint_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        joint_props["type"] = self.type
        return joint_props


class Mujoco1DOFJointConverter(MujocoJointConverter, Connection1DOFConverter):
    axis_str: str = "axis"
    range_str: str = "range"

    def _post_convert(
        self, entity: ActiveConnection1DOF, joint_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        joint_props = MujocoJointConverter._post_convert(self, entity, joint_props)
        if not numpy.allclose(joint_props["quat"], [1.0, 0.0, 0.0, 0.0]):
            joint_axis = numpy.array(joint_props["axis"])
            R_joint = Rotation.from_quat(quat=joint_props["quat"], scalar_first=True)  # type: ignore
            joint_props["axis"] = R_joint.apply(joint_axis).tolist()
        del joint_props["quat"]
        return joint_props


class MujocoRevoluteJointConverter(
    Mujoco1DOFJointConverter, ConnectionRevoluteConverter
):
    type: mujoco.mjtJoint = mujoco.mjtJoint.mjJNT_HINGE

    def _post_convert(
        self, entity: ActiveConnection1DOF, joint_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        joint_props = super()._post_convert(entity, joint_props, **kwargs)
        joint_range = joint_props.pop("range")
        if not any(limit is None for limit in joint_range):
            joint_props["range"] = joint_range
        return joint_props


class MujocoPrismaticJointConverter(
    Mujoco1DOFJointConverter, ConnectionPrismaticConverter
):
    type: mujoco.mjtJoint = mujoco.mjtJoint.mjJNT_SLIDE


class Mujoco6DOFJointConverter(MujocoJointConverter, Connection6DOFConverter):
    type: mujoco.mjtJoint = mujoco.mjtJoint.mjJNT_FREE


class MujocoActuatorConverter(ActuatorConverter, ABC):

    entity_type: ClassVar[Type[MujocoActuator]] = MujocoActuator

    def _post_convert(
        self, entity: MujocoActuator, actuator_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        return actuator_props


class MujocoGeneralActuatorConverter(MujocoActuatorConverter, ActuatorConverter):

    def _post_convert(
        self, entity: MujocoActuator, actuator_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        actuator_props["actlimited"] = entity.activation_limited
        actuator_props["actrange"] = entity.activation_range
        actuator_props["ctrllimited"] = entity.ctrl_limited
        actuator_props["ctrlrange"] = entity.ctrl_range
        actuator_props["forcelimited"] = entity.force_limited
        actuator_props["forcerange"] = entity.force_range
        actuator_props["biasprm"] = entity.bias_parameters
        actuator_props["biastype"] = entity.bias_type
        actuator_props["dynprm"] = entity.dynamics_parameters
        actuator_props["dyntype"] = entity.dynamics_type
        actuator_props["gainprm"] = entity.gain_parameters
        actuator_props["gaintype"] = entity.gain_type
        return actuator_props


class MujocoCameraConverter(CameraConverter, ABC):

    entity_type: ClassVar[Type[MujocoCamera]] = MujocoCamera

    def _post_convert(
        self, entity: MujocoCamera, camera_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        camera_props["mode"] = entity.mode
        camera_props["orthographic"] = entity.orthographic
        camera_props["fovy"] = entity.fovy
        camera_props["resolution"] = entity.resolution
        camera_props["focal_length"] = entity.focal_length
        camera_props["focal_pixel"] = entity.focal_pixel
        camera_props["principal_length"] = entity.principal_length
        camera_props["principal_pixel"] = entity.principal_pixel
        camera_props["sensor_size"] = entity.sensor_size
        camera_props["ipd"] = entity.ipd
        camera_props["pos"] = entity.pos
        camera_props["quat"] = entity.quat
        return camera_props


@dataclass
class MultiSimBuilder(ABC):
    """
    A builder to build a world in the Multiverse simulator.
    """

    _world: Optional[World] = None
    """
    The world to be built.
    """

    def build_world(self, world: World, file_path: str):
        """
        Builds the world in the simulator and saves it to a file.

        :param world: The world to be built.
        :param file_path: The file path to save the world to.
        """
        self._world = world
        self._asset_folder_path = os.path.join(os.path.dirname(file_path), "assets")
        if not os.path.exists(self.asset_folder_path):
            os.makedirs(self.asset_folder_path)
        if len(self.world.bodies) == 0:
            with self.world.modify_world():
                root = Body(name=PrefixedName("world"))
                self.world.add_body(root)
        elif self.world.bodies[0].name.name != "world":
            with world.modify_world():
                root_bodies = [
                    body for body in self.world.bodies if body.parent_connection is None
                ]
                root = Body(name=PrefixedName("world"))
                self.world.add_body(root)
                for root_body in root_bodies:
                    connection = FixedConnection(parent=root, child=root_body)
                    self.world.add_connection(connection)

        self._start_build(file_path=file_path)

        for body in world.bodies:
            self.build_body(body=body)

        for region in world.regions:
            self.build_region(region=region)

        for connection in world.connections:
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
        for camera in body.get_semantic_annotations_by_type(MultiSimCamera):
            self._build_camera(camera=camera)

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
        parent: Union[Body, Region],
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
class MujocoBuilder(MultiSimBuilder):
    """
    A builder to build a world in the Mujoco simulator.
    """

    spec: mujoco.MjSpec = field(default=mujoco.MjSpec())

    def _start_build(self, file_path: str):
        self.spec = mujoco.MjSpec()
        self.spec.modelname = "scene"
        self.spec.compiler.degree = 0

    def _end_build(self, file_path: str):
        self._build_equalities()
        self.spec.compile()
        self.spec.to_file(file_path)
        import xml.etree.ElementTree as ET

        tree = ET.parse(file_path)
        root = tree.getroot()
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
        tree.write(file_path, encoding="utf-8", xml_declaration=True)

    def _build_body(self, body: Body):
        self._build_mujoco_body(body=body)

    def _build_region(self, region: Region):
        self._build_mujoco_body(body=region)

    def _build_shape(
        self,
        parent: Union[Body, Region],
        shape: Shape,
        is_visible: bool,
        is_collidable: bool,
    ):
        geom_props = MujocoGeomConverter.convert(
            shape, visible=is_visible, collidable=is_collidable
        )
        parent_body_name = parent.name.name
        parent_body_spec = self._find_entity(
            entity_type=mujoco.mjtObj.mjOBJ_BODY, entity_name=parent_body_name
        )
        if parent_body_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=parent_body_name, entity_type=mujoco.mjtObj.mjOBJ_BODY
            )
        if geom_props["type"] == mujoco.mjtGeom.mjGEOM_MESH and not self._parse_geom(
            geom_props=geom_props
        ):
            logger.warning(
                f"Mesh {shape.mesh} could not be parsed. Skipping geom {geom_props['name']}."
            )
            return
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

    def _parse_geom(self, geom_props: Dict[str, Any]) -> bool:
        """
        Parses the geometry properties for a mesh geom. Adds the mesh to the spec if it doesn't exist.

        :param geom_props: The geometry properties to parse.
        :return: True if the mesh was parsed successfully, False otherwise.
        """
        mesh_entity = geom_props.pop("mesh")
        if isinstance(mesh_entity, TriangleMesh):
            mesh_name = os.path.basename(mesh_entity.file.name)
            mesh_file_path = os.path.join(self.asset_folder_path, f"{mesh_name}.obj")
            shutil.move(mesh_entity.file.name, mesh_file_path)
        elif isinstance(mesh_entity, FileMesh):
            mesh_file_path = mesh_entity.filename
        else:
            raise NotImplementedError(
                f"Mesh type {type(mesh_entity)} not supported in Mujoco."
            )
        mesh_ext = os.path.splitext(mesh_file_path)[1].lower()
        if mesh_ext == ".dae":
            logger.warning(
                f"Cannot use .dae files in Mujoco. Skipping mesh {mesh_file_path}."
            )
            return False
        mesh_name = os.path.splitext(os.path.basename(mesh_file_path))[0]
        mesh_scale = [mesh_entity.scale.x, mesh_entity.scale.y, mesh_entity.scale.z]
        if not numpy.allclose(mesh_scale, [1.0, 1.0, 1.0]):
            mesh_name += f"_{'_'.join(map(str, mesh_scale))}"
        if mesh_name not in [mesh.name for mesh in self.spec.meshes]:
            mesh = self.spec.add_mesh(name=mesh_name)
            mesh.file = mesh_file_path
            mesh.scale = mesh_scale
        geom_props["meshname"] = mesh_name
        texture_file_path = geom_props.pop("texture_file_path", None)
        if isinstance(texture_file_path, str):
            texture_name = os.path.splitext(os.path.basename(texture_file_path))[0]
            if texture_name in [
                self.spec.textures[i].name for i in range(len(self.spec.textures))
            ]:
                return True
            material_name = texture_name
            if material_name.startswith("T_"):
                material_name = material_name[2:]
            material_name = f"M_{material_name}"
            geom_props["material"] = material_name
            if material_name in [
                self.spec.materials[i].name for i in range(len(self.spec.materials))
            ]:
                return True
            if not os.path.exists(texture_file_path):
                return True
            self.spec.add_texture(
                name=texture_name,
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=texture_file_path,
            )
            material = self.spec.add_material(name=material_name)
            material.textures[0] = texture_name
        return True

    def _build_connection(self, connection: Connection):
        if isinstance(connection, FixedConnection):
            return
        joint_props = MujocoJointConverter.convert(connection)
        if "equality_joint" in joint_props:
            equality_joint = joint_props.pop("equality_joint")
            equality = self.spec.add_equality()
            equality.type = mujoco.mjtEq.mjEQ_JOINT
            equality.objtype = mujoco.mjtObj.mjOBJ_JOINT
            equality.name1 = joint_props["name"]
            equality.name2 = equality_joint["joint"]
            equality.data = equality_joint["data"]

        child_body_name = connection.child.name.name
        child_body_spec = self._find_entity(
            entity_type=mujoco.mjtObj.mjOBJ_BODY, entity_name=child_body_name
        )
        if child_body_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=child_body_name,
                entity_type=mujoco.mjtObj.mjOBJ_BODY,
            )
        joint_name = connection.name.name
        joint_spec = child_body_spec.add_joint(**joint_props)
        if joint_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=joint_name,
                entity_type=mujoco.mjtObj.mjOBJ_JOINT,
                action="add",
            )

    def _build_actuator(self, actuator: Actuator):
        actuator_props = MujocoActuatorConverter.convert(actuator)
        dof_names = actuator_props.pop("dof_names")
        assert len(dof_names) == 1, "Actuator must be associated with exactly one DOF."
        dof_name = dof_names[0]
        connection = next(
            (
                conn
                for conn in actuator._world.connections
                if dof_name in [dof.name.name for dof in conn.dofs]
            ),
            None,
        )
        if connection is None:
            raise MultiSimError(
                f"Connection for DOF {dof_name} not found, it need to be added first."
            )
        connection_name = connection.name.name
        joint_spec = self._find_entity(
            entity_type=mujoco.mjtObj.mjOBJ_JOINT, entity_name=connection_name
        )
        if joint_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=connection_name,
                entity_type=mujoco.mjtObj.mjOBJ_JOINT,
            )
        actuator_props["target"] = joint_spec.name
        actuator_props["trntype"] = mujoco.mjtTrn.mjTRN_JOINT
        actuator_name = actuator.name.name
        actuator_spec = self.spec.add_actuator(**actuator_props)
        if actuator_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=actuator_name,
                entity_type=mujoco.mjtObj.mjOBJ_ACTUATOR,
                action="add",
            )

    def _build_camera(self, camera: MultiSimCamera):
        camera_name = camera.name.name
        camera_props = MujocoCameraConverter.convert(camera)
        body_name = camera_props.pop("body")
        body_spec = self._find_entity(
            entity_type=mujoco.mjtObj.mjOBJ_BODY, entity_name=body_name
        )
        if body_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=body_name, entity_type=mujoco.mjtObj.mjOBJ_BODY
            )
        camera_spec = body_spec.add_camera(**camera_props)
        if camera_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=camera_name,
                entity_type=mujoco.mjtObj.mjOBJ_CAMERA,
                action="add",
            )

    def _build_mujoco_body(self, body: Union[Region, Body]):
        """
        Builds a body in the Mujoco spec. In Mujoco, regions are also represented as bodies.

        :param body: The body or region to build.
        """
        if body.name.name == "world":
            return
        body_props = MujocoKinematicStructureEntityConverter.convert(body)
        parent_body_name = body.parent_connection.parent.name.name
        parent_body_spec = self._find_entity(
            entity_type=mujoco.mjtObj.mjOBJ_BODY, entity_name=parent_body_name
        )
        if parent_body_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=parent_body_name,
                entity_type=mujoco.mjtObj.mjOBJ_BODY,
            )
        if any(
            [
                semantic_annotation.body == body
                for semantic_annotation in self.world.get_semantic_annotations_by_type(
                    MujocoMocapBody
                )
            ]
        ):
            body_props["mocap"] = 1
        body_spec = parent_body_spec.add_body(**body_props)
        if body_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=parent_body_name,
                entity_type=mujoco.mjtObj.mjOBJ_BODY,
                action="add",
            )

    def _build_equalities(self):
        """
        Builds all equalities in the Mujoco spec.
        """
        for equality_semantic_annotation in self.world.get_semantic_annotations_by_type(
            MujocoEquality
        ):
            equality = self.spec.add_equality()
            equality.type = equality_semantic_annotation.type
            equality.objtype = equality_semantic_annotation.obj_type
            equality.name1 = equality_semantic_annotation.name_1
            equality.name2 = equality_semantic_annotation.name_2
            equality.data = equality_semantic_annotation.data

    def _find_entity(
        self,
        entity_type: mujoco.mjtObj,
        entity_name: str,
    ) -> Optional[
        Union[mujoco.MjsBody, mujoco.MjsGeom, mujoco.MjsJoint, mujoco.MjsSite]
    ]:
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


class EntitySpawner(ABC):
    """
    A spawner to spawn a WorldEntity object in the Multiverse simulator.
    """

    entity_type: ClassVar[Type[Any]] = Any
    """
    The type of the entity to spawn.
    """

    @classmethod
    def spawn(cls, simulator: MultiverseSimulator, entity: entity_type) -> bool:  # type: ignore
        """
        Spawns a WorldEntity object in the Multiverse simulator.

        :param simulator: The Multiverse simulator to spawn the entity in.
        :param entity: The WorldEntity object to spawn.

        :return: True if the entity was spawned successfully, False otherwise.
        """
        for subclass in recursive_subclasses(cls):
            if (
                not inspect.isabstract(subclass)
                and not inspect.isabstract(subclass.entity_type)
                and type(entity) is subclass.entity_type
            ):
                return subclass()._spawn(simulator, entity)
        raise NotImplementedError(f"No converter found for entity type {type(entity)}.")

    @abstractmethod
    def _spawn(self, simulator: MultiverseSimulator, entity: Any) -> bool:
        """
        The actual spawning method to be implemented by subclasses.

        :param simulator: The Multiverse simulator to spawn the entity in.
        :param entity: The WorldEntity object to spawn.
        :return: True if the entity was spawned successfully, False otherwise.
        """
        raise NotImplementedError


class KinematicStructureEntitySpawner(EntitySpawner):
    """
    A spawner to spawn a KinematicStructureEntity object in the Multiverse simulator.
    """

    entity_type: ClassVar[Type[KinematicStructureEntity]] = KinematicStructureEntity
    """
    The type of the entity to spawn.
    """

    def _spawn(
        self, simulator: MultiverseSimulator, entity: KinematicStructureEntity
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
        self, simulator: MultiverseSimulator, entity: KinematicStructureEntity
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
        self, simulator: MultiverseSimulator, entity: KinematicStructureEntity
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
        parent: Union[Body, Region],
        simulator: MultiverseSimulator,
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


class BodySpawner(KinematicStructureEntitySpawner, ABC):
    """
    A spawner to spawn a Body object in the Multiverse simulator.
    """

    entity_type: ClassVar[Type[Body]] = Body
    """
    The type of the entity to spawn.
    """

    def _spawn_shapes(self, simulator: MultiverseSimulator, parent: Body) -> bool:
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


class RegionSpawner(KinematicStructureEntitySpawner, ABC):
    """
    A spawner to spawn a Region object in the Multiverse simulator.
    """

    entity_type: ClassVar[Type[Region]] = Region
    """
    The type of the entity to spawn.
    """

    def _spawn_shapes(self, simulator: MultiverseSimulator, parent: Region) -> bool:
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


class ActuatorSpawner(EntitySpawner):
    """
    A spawner to spawn an Actuator object in the Multiverse simulator.
    """

    entity_type: ClassVar[Type[Actuator]] = Actuator
    """
    The type of the entity to spawn.
    """

    def _spawn(self, simulator: MultiverseSimulator, entity: Actuator) -> bool:
        """
        Spawns a Actuator object in the Multiverse simulator including its dofs.

        :param simulator: The Multiverse simulator to spawn the entity in.
        :param entity: The Actuator object to spawn.

        :return: True if the entity is spawned successfully, False otherwise.
        """
        return self._spawn_actuator(simulator, entity)

    @abstractmethod
    def _spawn_actuator(
        self, simulator: MultiverseSimulator, actuator: Actuator
    ) -> bool:
        raise NotImplementedError


class MujocoEntitySpawner(EntitySpawner, ABC):
    """
    A spawner to spawn a WorldEntity object in the Mujoco simulator.
    """

    ...


class MujocoKinematicStructureEntitySpawner(
    MujocoEntitySpawner, KinematicStructureEntitySpawner, ABC
):
    """
    A spawner to spawn a KinematicStructureEntity object in the Mujoco simulator.
    """

    def _spawn_kinematic_structure_entity(
        self, simulator: MultiverseMujocoConnector, entity: KinematicStructureEntity
    ) -> bool:
        kinematic_structure_entity_props = (
            MujocoKinematicStructureEntityConverter.convert(entity)
        )
        entity_name = kinematic_structure_entity_props["name"]
        del kinematic_structure_entity_props["name"]
        result = simulator.add_entity(
            entity_name=entity_name,
            entity_type="body",
            entity_properties=kinematic_structure_entity_props,
            parent_name=entity.parent_connection.parent.name.name,
        )
        return (
            result.type
            == MultiverseCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        )

    def _spawn_shape(
        self,
        parent: Body,
        simulator: MultiverseMujocoConnector,
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
            entity_type="geom",
            entity_properties=shape_props,
            parent_name=parent.name.name,
        )
        return (
            result.type
            == MultiverseCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        )


class MujocoBodySpawner(MujocoKinematicStructureEntitySpawner, BodySpawner):
    """
    A spawner to spawn a Body object in the Mujoco simulator.
    """

    ...


class MujocoRegionSpawner(MujocoKinematicStructureEntitySpawner, RegionSpawner):
    """
    A spawner to spawn a Region object in the Mujoco simulator.
    """

    ...


class MujocoActuatorSpawner(MujocoEntitySpawner, ActuatorSpawner):
    """
    A spawner to spawn a MujocoActuator object in the MuJoCo simulator.
    """

    entity_type: ClassVar[Type[MujocoActuator]] = MujocoActuator

    def _spawn_actuator(
        self, simulator: MultiverseMujocoConnector, actuator: MujocoActuator
    ) -> bool:
        actuator_props = MujocoActuatorConverter.convert(actuator)
        actuator_name = actuator_props.pop("name")
        dof_names = actuator_props.pop("dof_names")
        assert len(dof_names) == 1, "Actuator must be associated with exactly one DOF."
        dof_name = dof_names[0]
        connection = next(
            (
                conn
                for conn in actuator._world.connections
                if dof_name in [dof.name.name for dof in conn.dofs]
            ),
            None,
        )
        assert connection is not None, f"Connection for DOF {dof_name} not found."
        connection_name = connection.name.name
        joint_spec = simulator.get_joint(joint_name=connection_name).result
        if joint_spec is None:
            raise MujocoEntityNotFoundError(
                entity_name=connection_name,
                entity_type=mujoco.mjtObj.mjOBJ_JOINT,
            )
        actuator_props["target"] = joint_spec.name
        actuator_props["trntype"] = mujoco.mjtTrn.mjTRN_JOINT
        result = simulator.add_entity(
            entity_name=actuator_name,
            entity_type="actuator",
            entity_properties=actuator_props,
        )
        return (
            result.type
            == MultiverseCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        )


@dataclass
class MultiSimSynchronizer(ModelChangeCallback, ABC):
    """
    A callback to synchronize the world model with the Multiverse simulator.
    This callback will listen to the world model changes and update the Multiverse simulator accordingly.
    """

    world: World
    """
    The world to synchronize with the simulator.
    """

    simulator: MultiverseSimulator
    """
    The Multiverse simulator to synchronize with the world.
    """

    entity_converter: Type[EntityConverter] = NoneType
    """
    The converter to convert WorldEntity, Shape, and Connection objects to dictionaries of properties for the simulator.
    """

    entity_spawner: Type[EntitySpawner] = NoneType
    """
    The spawner to spawn WorldEntity, Shape, and Connection objects in the simulator.
    """

    def _notify(self):
        for modification in self.world._model_manager.model_modification_blocks[-1]:
            if isinstance(modification, AddKinematicStructureEntityModification):
                entity = modification.kinematic_structure_entity
                self.entity_spawner.spawn(simulator=self.simulator, entity=entity)
            elif isinstance(modification, AddActuatorModification):
                entity = modification.actuator
                self.entity_spawner.spawn(simulator=self.simulator, entity=entity)

    def stop(self):
        self.world._model_manager.model_change_callbacks.remove(self)


@dataclass
class MujocoSynchronizer(MultiSimSynchronizer):
    simulator: MultiverseMujocoConnector
    entity_converter: Type[EntityConverter] = field(default=MujocoConverter)
    entity_spawner: Type[EntitySpawner] = field(default=MujocoEntitySpawner)


class MultiSim(ABC):
    """
    Class to handle the simulation of a world using the Multiverse simulator.
    """

    simulator_class: ClassVar[Type[MultiverseSimulator]]
    """
    The class of the Multiverse simulator to use.
    """

    synchronizer_class: ClassVar[Type[MultiSimSynchronizer]]
    """
    The class of the MultiSimSynchronizer to use.
    """

    builder_class: ClassVar[Type[MultiSimBuilder]]
    """
    The class of the MultiSimBuilder to use.
    """

    simulator: MultiverseSimulator
    """
    The Multiverse simulator instance.
    """

    synchronizer: MultiSimSynchronizer
    """
    The MultiSimSynchronizer instance.
    """

    default_file_path: str
    """
    The default file path to save the world to.
    """

    def __init__(
        self,
        world: World,
        viewer: MultiverseViewer,
        headless: bool = False,
        step_size: float = 1e-3,
        real_time_factor: float = 1.0,
        **kwargs,
    ):
        """
        Initializes the MultiSim class.

        :param world: The world to simulate.
        :param viewer: The MultiverseViewer to read/write objects.
        :param headless: Whether to run the simulation in headless mode.
        :param step_size: The step size for the simulation.
        :param real_time_factor: The real time factor for the simulation (1.0 = real time, 2.0 = twice as fast, -1.0 = as fast as possible).
        """
        self.builder_class().build_world(world=world, file_path=self.default_file_path)
        self.simulator = self.simulator_class(
            file_path=self.default_file_path,
            viewer=viewer,
            headless=headless,
            step_size=step_size,
            real_time_factor=real_time_factor,
            **kwargs,
        )
        self.synchronizer = self.synchronizer_class(
            world=world,
            simulator=self.simulator,
        )
        self._viewer = viewer

    def start_simulation(self):
        """
        Starts the simulation. This will start one physics simulation thread and render it at 60Hz.
        """
        assert (
            self.simulator.state != MultiverseSimulatorState.RUNNING
        ), "Simulation is already running."
        self.simulator.start()

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
        if self.simulator.state != MultiverseSimulatorState.PAUSED:
            self.simulator.pause()

    def unpause_simulation(self):
        """
        Unpauses the simulation. This will unpause the physics simulation.
        """
        if self.simulator.state == MultiverseSimulatorState.PAUSED:
            self.simulator.unpause()

    def reset_simulation(self):
        """
        Resets the simulation. This will reset the physics simulation to the initial state.
        """
        self.simulator.reset()

    def set_write_objects(self, write_objects: Dict[str, Dict[str, List[float]]]):
        """
        Sets the objects to be written to the simulator.
        For example, to set the position and quaternion of an object, you can use the following format:
        {
            "object_name": {
                "position": [x, y, z],
                "quaternion": [w, x, y, z]
            }
        }

        :param write_objects: The objects to be written to the simulator.
        """
        self._viewer.write_objects = write_objects
        if self.simulator.state == MultiverseSimulatorState.PAUSED:
            self.simulator.step()

    def set_read_objects(self, read_objects: Dict[str, Dict[str, List[float]]]):
        """
        Sets the objects to be read from the simulator.

        For example, to read the position and quaternion of an object, you can use the following format:
        {
            "object_name": {
                "position": [0.0, 0.0, 0.0], # Default value
                "quaternion": [1.0, 0.0, 0.0], # Default value
            }
        }
        :param read_objects: The objects to be read from the simulator.
        """
        self._viewer.read_objects = read_objects
        if self.simulator.state == MultiverseSimulatorState.PAUSED:
            self.simulator.step()

    def get_read_objects(self) -> Dict[str, Dict[str, MultiverseAttribute]]:
        """
        Gets the objects that are being read from the simulator.
        For example, if you have set the read objects as follows:
        {
            "object_name": {
                "position": [0.0, 0.0, 0.0],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
            }
        }
        You will get the following format:
        {
            "object_name": {
                "position": MultiverseAttribute(...),
                "quaternion": MultiverseAttribute(...),
            }
        }
        where MultiverseAttribute contains the values of the attribute via the .values() method.
        It will return the values that are being read from the simulator in every simulation step.

        :return: The objects that are being read from the simulator.
        """
        if self.simulator.state == MultiverseSimulatorState.PAUSED:
            self.simulator.step()
        return self._viewer.read_objects

    def is_stable(
        self, body_names: List[str], max_simulation_steps: int = 100, atol: float = 1e-2
    ) -> bool:
        """
        Checks if an object is stable in the world. Stable meaning that it's pose will not change after simulating
        physics in the World. This function will pause the simulation, set the read objects to the given body names,
        unpause the simulation, and check if the pose of the objects change after a certain number of simulation steps.
        If the pose of the objects change, the function will return False. If the pose of the objects do not change,
        the function will return True. After checking, the function will restore the read objects and the simulation state.

        :param body_names: The names of the bodies to check for stability
        :param max_simulation_steps: The maximum number of simulation steps to run
        :param atol: The absolute tolerance for comparing the pose
        :return: True if the object is stable, False otherwise
        """

        origin_read_objects = self.get_read_objects()
        origin_state = self.simulator.state

        self.pause_simulation()
        self.set_read_objects(
            read_objects={
                body_name: {
                    "position": [0.0, 0.0, 0.0],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                }
                for body_name in body_names
            }
        )
        initial_body_state = numpy.array(self._viewer.read_data)
        current_simulation_step = self.simulator.current_number_of_steps
        self.unpause_simulation()
        stable = True
        while (
            self.simulator.current_number_of_steps
            < current_simulation_step + max_simulation_steps
        ):
            if numpy.abs(initial_body_state - self._viewer.read_data).max() > atol:
                stable = False
                break
            time.sleep(1e-3)
        self._viewer.read_objects = origin_read_objects
        if origin_state == MultiverseSimulatorState.PAUSED:
            self.pause_simulation()
        return stable


class MujocoSim(MultiSim):
    simulator_class: ClassVar[Type[MultiverseSimulator]] = MultiverseMujocoConnector
    synchronizer_class: ClassVar[Type[MultiSimSynchronizer]] = MujocoSynchronizer
    builder_class: ClassVar[Type[MultiSimBuilder]] = MujocoBuilder
    simulator: MultiverseMujocoConnector
    synchronizer: Type[MultiSimSynchronizer] = MujocoSynchronizer
    default_file_path: str = "/tmp/scene.xml"

import logging
import os
from typing_extensions import Optional, Dict
import numpy
from dataclasses import dataclass, field

import mujoco
from scipy.spatial.transform import Rotation

from ..datastructures.prefixed_name import PrefixedName
from ..exceptions import WorldEntityNotFoundError
from ..spatial_types import (
    HomogeneousTransformationMatrix,
    RotationMatrix,
    Point3,
    Vector3,
)
from ..spatial_types.derivatives import DerivativeMap
from ..world import World, Body
from ..world_description.connection_properties import JointDynamics
from ..world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    FixedConnection,
    Connection6DoF,
)
from ..world_description.degree_of_freedom import DegreeOfFreedom
from ..world_description.geometry import (
    Box,
    Sphere,
    Cylinder,
    Scale,
    Shape,
    Color,
    FileMesh,
)
from ..world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
    PrincipalMoments,
    PrincipalAxes,
)
from ..world_description.shape_collection import ShapeCollection
from .multi_sim import (
    MujocoActuator,
    GeomVisibilityAndCollisionType,
    MujocoCamera,
    MujocoEquality,
    MujocoMocapBody,
)

logger = logging.getLogger(__name__)


@dataclass
class MJCFParser:
    """
    Class to parse an MJCF file and convert it into a World object.
    """

    file_path: str
    """
    The file path of the scene.
    """

    mimic_joints: Dict[str, str] = field(default_factory=dict)
    """
    A dictionary mapping joint names to the names of the joints they mimic.
    """

    prefix: Optional[str] = None
    """
    The prefix for every name used in this world.
    """

    def __post_init__(self):
        if self.prefix is None:
            self.prefix = os.path.basename(self.file_path).split(".")[0]
        self.spec: mujoco.MjSpec = mujoco.MjSpec.from_file(self.file_path)
        self.world = World()

    def parse(self) -> World:
        """
        Parse the MJCF file and convert it into a World object.

        :return: The World object representing the MJCF scene.
        """

        worldbody: mujoco.MjsBody = self.spec.worldbody
        with self.world.modify_world():
            self.parse_equalities()

            root = Body(name=PrefixedName(worldbody.name))
            self.world.add_body(root)

            for mujoco_body in worldbody.bodies:
                self.parse_body(mujoco_body=mujoco_body)

            for mujoco_body in self.spec.bodies[1:]:
                self.parse_joints(mujoco_body=mujoco_body)

            for mujoco_camera in self.spec.cameras:
                self.parse_camera(mujoco_camera=mujoco_camera)

            for mujoco_actuator in self.spec.actuators:
                self.parse_actuator(mujoco_actuator=mujoco_actuator)

        return self.world

    def parse_body(self, mujoco_body: mujoco.MjsBody):
        """
        Parse a Mujoco body and add it to the world.

        :param mujoco_body: The Mujoco body to parse.
        """
        body = Body(name=PrefixedName(mujoco_body.name))
        visuals = []
        collisions = []
        for mujoco_geom in mujoco_body.geoms:
            shape = self.parse_geom(mujoco_geom=mujoco_geom)
            if mujoco_geom.contype != 0 or mujoco_geom.conaffinity != 0:
                collisions.append(shape)
            if mujoco_geom.group in [
                GeomVisibilityAndCollisionType.VISIBLE_AND_COLLIDABLE_1,
                GeomVisibilityAndCollisionType.VISIBLE_AND_COLLIDABLE_2,
                GeomVisibilityAndCollisionType.ONLY_VISIBLE,
            ]:
                visuals.append(shape)
        body.inertial = self.parse_inertial(mujoco_body=mujoco_body)
        body.visual = ShapeCollection(shapes=visuals, reference_frame=body)
        body.collision = ShapeCollection(shapes=collisions, reference_frame=body)
        self.world.add_kinematic_structure_entity(body)
        if mujoco_body.mocap:
            self.world.add_semantic_annotation(MujocoMocapBody(body=body))
        for mujoco_child_body in mujoco_body.bodies:
            self.parse_body(mujoco_body=mujoco_child_body)

    def parse_inertial(self, mujoco_body: mujoco.MjsBody) -> Optional[Inertial]:
        """
        Parse the inertial properties of a Mujoco body.

        :param mujoco_body: The Mujoco body to parse.
        :return: The Inertial properties of the body, or None if not applicable.
        """
        if numpy.isclose(mujoco_body.mass, 0.0) or numpy.isnan(mujoco_body.ipos).any():
            return None

        full_inertia = mujoco_body.fullinertia
        if not numpy.isnan(full_inertia).any():
            inertia_tensor = InertiaTensor.from_values(
                ixx=full_inertia[0],
                iyy=full_inertia[1],
                izz=full_inertia[2],
                ixy=full_inertia[3],
                ixz=full_inertia[4],
                iyz=full_inertia[5],
            )
        else:
            principal_moments = PrincipalMoments.from_values(
                i1=mujoco_body.inertia[0],
                i2=mujoco_body.inertia[1],
                i3=mujoco_body.inertia[2],
            )
            principal_axes = PrincipalAxes.from_rotation_matrix(
                RotationMatrix(
                    data=Rotation.from_quat(
                        mujoco_body.iquat, scalar_first=True
                    ).as_matrix()
                )
            )
            inertia_tensor = InertiaTensor.from_principal_moments_and_axes(
                moments=principal_moments,
                axes=principal_axes,
            )

        return Inertial(
            mass=mujoco_body.mass,
            center_of_mass=Point3.from_iterable(mujoco_body.ipos),
            inertia=inertia_tensor,
        )

    def parse_joints(self, mujoco_body: mujoco.MjsBody):
        """
        Parse the joints of a Mujoco body and add them to the world.

        :param mujoco_body: The Mujoco body whose joints to parse.
        """
        for mujoco_joint in mujoco_body.joints:
            self.parse_joint(
                parent_name=mujoco_body.parent.name,
                child_name=mujoco_body.name,
                mujoco_joint=mujoco_joint,
            )
        if len(mujoco_body.joints) == 0:
            body_pos = mujoco_body.pos
            body_quat = mujoco_body.quat
            body_quat /= numpy.linalg.norm(body_quat)
            parent_body_to_child_body_transform = (
                HomogeneousTransformationMatrix.from_xyz_quaternion(
                    pos_x=body_pos[0],
                    pos_y=body_pos[1],
                    pos_z=body_pos[2],
                    quat_w=body_quat[0],
                    quat_x=body_quat[1],
                    quat_y=body_quat[2],
                    quat_z=body_quat[3],
                )
            )
            parent_body = self.world.get_kinematic_structure_entity_by_name(
                mujoco_body.parent.name
            )
            child_body = self.world.get_kinematic_structure_entity_by_name(
                mujoco_body.name
            )
            self.world.add_connection(
                FixedConnection(
                    parent=parent_body,
                    child=child_body,
                    parent_T_connection_expression=parent_body_to_child_body_transform,
                )
            )

    def parse_geom(self, mujoco_geom: mujoco.MjsGeom) -> Shape:
        """
        Parse a Mujoco geometry and convert it into a Shape object.

        :param mujoco_geom: The Mujoco geometry to parse.
        :return: The Shape object representing the geometry.
        """
        geom_pos = mujoco_geom.pos
        geom_quat = mujoco_geom.quat
        geom_quat /= numpy.linalg.norm(geom_quat)
        origin_transform = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=geom_pos[0],
            pos_y=geom_pos[1],
            pos_z=geom_pos[2],
            quat_w=geom_quat[0],
            quat_x=geom_quat[1],
            quat_y=geom_quat[2],
            quat_z=geom_quat[3],
        )
        size = mujoco_geom.size * 2
        for i in range(len(size)):
            if numpy.isclose(size[i], 0.0):
                size[i] = 100.0  # Handle infinite size
        color = Color(*mujoco_geom.rgba)
        match mujoco_geom.type:
            case mujoco.mjtGeom.mjGEOM_PLANE:
                return Box(
                    origin=origin_transform,
                    scale=Scale(*size[:2], 0.0),
                    color=color,
                )
            case mujoco.mjtGeom.mjGEOM_BOX:
                return Box(
                    origin=origin_transform,
                    scale=Scale(*size),
                    color=color,
                )
            case mujoco.mjtGeom.mjGEOM_SPHERE:
                return Sphere(
                    origin=origin_transform,
                    radius=size[0] / 2,
                    color=color,
                )
            case mujoco.mjtGeom.mjGEOM_CYLINDER:
                return Cylinder(
                    origin=origin_transform,
                    width=size[0],
                    height=size[1] / 2,
                    color=color,
                )
            case mujoco.mjtGeom.mjGEOM_MESH:
                mujoco_mesh: mujoco.MjsMesh = self.spec.mesh(mujoco_geom.meshname)
                meshdir = os.path.join(
                    os.path.dirname(self.file_path), self.spec.meshdir
                )
                filename = str(os.path.join(meshdir, mujoco_mesh.file))
                mujoco_material: mujoco.MjsMaterial = self.spec.material(
                    mujoco_geom.material
                )
                meshscale = Scale(*mujoco_mesh.scale)
                if mujoco_material is None:
                    return FileMesh(
                        filename=filename,
                        origin=origin_transform,
                        color=color,
                        scale=meshscale,
                    )
                else:
                    texture_name = mujoco_material.textures[1]
                    mujoco_texture: mujoco.MjsTexture = self.spec.texture(texture_name)
                    if mujoco_texture is None:
                        color = Color(*mujoco_material.rgba)
                        return FileMesh(
                            filename=filename,
                            origin=origin_transform,
                            color=color,
                            scale=meshscale,
                        )
                    texturedir = os.path.join(
                        os.path.dirname(self.file_path), self.spec.texturedir
                    )
                    texture_file_path = os.path.join(texturedir, mujoco_texture.file)
                    if os.path.isfile(texture_file_path):
                        return FileMesh.from_file(
                            file_path=filename,
                            origin=origin_transform,
                            color=color,
                            texture_file_path=texture_file_path,
                            scale=meshscale,
                        )
                    else:
                        return FileMesh(
                            filename=filename,
                            origin=origin_transform,
                            color=color,
                            scale=meshscale,
                        )

        raise NotImplementedError(f"Geometry type {mujoco_geom.type} not implemented.")

    def parse_joint(
        self,
        parent_name: str,
        child_name: str,
        mujoco_joint: Optional[mujoco.MjsJoint] = None,
    ):
        """
        Parse a Mujoco joint and add it to the world.

        :param parent_name: The name of the parent body.
        :param child_name: The name of the child body.
        :param mujoco_joint: The Mujoco joint to parse. If None, a fixed connection is created.
        """
        parent_body = self.world.get_kinematic_structure_entity_by_name(parent_name)
        child_body = self.world.get_kinematic_structure_entity_by_name(child_name)
        if mujoco_joint is None:
            connection = FixedConnection(parent=parent_body, child=child_body)
        else:
            mujoco_child_body = self.spec.body(child_name)
            child_body_pos = mujoco_child_body.pos
            child_body_quat = mujoco_child_body.quat
            child_body_quat /= numpy.linalg.norm(child_body_quat)
            parent_body_to_child_body_transform = (
                HomogeneousTransformationMatrix.from_xyz_quaternion(
                    pos_x=child_body_pos[0],
                    pos_y=child_body_pos[1],
                    pos_z=child_body_pos[2],
                    quat_w=child_body_quat[0],
                    quat_x=child_body_quat[1],
                    quat_y=child_body_quat[2],
                    quat_z=child_body_quat[3],
                )
            )
            child_body_to_joint_transform = (
                HomogeneousTransformationMatrix.from_xyz_quaternion(
                    pos_x=mujoco_joint.pos[0],
                    pos_y=mujoco_joint.pos[1],
                    pos_z=mujoco_joint.pos[2],
                )
            )
            parent_body_to_joint_transform = (
                parent_body_to_child_body_transform @ child_body_to_joint_transform
            )
            if mujoco_joint.type == mujoco.mjtJoint.mjJNT_FREE:
                connection = Connection6DoF.create_with_dofs(
                    world=self.world,
                    parent=parent_body,
                    child=child_body,
                    parent_T_connection_expression=parent_body_to_joint_transform,
                )
            else:
                joint_to_child_body_transform = child_body_to_joint_transform.inverse()
                joint_axis = Vector3(
                    mujoco_joint.axis[0],
                    mujoco_joint.axis[1],
                    mujoco_joint.axis[2],
                    reference_frame=parent_body,
                )
                dof = self.parse_dof(mujoco_joint=mujoco_joint)
                joint_dynamics = JointDynamics(
                    armature=mujoco_joint.armature,
                    dry_friction=mujoco_joint.frictionloss,
                    damping=mujoco_joint.damping,
                )
                if mujoco_joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                    connection = RevoluteConnection(
                        name=PrefixedName(mujoco_joint.name),
                        parent=parent_body,
                        child=child_body,
                        parent_T_connection_expression=parent_body_to_joint_transform,
                        connection_T_child_expression=joint_to_child_body_transform,
                        axis=joint_axis,
                        dof_id=dof.id,
                        dynamics=joint_dynamics,
                    )
                elif mujoco_joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                    connection = PrismaticConnection(
                        name=PrefixedName(mujoco_joint.name),
                        parent=parent_body,
                        child=child_body,
                        parent_T_connection_expression=parent_body_to_joint_transform,
                        connection_T_child_expression=joint_to_child_body_transform,
                        axis=joint_axis,
                        dof_id=dof.id,
                        dynamics=joint_dynamics,
                    )
                else:
                    raise NotImplementedError(
                        f"Joint type {mujoco_joint.type} not implemented yet."
                    )
        self.world.add_connection(connection)

    def parse_dof(self, mujoco_joint: mujoco.MjsJoint) -> DegreeOfFreedom:
        """
        Parse a Mujoco joint and return the corresponding DegreeOfFreedom.

        :param mujoco_joint: The Mujoco joint to parse.
        :return: The DegreeOfFreedom corresponding to the joint.
        """
        dof_name = mujoco_joint.name
        try:
            return self.world.get_degree_of_freedom_by_name(dof_name)
        except WorldEntityNotFoundError:
            if dof_name in self.mimic_joints:
                dof_name = self.mimic_joints[dof_name]
            if (
                mujoco_joint.range is None
                or mujoco_joint.range[0] == 0
                and mujoco_joint.range[1] == 0
            ):
                dof = DegreeOfFreedom(
                    name=PrefixedName(dof_name),
                )
            else:
                lower_limits = DerivativeMap()
                lower_limits.position = mujoco_joint.range[0]
                upper_limits = DerivativeMap()
                upper_limits.position = mujoco_joint.range[1]
                dof = DegreeOfFreedom(
                    name=PrefixedName(dof_name),
                    lower_limits=lower_limits,
                    upper_limits=upper_limits,
                )
            self.world.add_degree_of_freedom(dof)
            return dof

    def parse_actuator(self, mujoco_actuator: mujoco.MjsActuator):
        """
        Parse a Mujoco actuator and add it to the world.

        :param mujoco_actuator: The Mujoco actuator to parse.
        """
        actuator_name = mujoco_actuator.name
        if mujoco_actuator.trntype != mujoco.mjtTrn.mjTRN_JOINT:
            print(
                f"Warning: Actuator {actuator_name} has trntype {mujoco_actuator.trntype}, which is not supported. Skipping actuator."
            )
            return
        joint_name = mujoco_actuator.target
        connection = self.world.get_connection_by_name(joint_name)
        dofs = list(connection.dofs)
        assert (
            len(dofs) == 1
        ), f"Actuator {actuator_name} is associated with joint {joint_name} which has {len(connection.dofs)} DOFs, but only single-DOF joints are supported for actuators."
        dof = dofs[0]
        actuator = MujocoActuator(
            name=PrefixedName(actuator_name),
            activation_limited=mujoco_actuator.actlimited,
            activation_range=[*mujoco_actuator.actrange],
            ctrl_limited=mujoco_actuator.ctrllimited,
            ctrl_range=[*mujoco_actuator.ctrlrange],
            force_limited=mujoco_actuator.forcelimited,
            force_range=[*mujoco_actuator.forcerange],
            bias_parameters=[*mujoco_actuator.biasprm],
            bias_type=mujoco_actuator.biastype,
            dynamics_parameters=[*mujoco_actuator.dynprm],
            dynamics_type=mujoco_actuator.dyntype,
            gain_parameters=[*mujoco_actuator.gainprm],
            gain_type=mujoco_actuator.gaintype,
        )
        actuator.add_dof(dof)
        self.world.add_actuator(actuator)

    def parse_camera(self, mujoco_camera: mujoco.MjsCamera):
        camera_name = PrefixedName(mujoco_camera.name)
        body_name = mujoco_camera.parent.name
        body = self.world.get_body_by_name(body_name)
        resolution = (
            [1, 1]
            if numpy.isnan(mujoco_camera.resolution).any()
            else mujoco_camera.resolution.tolist()
        )
        focal_length = (
            [0, 0]
            if numpy.isnan(mujoco_camera.focal_length).any()
            else mujoco_camera.focal_length.tolist()
        )
        focal_pixel = (
            [0, 0]
            if numpy.isnan(mujoco_camera.focal_pixel).any()
            else mujoco_camera.focal_pixel.astype(int).tolist()
        )
        principal_length = (
            [0, 0]
            if numpy.isnan(mujoco_camera.principal_length).any()
            else mujoco_camera.principal_length.tolist()
        )
        principal_pixel = (
            [0, 0]
            if numpy.isnan(mujoco_camera.principal_pixel).any()
            else mujoco_camera.principal_pixel.astype(int).tolist()
        )
        sensor_size = (
            [0, 0]
            if numpy.isnan(mujoco_camera.sensor_size).any()
            else mujoco_camera.sensor_size.tolist()
        )
        pos = (
            [0, 0, 0]
            if numpy.isnan(mujoco_camera.pos).any()
            else mujoco_camera.pos.tolist()
        )
        quat = (
            [1, 0, 0, 0]
            if numpy.isnan(mujoco_camera.quat).any()
            else mujoco_camera.quat.tolist()
        )

        camera = MujocoCamera(
            name=camera_name,
            body=body,
            mode=mujoco_camera.mode,
            orthographic=mujoco_camera.orthographic,
            fovy=mujoco_camera.fovy,
            resolution=resolution,
            focal_length=focal_length,
            focal_pixel=focal_pixel,
            principal_length=principal_length,
            principal_pixel=principal_pixel,
            sensor_size=sensor_size,
            ipd=mujoco_camera.ipd,
            pos=pos,
            quat=quat,
        )
        self.world.add_semantic_annotation(camera)

    def parse_equalities(self):
        self.mimic_joints = {}
        equality: mujoco.MjsEquality
        for equality in self.spec.equalities:
            match equality.type:
                case mujoco.mjtEq.mjEQ_JOINT:
                    self.mimic_joints[equality.name2] = equality.name1
                case mujoco.mjtEq.mjEQ_WELD:
                    self.world.add_semantic_annotation(
                        MujocoEquality(
                            type=mujoco.mjtEq.mjEQ_WELD,
                            obj_type=mujoco.mjtObj.mjOBJ_BODY,
                            name_1=equality.name1,
                            name_2=equality.name2,
                            data=equality.data,
                        )
                    )
                case _:
                    logger.warning(
                        f"Equality of type {equality.type} not supported yet. Skipping."
                    )

"""
This module reads geometric ground truth that only exists as RoboCasa's own runtime
state -- a placement sampler's resolved rectangle, a success condition's gripper
exclusion zone -- and attaches it to a :class:`~semantic_digital_twin.world.World` as a
:class:`~semantic_digital_twin.world_description.world_entity.Region` carrying a
:class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.PlacementArea`
or
:class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.GripperExclusionZone`
annotation, so both the shape and what it means to the task live in the same model as
everything else the world already represents, instead of a parallel, adapter specific
format.

Unlike :class:`~semantic_digital_twin.adapters.robocasa_dataset.loader.RoboCasaDatasetLoader`,
this module does not know which tasks or bodies are of interest: it is handed an already
resolved value (a sampler, or a body name and a radius) and turns it into an annotated
Region. Deciding which values are worth reading is left to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, List, Optional

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    GripperExclusionZone,
    PlacementArea,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale, Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Region


@dataclass(frozen=True)
class PlacementSamplerRegion:
    """
    The rectangle a RoboCasa placement sampler resolves for one object, read from its
    sampler parameters rather than reconstructed from the task's source.
    """

    name: PrefixedName
    """
    Name the attached Region is given.
    """

    body_name: str
    """
    Name of the body, already present in the world, this region was resolved for.
    """

    local_center: Point3
    """
    Centre of the region in the sampler's own frame, before the reference rotation and
    translation are applied.
    """

    world_T_sampler: HomogeneousTransformationMatrix
    """
    Pose of the sampler's frame in world coordinates.
    """

    width: float
    """
    Extent of the region along the sampler's own x axis.
    """

    depth: float
    """
    Extent of the region along the sampler's own y axis.
    """

    thickness: float = 0.01
    """
    Vertical extent given to the region's box.

    A placement region is a horizontal rectangle rather than a volume, so it has no
    thickness of its own. A box needs one to have a non zero volume, and this is thin
    enough relative to the surveyed regions to leave the represented area effectively
    unchanged.
    """

    def attach_to(self, world: World) -> PlacementArea:
        """
        :param world: The world to attach the region to, connected to its root at this
            region's resolved world pose.
        :return: The attached region's semantic annotation, recording which object it
            was resolved for.
        """
        box = Box(
            origin=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=self.local_center
            ),
            scale=Scale(self.width, self.depth, self.thickness),
        )
        region = Region.from_shape_collection(
            name=self.name, shape_collection=ShapeCollection(shapes=[box])
        )
        connection = FixedConnection(
            parent=world.root,
            child=region,
            parent_T_connection_expression=HomogeneousTransformationMatrix(
                self.world_T_sampler.to_np()
            ),
        )
        annotation = PlacementArea(
            root=region, placed_object=world.get_body_by_name(self.body_name)
        )
        with world.modify_world():
            world.add_connection(connection)
            world.add_semantic_annotation(annotation)
        return annotation


@dataclass
class PlacementSamplerRegionReader:
    """
    Reads the placement regions a reset RoboCasa environment's samplers resolved.
    """

    sampler_suffix: str = "_Sampler"
    """
    Suffix RoboCasa appends to an object's name when it names that object's sampler.
    """

    def read_all(
        self, environment: Any, name_prefix: str
    ) -> List[PlacementSamplerRegion]:
        """
        :param environment: A reset RoboCasa/robosuite task environment.
        :param name_prefix: Prefix the attached regions' names are given, so regions
            read from different tasks or resets stay distinguishable.
        :return: One entry per object with a known footprint whose sampler resolved a
            non degenerate region.
        """
        regions = []
        for (
            sampler_name,
            sampler,
        ) in environment.placement_initializer.samplers.items():
            object_name = sampler_name[: -len(self.sampler_suffix)]
            robocasa_object = environment.objects.get(object_name)
            if robocasa_object is None or not hasattr(
                robocasa_object, "get_bbox_points"
            ):
                continue
            region = self.read_one(
                sampler, name_prefix, object_name, robocasa_object.root_body
            )
            if region is not None:
                regions.append(region)
        return regions

    @staticmethod
    def read_one(
        sampler: Any, name_prefix: str, object_name: str, body_name: str
    ) -> Optional[PlacementSamplerRegion]:
        """
        :param sampler: The sampler an environment built for an object.
        :param name_prefix: Prefix the attached region's name is given.
        :param object_name: Name the environment gave the object.
        :param body_name: Name of the object's root body in the world.
        :return: The region's data, or ``None`` when the sampler resolved a degenerate
            (zero area) region.
        """
        x_minimum, x_maximum = sampler.x_range
        y_minimum, y_maximum = sampler.y_range
        width = x_maximum - x_minimum
        depth = y_maximum - y_minimum
        if width <= 0.0 or depth <= 0.0:
            return None
        world_x, world_y, world_z = (float(value) for value in sampler.reference_pos)
        return PlacementSamplerRegion(
            name=PrefixedName(object_name, prefix=name_prefix),
            body_name=body_name,
            local_center=Point3(
                (x_minimum + x_maximum) / 2.0,
                (y_minimum + y_maximum) / 2.0,
            ),
            world_T_sampler=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=world_x,
                y=world_y,
                z=world_z,
                yaw=float(sampler.reference_rot),
            ),
            width=float(width),
            depth=float(depth),
        )


@dataclass(frozen=True)
class GripperExclusionZoneData:
    """
    A ball around a body's position that a task's success condition requires the gripper
    to stay outside of, read from a live environment rather than reconstructed from a
    task's source.

    The admissible region this describes is this ball's complement, not the ball
    itself: unlike a placement region, what a success condition states directly is the
    excluded volume, and the space the gripper may occupy is everything else in the
    world.
    """

    name: PrefixedName
    """
    Name the attached Region is given.
    """

    body_name: str
    """
    Name of the body, already present in the world, the gripper must stay away from.
    """

    center: Point3
    """
    World coordinates of the body at the moment the zone was resolved.
    """

    radius: float
    """
    Distance from the centre the gripper must stay beyond.
    """

    def attach_to(self, world: World) -> GripperExclusionZone:
        """
        :param world: The world to attach the zone to, connected to its root at this
            zone's resolved world position.
        :return: The attached region's semantic annotation, recording which object the
            gripper is being kept away from.
        """
        sphere = Sphere(radius=self.radius)
        region = Region.from_shape_collection(
            name=self.name, shape_collection=ShapeCollection(shapes=[sphere])
        )
        connection = FixedConnection(
            parent=world.root,
            child=region,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=self.center
            ),
        )
        annotation = GripperExclusionZone(
            root=region, excluded_object=world.get_body_by_name(self.body_name)
        )
        with world.modify_world():
            world.add_connection(connection)
            world.add_semantic_annotation(annotation)
        return annotation


@dataclass
class GripperExclusionZoneReader:
    """
    Reads a gripper exclusion zone from a reset RoboCasa environment.
    """

    @staticmethod
    def read(
        environment: Any, name_prefix: str, object_name: str, radius: float
    ) -> GripperExclusionZoneData:
        """
        :param environment: A reset RoboCasa/robosuite task environment.
        :param name_prefix: Prefix the attached region's name is given.
        :param object_name: Name the environment gave the object the gripper must stay
            away from.
        :param radius: Distance from the body's centre the gripper must stay beyond.
        :return: The zone's data, centred on the body's resolved position.
        """
        position = environment.sim.data.body_xpos[environment.obj_body_id[object_name]]
        return GripperExclusionZoneData(
            name=PrefixedName(object_name, prefix=name_prefix),
            body_name=environment.objects[object_name].root_body,
            center=Point3(*(float(value) for value in position)),
            radius=radius,
        )

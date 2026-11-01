from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from types import NoneType
from typing_extensions import (
    TYPE_CHECKING,
    TypeVar,
    Generic,
    TypeVarTuple,
    Unpack,
    Union,
    get_args,
    get_origin,
)


from krrood.ormatic.utils import classproperty
from krrood.patterns.subclass_safe_generic import (
    SubClassSafeGeneric,
)
from krrood.utils import get_existing_field_by_name, get_generic_type_parameters
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    JointStateType,
)
from semantic_digital_twin.datastructures.lidar_reading import LidarReading
from semantic_digital_twin.reasoning.predicates import LeftOf, RightOf
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose, Vector3
from semantic_digital_twin.robots.exceptions import (
    MissingEndEffectorError,
    MissingInputSourceError,
    MissingLidarError,
    MissingMobileBaseError,
    MissingNeckError,
    MissingSensorsError,
    MissingTorsoError,
    TooFewArmsError,
    TooFewFingersError,
    UnexpectedArmCountError,
    UnexpectedFingerCountError,
    UnexpectedInputSourceError,
    UndeclaredTopicError,
)
from semantic_digital_twin.robots.input_source import InputSource

if TYPE_CHECKING:
    from rclpy.node import Node
    from semantic_digital_twin.datastructures.joint_state import JointState
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.connections import (
        ActiveConnection1DOF,
    )
    from semantic_digital_twin.world_description.geometry import BoundingBox
    from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger("semantic_digital_twin")

TGenericFingerOtherThanThumb = TypeVar("TGenericFingerOtherThanThumb")
TGenericThumb = TypeVar("TGenericThumb")
TGenericCamera = TypeVar("TGenericCamera")
TGenericEndEffector = TypeVar("TGenericEndEffector")
TGenericArm = TypeVar("TGenericArm")
TGenericMobileBase = TypeVar("TGenericMobileBase")
TGenericMountingTable = TypeVar("TGenericMountingTable")
TGenericTorso = TypeVar("TGenericTorso")
TGenericNeck = TypeVar("TGenericNeck")
TGenericLeftArm = TypeVar("TGenericLeftArm")
TGenericRightArm = TypeVar("TGenericRightArm")
TGenericLeftFinger = TypeVar("TGenericLeftFinger")
TGenericRightFinger = TypeVar("TGenericRightFinger")

TGenericFingers = TypeVarTuple("TGenericFingers")
TGenericArms = TypeVarTuple("TGenericArms")
TGenericSensors = TypeVarTuple("TGenericSensors")
TGenericLidar = TypeVar("TGenericLidar")
TGenericInputSource = TypeVar("TGenericInputSource", bound=InputSource)


@dataclass(eq=False)
class RobotPartMixin(ABC):
    """
    Base mixin class for robot parts.

    Every mixin states its own assumption in :meth:`validate` and then hands the check
    on to the next mixin of the part, so that a part combining several of them has all
    of their assumptions checked rather than only the first one's.
    """

    def validate(self):
        """
        Checks the assumptions this mixin makes about the robot part.

        Ends the chain of checks a part's mixins hand along, so a mixin that makes no
        assumption of its own needs no implementation.
        """


@dataclass(eq=False)
class HasFingers(
    Generic[TGenericThumb, Unpack[TGenericFingers]],
    SubClassSafeGeneric,
    RobotPartMixin,
    ABC,
):
    """
    Mixin class for robots or robot parts that have fingers as their direct children.
    """

    fingers: list[TGenericThumb | Unpack[TGenericFingers]] = field(
        default_factory=list, kw_only=True
    )
    """
    The list of fingers attached to the robot.
    """

    def validate(self):
        """
        :raises TooFewFingersError: If fewer fingers are attached than this mixin
            allows.
        """
        if len(self.fingers) < 2:
            raise TooFewFingersError(
                robot_part=self,
                minimum_count=2,
                actual_count=len(self.fingers),
            )
        super().validate()

    @property
    def thumb(self) -> TGenericThumb:
        concrete_thumb_class = get_generic_type_parameters(self, HasFingers)[0]
        [thumb] = [
            finger
            for finger in self.fingers
            if isinstance(finger, concrete_thumb_class)
        ]
        return thumb


# %% the fingertips a gripper closes with


@dataclass
class Fingertips:
    """
    The pair of fingertips a two-finger gripper closes an object between.

    Where a gripper grasps and how wide it stands are read off these two bodies rather
    than off the gripper's tool frame, which need not sit where the fingers meet.
    """

    thumb: Body
    """
    The tip of the thumb.
    """

    finger: Body
    """
    The tip of the other finger.
    """

    tool_frame: Body
    """
    The frame the measurements are expressed in.
    """

    closing_axis: Vector3
    """
    The axis the fingers close along, in :attr:`tool_frame`.
    """

    @property
    def both(self) -> tuple[Body, Body]:
        """
        The two tips, thumb first.
        """
        return self.thumb, self.finger

    # %% where they meet

    @property
    def tool_P_grasp(self) -> Point3:
        """
        Where the tips meet as the gripper stands, in :attr:`tool_frame`.

        A gripper's tool frame need not sit there, and aiming the tool frame itself at
        an object then closes the fingers beside it rather than on it. Each tip's own
        collision bounding box centre stands in for where that finger is, since a tip's
        link origin may sit at an edge of its mesh rather than at its centre.

        Read it inside a
        :meth:`~semantic_digital_twin.world.World.reset_state_context` to read it for a
        state the gripper is not standing in.
        """
        thumb, finger = (self._bounding_box(tip).center for tip in self.both)
        return thumb.midpoint(finger)

    # %% how wide they stand

    @property
    def separation(self) -> float:
        """
        How far apart the tips' facing surfaces stand right now, in metres, measured
        along :attr:`closing_axis` from :attr:`finger_P_facing_point` to
        :attr:`thumb_P_facing_point`: the widest object the gripper could close on as it
        stands.

        Negative where the surfaces already overlap, as a fully closed gripper's may.
        """
        tool_P_finger, tool_P_thumb = (
            self._world.transform(target_frame=self.tool_frame, spatial_object=point)
            for point in (self.finger_P_facing_point, self.thumb_P_facing_point)
        )
        return float((tool_P_thumb - tool_P_finger).dot(self.tool_V_finger_to_thumb))

    @property
    def tool_V_finger_to_thumb(self) -> Vector3:
        """
        :attr:`closing_axis`, pointing from the finger's tip towards the thumb's, in
        :attr:`tool_frame`.
        """
        thumb, finger = (
            self._closing_axis_coordinate(self._bounding_box(tip).center)
            for tip in self.both
        )
        if thumb < finger:
            return -self.closing_axis
        return self.closing_axis

    @property
    def thumb_P_facing_point(self) -> Point3:
        """
        The point of the thumb's tip that reaches furthest towards the finger's along
        :attr:`closing_axis`, in the thumb tip's own frame.
        """
        return self._facing_point(self.thumb, -self.tool_V_finger_to_thumb)

    @property
    def finger_P_facing_point(self) -> Point3:
        """
        The point of the finger's tip that reaches furthest towards the thumb's along
        :attr:`closing_axis`, in the finger tip's own frame.
        """
        return self._facing_point(self.finger, self.tool_V_finger_to_thumb)

    # %% reading the geometry

    @property
    def _world(self) -> World:
        """
        The world the tips belong to.
        """
        return self.thumb._world

    def _facing_point(self, tip: Body, tool_V_towards_other_tip: Vector3) -> Point3:
        """
        :param tip: One of the two tips.
        :param tool_V_towards_other_tip: The direction towards the other tip, in
            :attr:`tool_frame`.
        :return: The corner of the tip's collision bounding box, taken in the tip's own
            frame, that reaches furthest in that direction, in the tip's own frame.
        """
        box = tip.collision.as_bounding_box_collection_in_frame(tip).bounding_box()
        tool_T_tip = self._world.compute_forward_kinematics(self.tool_frame, tip)
        corners = (
            Point3(*corner, reference_frame=tip)
            for corner in product(
                (box.min_x, box.max_x), (box.min_y, box.max_y), (box.min_z, box.max_z)
            )
        )
        return max(
            corners,
            key=lambda tip_P_corner: float(
                (tool_T_tip @ tip_P_corner).vector3.dot(tool_V_towards_other_tip)
            ),
        )

    def _bounding_box(self, tip: Body) -> BoundingBox:
        """
        :param tip: One of the two tips.
        :return: Its collision bounding box, in :attr:`tool_frame`.
        """
        return tip.collision.as_bounding_box_collection_in_frame(
            self.tool_frame
        ).bounding_box()

    def _closing_axis_coordinate(self, tool_P_point: Point3) -> float:
        """
        :param tool_P_point: A point in :attr:`tool_frame`.
        :return: Where it lies along :attr:`closing_axis`.
        """
        return float(tool_P_point.vector3.dot(self.closing_axis))


@dataclass(eq=False)
class HasTwoFingers(
    Generic[TGenericLeftFinger, TGenericRightFinger],
    HasFingers[TGenericLeftFinger, TGenericRightFinger],
    SubClassSafeGeneric,
    ABC,
):
    """
    Mixin class for robots or robot parts that have exactly two fingers, one of which is
    a thumb.
    """

    def validate(self):
        """
        :raises UnexpectedFingerCountError: If a different number of fingers is attached
            than this mixin allows.
        """
        if len(self.fingers) != 2:
            raise UnexpectedFingerCountError(
                robot_part=self,
                expected_count=2,
                actual_count=len(self.fingers),
            )
        super().validate()

    @classproperty
    def fingers_are_modelled_like_the_real_ones(cls) -> bool:
        """
        Whether the description's fingers match the real gripper's closely enough that a
        width measured or reached on the model holds on the real gripper.
        """
        return True

    @classproperty
    def squeeze(cls) -> float:
        """
        How far past the width of what it grasps each finger is sent in a physical
        simulation, in metres: a servo grips only with the position error it is left
        with, so fingers sent no further than the object's surface touch it without
        holding it.
        """
        return 0.005

    @property
    def finger(self) -> TGenericLeftFinger | TGenericRightFinger:
        concrete_thumb_class = get_generic_type_parameters(self, HasFingers)[0]

        [finger] = [
            finger
            for finger in self.fingers
            if not isinstance(finger, concrete_thumb_class)
        ]
        return finger

    @property
    def fingertips(self) -> Fingertips:
        """
        The pair of tips this gripper closes an object between, which is what answers
        where it grasps and how wide it stands.
        """
        return Fingertips(
            thumb=self.thumb.tip,
            finger=self.finger.tip,
            tool_frame=self.tool_frame,
            closing_axis=self.closing_axis,
        )

    @cached_property
    def travel_when_closing(self) -> float:
        """
        How much further along :attr:`EndEffector.approach_axis` the fingers meet when
        closed than when open, in metres. Zero for fingers that only slide.

        A constant of the gripper's description, read once and kept. The gripper is put
        through its own open and closed states to read it, inside a
        :meth:`~semantic_digital_twin.world.World.reset_state_context` that stands the
        world back up afterwards.
        """
        fingertips = self.fingertips
        with self._world.reset_state_context():
            open_P_grasp, closed_P_grasp = (
                self._meeting_point_in_state(fingertips, state_type)
                for state_type in (GripperState.OPEN, GripperState.CLOSE)
            )
        travelled = closed_P_grasp - open_P_grasp
        return float(travelled.dot(self.approach_axis))

    def _meeting_point_in_state(
        self, fingertips: Fingertips, state_type: JointStateType
    ) -> Point3:
        """
        :param fingertips: The tips to read.
        :param state_type: The state to stand the gripper in.
        :return: Where the tips meet with the gripper standing in that state.

        ..warning:: This leaves the gripper standing in that state. Call it inside a
            :meth:`~semantic_digital_twin.world.World.reset_state_context`.
        """
        self.get_joint_state_by_type(state_type).apply_to(self._world)
        return fingertips.tool_P_grasp

    @cached_property
    def tool_P_grasp_when_closed(self) -> Point3:
        """
        Where the fingers meet once closed, in :attr:`tool_frame`.

        Jaws on a linkage carry the meeting point :attr:`travel_when_closing` further
        out as they close, so where the fingers meet depends on how far the gripper
        happens to stand open. A grasp planned while open and a release planned while
        holding would then be aimed at two different points, and an object placed after
        being picked up would land beside its target by that difference. This is the
        point an object is held at, which is the one both have to be aimed at.

        A constant of the gripper's description, read once and kept. The same point as
        :attr:`Fingertips.tool_P_grasp` for fingers that only slide.
        """
        with self._world.reset_state_context():
            return self._meeting_point_in_state(self.fingertips, GripperState.CLOSE)

    @cached_property
    def separation_when_open(self) -> float:
        """
        How far apart the fingertips' facing surfaces stand with the gripper open, in
        metres: the widest object it can close on.

        A constant of the gripper's description, read once and kept.
        """
        return self.separation_in(GripperState.OPEN)

    def separation_in(self, state_type: JointStateType) -> float:
        """
        :param state_type: The state to stand the gripper in.
        :return: How far apart the fingertips' facing surfaces stand with the gripper in
            the joint state of that type, see :meth:`separation_at`.
        """
        return self.separation_at(self.get_joint_state_by_type(state_type))

    def separation_at(self, joint_state: JointState) -> float:
        """
        :param joint_state: The joint state to stand the gripper in.
        :return: How far apart the fingertips' facing surfaces stand with the gripper in
            that joint state, in metres, see :attr:`Fingertips.separation`. The world
            stands as it did afterwards.
        """
        with self._world.reset_state_context():
            joint_state.apply_to(self._world)
            return self.fingertips.separation

    @property
    def finger_connections(self) -> list[ActiveConnection1DOF]:
        """
        The connections the gripper's own states drive its fingers by, each once.
        """
        connections = []
        for joint_state in self.joint_states:
            for connection in joint_state.connections:
                if connection not in connections:
                    connections.append(connection)
        return connections

    # %% reaching and reading back a grasp

    def tool_frame_goal_for_grasp(self, grasp_pose: Pose) -> Pose:
        """
        Express a grasp frame as the goal that puts :attr:`tool_P_grasp_when_closed` on
        it.

        :param grasp_pose: The grasp frame to meet.
        :return: The pose the tool frame has to reach, in ``grasp_pose``'s frame.
        """
        return self.tool_frame_goal_for_point(self.tool_P_grasp_when_closed, grasp_pose)

    @property
    def held_body_T_grasp(self) -> Pose:
        """
        The grasp this gripper has on the body it holds, read at the point it holds an
        object, which is the point :meth:`tool_frame_goal_for_grasp` aims at.

        :return: The grasp frame, in the held body's frame.
        """
        return self.held_body_T_grasp_at(self.tool_P_grasp_when_closed)


@dataclass(eq=False)
class HasSensors(
    Generic[Unpack[TGenericSensors]], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for robots or robot parts that have sensors.
    """

    sensors: list[Unpack[TGenericSensors]] = field(default_factory=list, kw_only=True)
    """
    The list of sensors associated with the robot part.
    """

    def validate(self):
        """
        :raises MissingSensorsError: If no sensor is attached.
        """
        if not self.sensors:
            raise MissingSensorsError(robot_part=self)
        super().validate()


@dataclass(eq=False)
class HasEndEffector(
    Generic[TGenericEndEffector], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for robots or robot parts that have an end effector as their direct
    child.
    """

    end_effector: TGenericEndEffector = field(default=None, kw_only=True)
    """
    The end effector attached to the robot part.
    """

    def validate(self):
        """
        :raises MissingEndEffectorError: If no end effector is attached.
        """
        if self.end_effector is None:
            raise MissingEndEffectorError(robot_part=self)
        super().validate()


@dataclass(eq=False)
class HasArms(Generic[Unpack[TGenericArms]], SubClassSafeGeneric, RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have arms as their direct children.
    """

    arms: list[Unpack[TGenericArms]] = field(default_factory=list, kw_only=True)
    """
    The list of arms attached to the robot part.
    """

    def validate(self):
        """
        :raises TooFewArmsError: If fewer arms are attached than this mixin allows.
        """
        if len(self.arms) < 1:
            raise TooFewArmsError(
                robot_part=self,
                minimum_count=1,
                actual_count=len(self.arms),
            )
        super().validate()


@dataclass(eq=False)
class HasOneArm(HasArms[TGenericArm], RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have exactly one arm.
    """

    def validate(self):
        """
        :raises UnexpectedArmCountError: If a different number of arms is attached than
            this mixin allows.
        """
        if len(self.arms) != 1:
            raise UnexpectedArmCountError(
                robot_part=self,
                expected_count=1,
                actual_count=len(self.arms),
            )
        super().validate()

    @property
    def arm(self) -> TGenericArm:
        [arm] = self.arms
        return arm


@dataclass(eq=False)
class HasLeftRightArm(
    HasArms[TGenericLeftArm, TGenericRightArm],
    SubClassSafeGeneric,
    RobotPartMixin,
    ABC,
):
    """
    Mixin class for robots or robot parts that have two arms and can specify which is
    the left and which is the right arm.
    """

    def validate(self):
        """
        :raises UnexpectedArmCountError: If a different number of arms is attached than
            this mixin allows.
        """
        self._validate_arm_count()
        super().validate()

    def _validate_arm_count(self):
        """
        :raises UnexpectedArmCountError: If a different number of arms is attached than
            this mixin allows.
        """
        if len(self.arms) != 2:
            raise UnexpectedArmCountError(
                robot_part=self,
                expected_count=2,
                actual_count=len(self.arms),
            )

    @cached_property
    def left_arm(self) -> TGenericLeftArm:
        from semantic_digital_twin.reasoning.predicates import LeftOf

        return self._assign_left_right_arms(LeftOf)

    @cached_property
    def right_arm(self) -> TGenericRightArm:
        from semantic_digital_twin.reasoning.predicates import RightOf

        return self._assign_left_right_arms(RightOf)

    def _assign_left_right_arms(
        self, relation: type[LeftOf | RightOf]
    ) -> TGenericLeftArm | TGenericRightArm:
        """
        Assigns the left and right arms based on their position relative to the robot's
        root body.

        :param relation: The relation to use for determining left or right (LeftOf or
            RightOf).
        :return: The arm that is on the left or right side of the robot.
        :raises UnexpectedArmCountError: If a different number of arms is attached than
            this mixin allows.
        """
        self._validate_arm_count()
        pov = self.root.global_transform
        [first_arm, second_arm] = self.arms
        # the arms may share a root, but the first body after the root should be different
        world_P_first_body = first_arm.bodies[1].global_transform.position
        world_P_second_body = second_arm.bodies[1].global_transform.position

        return (
            first_arm
            if relation(
                world_P_first_body,
                world_P_second_body,
                pov,
            )()
            else second_arm
        )


@dataclass(eq=False)
class HasMobileBase(
    Generic[TGenericMobileBase], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for robots that have a mobile base.
    """

    mobile_base: TGenericMobileBase = field(default=None, kw_only=True)
    """
    The mobile base attached to the robot part.
    """

    def validate(self):
        """
        :raises MissingMobileBaseError: If no mobile base is attached.
        """
        if self.mobile_base is None:
            raise MissingMobileBaseError(robot_part=self)
        super().validate()


@dataclass(eq=False)
class HasMountingTable(
    Generic[TGenericMountingTable], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for stationary robots bolted onto a table.
    """

    table: TGenericMountingTable = field(default=None, kw_only=True)
    """
    The table the robot is mounted on.
    """

    def validate(self):
        assert self.table is not None, "Expected table, got None"


@dataclass(eq=False)
class HasTorso(Generic[TGenericTorso], SubClassSafeGeneric, RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have a torso as their direct child.
    """

    torso: TGenericTorso = field(default=None, kw_only=True)
    """
    The torso attached to the robot part.
    """

    def validate(self):
        """
        :raises MissingTorsoError: If no torso is attached.
        """
        if self.torso is None:
            raise MissingTorsoError(robot_part=self)
        super().validate()


@dataclass(eq=False)
class HasNeck(Generic[TGenericNeck], SubClassSafeGeneric, RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have a neck as their direct child.
    """

    neck: TGenericNeck = field(default=None, kw_only=True)
    """
    The neck attached to the robot part.
    """

    def validate(self):
        """
        :raises MissingNeckError: If no neck is attached.
        """
        if self.neck is None:
            raise MissingNeckError(robot_part=self)
        super().validate()


@dataclass(eq=False)
class HasLidar(Generic[TGenericLidar], SubClassSafeGeneric, RobotPartMixin, ABC):
    """
    Mixin class for robots or robot parts that have a lidar as their direct child.
    """

    lidar: TGenericLidar = field(default=None, kw_only=True)
    """
    The lidar attached to the robot part.
    """

    def validate(self):
        """
        :raises MissingLidarError: If no lidar is attached.
        """
        if self.lidar is None:
            raise MissingLidarError(robot_part=self)
        super().validate()

    def get_lidar_reading(self) -> LidarReading:
        """
        :return: The most recent sweep of the attached lidar.
        """
        return self.lidar.get_lidar_reading()


# %% where a part is read from


@dataclass(eq=False)
class HasInputSource(
    Generic[TGenericInputSource], SubClassSafeGeneric, RobotPartMixin, ABC
):
    """
    Mixin class for robot parts that can be read either from the world they stand in or
    from the robot they stand for.

    The kind of source a part can be read from is bound as the generic parameter, so a
    part cannot be handed a source meant for another kind of part.
    """

    source: TGenericInputSource | None = field(default=None, kw_only=True)
    """
    Where this part is read from.

    ..note:: A family of parts re-declares this field under the same type variable, to
        give it the default its own kind of source has.
    """

    @classproperty
    def topic_name(cls) -> str | None:
        """
        The topic the real robot publishes this part's state on, if its description
        names one.
        """
        return None

    def validate(self):
        """
        :raises MissingInputSourceError: If nothing says where this part is read from.
        """
        if self.source is None:
            raise MissingInputSourceError(robot_part=self)
        super().validate()

    @classmethod
    def source_family(cls) -> type[TGenericInputSource]:
        """
        :return: The kind of source this part can be read from.

        ..note:: Read off :attr:`source`, which :class:`SubClassSafeGeneric` narrows to
            the type the part binds, so the binding stays the only place it is stated.
        """
        source_type = get_existing_field_by_name(cls, "source").type
        if get_origin(source_type) is not Union:
            return source_type
        [source_family] = [
            member for member in get_args(source_type) if member is not NoneType
        ]
        return source_family

    @classmethod
    @abstractmethod
    def simulated_source(cls) -> TGenericInputSource:
        """
        :return: The source reading this part from the world it stands in.
        """

    @abstractmethod
    def real_source(self, node: Node) -> TGenericInputSource:
        """
        :param node: The ros node the messages are received on.
        :return: The source reading this part from the robot itself, on the topic this
            part declares.
        """

    def use_simulated_source(self) -> None:
        """
        Read this part from the world it stands in.
        """
        self.use_source(self.simulated_source())

    def use_real_source(self, node: Node) -> None:
        """
        Read this part from the robot itself, on the topic it declares.

        :param node: The ros node the messages are received on.
        :raises UndeclaredTopicError: If this part declares no topic.
        """
        if self.topic_name is None:
            raise UndeclaredTopicError(robot_part=self)
        self.use_source(self.real_source(node))

    def use_source(self, source: TGenericInputSource) -> None:
        """
        Read this part from the given source from now on, releasing the one it was read
        from before.

        :param source: Where this part is read from.
        :raises UnexpectedInputSourceError: If the source is not one this part can be
            read from.
        """
        if not isinstance(source, self.source_family()):
            raise UnexpectedInputSourceError(
                robot_part=self,
                source=source,
                expected_source_family=self.source_family(),
            )
        if self.source is not None and self.source is not source:
            self.source.close()
        self.source = source

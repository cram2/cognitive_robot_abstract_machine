from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Optional

from geometry_msgs.msg import WrenchStamped

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.ros2_nodes.topic_monitor import TopicSubscriberNode
from semantic_digital_twin.spatial_types import Vector3
from enum import Enum

from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


class ForceTorqueThresholds(Enum):
    GRASP = 1
    PLACE = 2
    DOOR = 3
    DISHDOOR = 4
    POURING = 5  # not currently in use
    SHELF_GRASP = 6
    WIPING = 7
    HRI_GRASP = 8


class ObjectTypes(Enum):
    OT_Default = "Default"
    OT_Cutlery = "Cutlery"
    OT_Plate = "Plate"
    OT_Bowl = "Bowl"
    OT_Tray = "Tray"


@dataclass(eq=False, repr=False)
class PayloadForceTorque(TopicSubscriberNode[WrenchStamped]):
    """
    Monitors a WrenchStamped topic and turns True when the force/torque reading
    exceeds the threshold defined by the given threshold_enum and object_type.

    The wrench is transformed from the sensor frame into a configurable reference
    frame before threshold comparison.

    :param threshold_enum: Enum value selecting which ThresholdStrategy to use
                           (e.g. ForceTorqueThresholds.DOOR.value).
    :param topic_name: ROS2 topic publishing WrenchStamped messages.
    :param object_type: Object-type string used by GRASP/PLACE strategies to
                        select per-object thresholds. Pass an empty string or
                        None for strategies that do not require it (DOOR, WIPING,
                        HRI_GRASP, SHELF_GRASP).
    :param stay_true: If True the node latches in the TRUE state once triggered
                      and will not revert to FALSE on subsequent ticks.
    """

    threshold_enum: int = field(kw_only=True)
    topic_name: str = field(kw_only=True)
    object_type: Optional[str] = field(default=None, kw_only=True)
    stay_true: bool = field(default=True, kw_only=True)
    msg_type: type = field(init=False, default=WrenchStamped)

    # Resolved at build time
    _strategy: ThresholdStrategy = field(init=False, repr=False)
    _reference_frame: KinematicStructureEntity = field(init=False, repr=False)
    _sensor_frame: Optional[KinematicStructureEntity] = field(
        init=False, repr=False, default=None
    )

    # Latest raw message, written by callback and consumed by on_tick
    _latest_msg: Optional[WrenchStamped] = field(init=False, repr=False, default=None)
    _latched: bool = field(init=False, repr=False, default=False)

    def build(self, context: MotionStatechartContext):
        self._strategy = ThresholdStrategyFactory.get_strategy(
            self.object_type, self.threshold_enum
        )
        self._reference_frame = context.world.get_kinematic_structure_entity_by_name(
            self._strategy.reference_frame
        )
        return super().build(context)

    def callback(self, msg: WrenchStamped) -> None:
        """
        ROS2 subscriber callback. Caches the raw message for processing in
        on_tick, where the context is available for frame transforms.
        """
        self._latest_msg = msg

    def _transform_wrench(self, context: MotionStatechartContext, msg: WrenchStamped):
        """
        Transforms the force and torque vectors from the sensor frame into the
        strategy's reference frame using the context world.
        """
        if self._sensor_frame is None:
            self._sensor_frame = context.world.get_kinematic_structure_entity_by_name(
                msg.header.frame_id
            )

        force_vec = Vector3(
            x=msg.wrench.force.x,
            y=msg.wrench.force.y,
            z=msg.wrench.force.z,
            reference_frame=self._sensor_frame,
        )
        torque_vec = Vector3(
            x=msg.wrench.torque.x,
            y=msg.wrench.torque.y,
            z=msg.wrench.torque.z,
            reference_frame=self._sensor_frame,
        )

        force_transformed = context.world.transform(
            target_frame=self._reference_frame,
            spatial_object=force_vec,
        )
        torque_transformed = context.world.transform(
            target_frame=self._reference_frame,
            spatial_object=torque_vec,
        )

        return force_transformed, torque_transformed

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        super().on_tick(context)

        if not self.has_msg():
            return ObservationStateValues.UNKNOWN

        msg = copy(self.current_msg)
        rob_force, rob_torque = self._transform_wrench(context, msg)

        # Latch: once TRUE, stay TRUE
        if self.stay_true and self._latched:
            return ObservationStateValues.TRUE

        if self._strategy.check_thresholds(rob_force, rob_torque):
            self._latched = True
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

    def on_reset(self, context: MotionStatechartContext):
        super().on_reset(context)
        self._latest_msg = None
        self._sensor_frame = None
        self._latched = False


# ---------------------------------------------------------------------------
# Threshold strategies
# ---------------------------------------------------------------------------


class ThresholdStrategy:
    reference_frame: str = "base_footprint"

    def check_thresholds(self, rob_force, rob_torque) -> bool:
        raise NotImplementedError("This method should be overridden.")


class GraspThresholdStrategy(ThresholdStrategy):
    def __init__(self, object_type: str):
        self.object_type = object_type

    def check_thresholds(self, rob_force, rob_torque) -> bool:
        if self.object_type == ObjectTypes.OT_Default.value:
            torque_y_threshold = 2
            return abs(rob_torque[1]) > torque_y_threshold

        elif self.object_type == ObjectTypes.OT_Cutlery.value:
            force_z_threshold = 85
            return abs(rob_force[2]) > force_z_threshold

        # NOT CURRENTLY USED: plates are neither placed nor picked up
        elif self.object_type == ObjectTypes.OT_Plate.value:
            torque_y_threshold = 0.02
            return (
                abs(rob_force[1]) > torque_y_threshold
                or abs(rob_force[1]) > torque_y_threshold
            )

        elif self.object_type == ObjectTypes.OT_Bowl.value:
            force_z_threshold = 16
            return abs(rob_force[2]) >= force_z_threshold

        # NOT CURRENTLY IN USE: tray is not picked up
        elif self.object_type == ObjectTypes.OT_Tray.value:
            force_y_threshold = 60.0
            # TODO: change to correct axis
            return abs(rob_force[1]) > force_y_threshold

        else:
            raise ValueError(
                f"No valid object_type found for GraspThresholdStrategy: {self.object_type!r}"
            )


class PlaceThresholdStrategy(ThresholdStrategy):
    def __init__(self, object_type: str):
        self.object_type = object_type

    # TODO: Change approach of how values are being chosen (e.g. data-driven approach or soft thresholding)
    def check_thresholds(self, rob_force, rob_torque) -> bool:
        force_z_threshold = 35

        if self.object_type == ObjectTypes.OT_Default.value:
            return abs(rob_force[2]) >= force_z_threshold

        elif self.object_type == ObjectTypes.OT_Cutlery.value:
            return abs(rob_force[2]) >= force_z_threshold

        # NOT CURRENTLY USED: plates are neither placed nor picked up
        elif self.object_type == ObjectTypes.OT_Plate.value:
            force_z_threshold = 1.0
            return abs(rob_force[2]) >= force_z_threshold

        elif self.object_type == ObjectTypes.OT_Bowl.value:
            return abs(rob_force[2]) >= force_z_threshold

        elif self.object_type == ObjectTypes.OT_Tray.value:
            return abs(rob_force[2]) >= force_z_threshold

        else:
            raise ValueError(
                f"No valid object_type found for PlaceThresholdStrategy: {self.object_type!r}"
            )


class DoorThresholdStrategy(ThresholdStrategy):
    reference_frame = "hand_gripper_tool_frame"

    def check_thresholds(self, rob_force, rob_torque) -> bool:
        force_z_threshold = 80
        return abs(rob_force[2]) >= force_z_threshold


class ShelfGraspThresholdStrategy(ThresholdStrategy):
    reference_frame = "hand_gripper_tool_frame"

    def check_thresholds(self, rob_force, rob_torque) -> bool:
        force_z_threshold = 30
        return abs(rob_force[2]) >= force_z_threshold


class WipeThresholdStrategy(ThresholdStrategy):
    reference_frame = "hand_gripper_tool_frame"

    def check_thresholds(self, rob_force, rob_torque) -> bool:
        # TODO: Establish proper threshold value
        force_z_threshold = 1
        return abs(rob_force[2]) >= force_z_threshold


class HRIGThresholdStrategy(ThresholdStrategy):
    reference_frame = "hand_gripper_tool_frame"

    def check_thresholds(self, rob_force, rob_torque) -> bool:
        # TODO: Establish proper threshold value
        # Consider replacing with y-torque check if force_z proves unreliable
        force_z_threshold = 20
        return abs(rob_force[2]) >= force_z_threshold


class ThresholdStrategyFactory:
    """
    Selects and returns the appropriate ThresholdStrategy based on the given
    threshold_enum and object_type.
    """

    @staticmethod
    def get_strategy(
        object_type: Optional[str], threshold_enum: int
    ) -> ThresholdStrategy:
        if threshold_enum == ForceTorqueThresholds.GRASP.value:
            return GraspThresholdStrategy(object_type)
        elif threshold_enum == ForceTorqueThresholds.PLACE.value:
            return PlaceThresholdStrategy(object_type)
        elif threshold_enum == ForceTorqueThresholds.DOOR.value:
            return DoorThresholdStrategy()
        elif threshold_enum == ForceTorqueThresholds.SHELF_GRASP.value:
            return ShelfGraspThresholdStrategy()
        elif threshold_enum == ForceTorqueThresholds.WIPING.value:
            return WipeThresholdStrategy()
        elif threshold_enum == ForceTorqueThresholds.HRI_GRASP.value:
            return HRIGThresholdStrategy()
        else:
            raise ValueError(f"Invalid threshold_enum: {threshold_enum}")

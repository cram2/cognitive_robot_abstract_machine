from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from typing_extensions import Optional

from krrood.adapters.json_serializer import SubclassJSONSerializer


class GriplinkGripPreset(Enum):
    """
    Grip preset selectable on a WEISS WPG gripper controller.
    """

    PRESET_0 = 0
    PRESET_1 = 1
    PRESET_2 = 2
    PRESET_3 = 3
    PRESET_4 = 4
    PRESET_5 = 5
    PRESET_6 = 6
    PRESET_7 = 7


@dataclass
class GriplinkGripperConfiguration(SubclassJSONSerializer):
    """
    Hardware parameters for the WEISS WPG 300-120 gripper, forwarded to the griplink
    action server.

    Only the fields relevant to the commanded motion are read: :attr:`grip_preset` for
    ``Grip``/``Release``, and :attr:`grip_position`/:attr:`grip_force`/
    :attr:`grip_speed`/:attr:`grip_acceleration` for ``Flexgrip``/``Flexrelease``
    (:attr:`grip_force` is ignored for ``Flexrelease``, which has no force goal).
    """

    grip_preset: GriplinkGripPreset = GriplinkGripPreset.PRESET_0
    """
    Stored grip preset selected on the controller, used by ``Grip``/``Release``.
    """

    grip_position: Optional[int] = None
    """
    Opening width of the gripper in millimetres [-5..120], used by ``Flexgrip``/
    ``Flexrelease``.

    ``None`` defers to the per-motion default
    :class:`~giskardpy.motion_statechart.ros2_nodes.griplink.GriplinkFlexActionServerTask`
    picks in ``build_msg`` (0 for ``Flexgrip``, 120 for ``Flexrelease``).
    """

    grip_force: Optional[int] = None
    """
    Force the gripper applies to the object in newtons [30..300], used by ``Flexgrip``
    only (ignored for ``Flexrelease``).

    ``None`` defers to the default
    :class:`~giskardpy.motion_statechart.ros2_nodes.griplink.GriplinkFlexActionServerTask`
    picks in ``build_msg`` (90).
    """

    grip_speed: Optional[int] = None
    """
    Motion speed of the gripper in millimetres per second [5..350], used by
    ``Flexgrip``/``Flexrelease``.

    ``None`` defers to the per-motion default
    :class:`~giskardpy.motion_statechart.ros2_nodes.griplink.GriplinkFlexActionServerTask`
    picks in ``build_msg`` (150 for ``Flexgrip``, 250 for ``Flexrelease``).
    """

    grip_acceleration: Optional[int] = None
    """
    Motion acceleration of the gripper in millimetres per second squared [100..4000],
    used by ``Flexgrip``/``Flexrelease``.

    ``None`` defers to the per-motion default
    :class:`~giskardpy.motion_statechart.ros2_nodes.griplink.GriplinkFlexActionServerTask`
    picks in ``build_msg`` (600 for ``Flexgrip``, 2000 for ``Flexrelease``).
    """

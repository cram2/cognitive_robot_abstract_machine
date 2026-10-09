from enum import Enum, auto


class JointStateType(Enum): ...


class GripperState(JointStateType):
    """
    The states a gripper's joint states are labelled with.

    A gripper declares a :class:`~semantic_digital_twin.datastructures.joint_state.JointState`
    for each state it can be commanded into; specifications and motions select them by
    these members.
    """

    OPEN = auto()
    """
    The gripper's fully opened state.
    """

    CLOSE = auto()
    """
    The gripper's fully closed state, gripping with full stroke.
    """

    MEDIUM = auto()
    """
    A gripper configuration between open and close.
    """

    FLEXOPEN = auto()
    """
    A partially opened state commanded by an opening width rather than the full stroke.
    """

    FLEXCLOSE = auto()
    """
    A partially closed state commanded by an opening width rather than the full stroke.
    """


class TorsoState(JointStateType):
    HIGH = auto()
    MID = auto()
    LOW = auto()


class StaticJointState(JointStateType):
    PARK = auto()

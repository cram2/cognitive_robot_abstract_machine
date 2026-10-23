from enum import Enum, StrEnum, auto


class JointStateType(Enum): ...


class GripperState(JointStateType):
    OPEN = auto()
    CLOSE = auto()
    MEDIUM = auto()


class TorsoState(JointStateType):
    HIGH = auto()
    MID = auto()
    LOW = auto()


class StaticJointState(JointStateType):
    PARK = auto()


class JointName(StrEnum):
    """
    The names a robot's description gives its joints.
    """

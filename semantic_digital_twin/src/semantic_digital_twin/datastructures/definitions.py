from enum import Enum, auto


class JointStateType(Enum):
    OPEN = auto()
    CLOSE = auto()
    MEDIUM = auto()
    HIGH = auto()
    MID = auto()
    LOW = auto()
    PARK = auto()


GripperState = JointStateType
TorsoState = JointStateType
StaticJointState = JointStateType


# class GripperState(Enum):
#     OPEN = auto()
#     CLOSE = auto()
#     MEDIUM = auto()
#
#
# class TorsoState(Enum):
#     HIGH = auto()
#     MID = auto()
#     LOW = auto()
#
#
# class StaticJointState(Enum):
#     PARK = auto()

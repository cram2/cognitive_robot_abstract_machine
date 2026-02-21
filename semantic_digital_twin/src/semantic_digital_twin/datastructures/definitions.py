from enum import Enum, auto


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

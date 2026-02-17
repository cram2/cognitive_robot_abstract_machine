"""Module holding all enums of PyCRAM."""

from enum import Enum, auto, IntEnum

from ..failures import UnsupportedJointType


class AdjacentBodyMethod(Enum):
    ClosestPoints = auto()
    """
    The ClosestPoints method is used to find the closest points in other bodies to the body.
    """
    RayCasting = auto()
    """
    The RayCasting method is used to find the points in other bodies that are intersected by rays cast
     from the body bounding box to 6 directions (up, down, left, right, front, back).
    """


class ContainerManipulationType(Enum):
    """
    Enum for the different types of container manipulation.
    """

    Opening = auto()
    """
    The Opening type is used to open a container.
    """
    Closing = auto()
    """
    The Closing type is used to close a container.
    """


class FindBodyInRegionMethod(Enum):
    """
    Enum for the different methods to find a body in a region.
    """

    FingerToCentroid = auto()
    """
    The FingerToCentroid method is used to find the body in a region by casting a ray from each finger to the
     centroid of the region.
    """
    Centroid = auto()
    """
    The Centroid method is used to find the body in a region by calculating the centroid of the region and
    casting two rays from opposite sides of the region to the centroid.
    """
    MultiRay = auto()
    """
    The MultiRay method is used to find the body in a region by casting multiple rays covering the region.
    """


class ExecutionType(Enum):
    """Enum for Execution Process Module types."""

    REAL = auto()
    SIMULATED = auto()
    SEMI_REAL = auto()
    NO_EXECUTION = auto()


class Arms(IntEnum):
    """Enum for Arms."""

    # LEFT = "left"
    # RIGHT = "right"
    # BOTH = "both"
    LEFT = 0
    RIGHT = 1
    BOTH = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class TaskStatus(int, Enum):
    """
    Enum for readable descriptions of a tasks' status.
    """

    CREATED = 0
    RUNNING = 1
    SUCCEEDED = 2
    FAILED = 3
    INTERRUPTED = 4
    SLEEPING = 5

class JointType(Enum):
    """
    Enum for readable joint types.
    """

    REVOLUTE = 0
    PRISMATIC = 1
    SPHERICAL = 2
    PLANAR = 3
    FIXED = 4
    UNKNOWN = 5
    CONTINUOUS = 6
    FLOATING = 7


class AxisIdentifier(Enum):
    """
    Enum for translating the axis name to a vector along that axis.
    """

    X = (1, 0, 0)
    Y = (0, 1, 0)
    Z = (0, 0, 1)
    Undefined = (0, 0, 0)

    @classmethod
    def from_tuple(cls, axis_tuple):
        return next((axis for axis in cls if axis.value == axis_tuple), None)


class Grasp:
    """
    Base class for grasp enums.
    """

    def __hash__(self):
        return [index for index, value in enumerate(self.__class__) if self == value][0]

    @classmethod
    def from_axis_direction(cls, axis: AxisIdentifier, direction: int):
        """Get the Grasp face from an axis-index tuple"""
        return next((grasp for grasp in cls if grasp.value == (axis, direction)), None)


class ApproachDirection(Grasp, Enum):
    """
    Enum for the approach direction of a gripper.
    The AxisIdentifier is used to identify the axis of the gripper, and the int is used to identify the direction along
     that axis.
    """

    FRONT = (AxisIdentifier.X, -1)
    BACK = (AxisIdentifier.X, 1)
    RIGHT = (AxisIdentifier.Y, -1)
    LEFT = (AxisIdentifier.Y, 1)

    @property
    def axis(self) -> AxisIdentifier:
        """
        Returns the axis of the approach direction.
        """
        return self.value[0]


class VerticalAlignment(Grasp, Enum):
    """
    Enum for the vertical alignment of a gripper.
    The AxisIdentifier is used to identify the axis of the gripper, and the int is used to identify the direction along
     that axis.
    """

    TOP = (AxisIdentifier.Z, -1)
    BOTTOM = (AxisIdentifier.Z, 1)
    NoAlignment = (AxisIdentifier.Undefined, 0)


class GripperType(Enum):
    """
    Enum for the different types of grippers.
    """

    PARALLEL = auto()
    SUCTION = auto()
    FINGER = auto()
    HYDRAULIC = auto()
    PNEUMATIC = auto()
    CUSTOM = auto()


class ImageEnum(Enum):
    """
    Enum for image switch view on hsrb display.
    """

    HI = 0
    TALK = 1
    DISH = 2
    DONE = 3
    DROP = 4
    HANDOVER = 5
    ORDER = 6
    PICKING = 7
    PLACING = 8
    REPEAT = 9
    SEARCH = 10
    WAVING = 11
    FOLLOWING = 12
    DRIVINGBACK = 13
    PUSHBUTTONS = 14
    FOLLOWSTOP = 15
    JREPEAT = 16
    SOFA = 17
    INSPECT = 18
    CHAIR = 37


class DetectionTechnique(int, Enum):
    """
    Enum for techniques for detection tasks.
    """

    ALL = 0
    HUMAN = 1
    TYPES = 2
    REGION = 3
    HUMAN_ATTRIBUTES = 4
    HUMAN_WAVING = 5


class DetectionState(int, Enum):
    """
    Enum for the state of the detection task.
    """

    START = 0
    STOP = 1
    PAUSE = 2


class MovementType(Enum):
    """
    Enum for the different movement types of the robot.
    """

    STRAIGHT_TRANSLATION = auto()
    STRAIGHT_CARTESIAN = auto()
    TRANSLATION = auto()
    CARTESIAN = auto()


class WaypointsMovementType(Enum):
    """
    Enum for the different movement types of the robot.
    """

    ENFORCE_ORIENTATION_STRICT = auto()
    ENFORCE_ORIENTATION_FINAL_POINT = auto()


class FilterConfig(Enum):
    """
    Declare existing filter methods.
    Currently supported: Butterworth
    """

    butterworth = 1


class MonitorBehavior(Enum):
    """
    Enum for the different monitor behaviors.
    """

    INTERRUPT = auto()
    """
    Interrupt the task when the condition is met.
    """
    PAUSE = auto()
    """
    Pause the task when the condition is met.
    """
    RESUME = auto()
    """
    Resume the task when the condition is met.
    """

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.exceptions import UsageError

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_part_mixins import RobotPartMixin

# %% assumptions a robot part makes about what is attached to it


@dataclass
class RobotPartValidationError(UsageError):
    """
    Raised when a robot part does not carry what one of its mixins assumes.
    """

    robot_part: RobotPartMixin
    """
    The part whose assumption does not hold.
    """

    def error_message(self) -> str:
        return f"Robot part: {self.robot_part} could not be validated"

    def suggest_correction(self) -> str:
        return ""


# %% a required child that is not attached


@dataclass
class MissingEndEffectorError(RobotPartValidationError):
    """
    Raised when a robot part has no end effector attached.
    """

    def error_message(self) -> str:
        return f"{type(self.robot_part).__name__} has no end effector."


@dataclass
class MissingMobileBaseError(RobotPartValidationError):
    """
    Raised when a robot part has no mobile base attached.
    """

    def error_message(self) -> str:
        return f"{type(self.robot_part).__name__} has no mobile base."


@dataclass
class MissingTorsoError(RobotPartValidationError):
    """
    Raised when a robot part has no torso attached.
    """

    def error_message(self) -> str:
        return f"{type(self.robot_part).__name__} has no torso."


@dataclass
class MissingNeckError(RobotPartValidationError):
    """
    Raised when a robot part has no neck attached.
    """

    def error_message(self) -> str:
        return f"{type(self.robot_part).__name__} has no neck."


@dataclass
class MissingLidarError(RobotPartValidationError):
    """
    Raised when a robot part has no lidar attached.
    """

    def error_message(self) -> str:
        return f"{type(self.robot_part).__name__} has no lidar."


@dataclass
class MissingSensorsError(RobotPartValidationError):
    """
    Raised when a robot part has no sensors attached.
    """

    def error_message(self) -> str:
        return f"{type(self.robot_part).__name__} has no sensors."


# %% a number of children the mixin does not allow


@dataclass
class TooFewFingersError(RobotPartValidationError):
    """
    Raised when a robot part has fewer fingers than its mixin allows.
    """

    minimum_count: int
    """
    How many fingers the mixin requires at least.
    """

    actual_count: int
    """
    How many fingers are attached.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.robot_part).__name__} has {self.actual_count} fingers, "
            f"but at least {self.minimum_count} are required."
        )

    def suggest_correction(self) -> str:
        return "use HasTwoFingers if this part is supposed to have exactly two fingers."


@dataclass
class UnexpectedFingerCountError(RobotPartValidationError):
    """
    Raised when a robot part has a different number of fingers than its mixin allows.
    """

    expected_count: int
    """
    How many fingers the mixin requires.
    """

    actual_count: int
    """
    How many fingers are attached.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.robot_part).__name__} has {self.actual_count} fingers, "
            f"but exactly {self.expected_count} are required."
        )


@dataclass
class TooFewArmsError(RobotPartValidationError):
    """
    Raised when a robot part has fewer arms than its mixin allows.
    """

    minimum_count: int
    """
    How many arms the mixin requires at least.
    """

    actual_count: int
    """
    How many arms are attached.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.robot_part).__name__} has {self.actual_count} arms, "
            f"but at least {self.minimum_count} are required."
        )

    def suggest_correction(self) -> str:
        return (
            "use HasOneArm if this part is supposed to have one arm, or HasLeftRightArm "
            "if it is supposed to have a left and a right one."
        )


@dataclass
class UnexpectedArmCountError(RobotPartValidationError):
    """
    Raised when a robot part has a different number of arms than its mixin allows.
    """

    expected_count: int
    """
    How many arms the mixin requires.
    """

    actual_count: int
    """
    How many arms are attached.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.robot_part).__name__} has {self.actual_count} arms, "
            f"but exactly {self.expected_count} are required."
        )

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import List, TYPE_CHECKING

from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_part_mixins import HasInputSource
    from semantic_digital_twin.robots.robot_parts import AbstractRobot

# %% base classes


@dataclass
class InputSynchronizer(ABC):
    """
    Writes an external source of truth, e.g. a robot's joint states, into the world
    state.
    """

    world: World
    """
    The world whose state is kept in sync with the external source.
    """

    @abstractmethod
    def apply(self) -> bool:
        """
        Write the most recent input into the world state.

        :return: Whether anything was written.
        """

    def rewriting_every_cycle(self) -> InputSynchronizer:
        """
        :return: An input that writes what it last read in every cycle, for a loop that
            moves the world state away from it between cycles.

        Defaults to this input, which already writes whatever it holds whenever it is
        applied.
        """
        return self

    def close(self) -> None:
        """
        Release the resources used to receive inputs.
        """


@dataclass
class WorldStateInputs:
    """
    All inputs that one loop reads before it computes anything.
    """

    world: World
    """
    The world whose state is written and whose observers are notified.
    """

    synchronizers: List[InputSynchronizer] = field(default_factory=list)
    """
    The inputs, applied in the order they were added.
    """

    reapplies_inputs: bool = False
    """
    Whether this loop needs its inputs written again in every cycle, as a loop does that
    moves the world state away from what it last read.
    """

    def read_robot(self, robot: AbstractRobot) -> None:
        """
        Apply everything the parts of a robot are read from in this loop from now on.

        Parts that are read from the world they stand in need nothing applied and are
        therefore left out.

        :param robot: The robot whose parts this loop reads.
        """
        for synchronizer in robot.get_input_synchronizers():
            self.read(synchronizer)

    def read_robot_part(self, robot_part: HasInputSource) -> None:
        """
        Apply what one part of a robot is read from in this loop from now on.

        A part that is read from the world it stands in needs nothing applied and is
        therefore left out.

        :param robot_part: The part this loop reads.
        """
        if not isinstance(robot_part.source, InputSynchronizer):
            return
        self.read(robot_part.source)

    def read(self, synchronizer: InputSynchronizer) -> None:
        """
        Apply the given input in this loop from now on, in the way this loop needs it.

        :param synchronizer: The input this loop reads.
        """
        if self.reapplies_inputs:
            synchronizer = synchronizer.rewriting_every_cycle()
        self.synchronizers.append(synchronizer)

    def apply_inputs(self) -> bool:
        """
        Write all inputs into the world state, in the order they were added.

        :return: Whether any of them wrote something.
        """
        wrote_something = False
        for synchronizer in self.synchronizers:
            wrote_something |= synchronizer.apply()
        return wrote_something

    def synchronize(self) -> None:
        """
        Write all inputs into the world state and announce the change.

        Nothing is announced when no input wrote, because announcing recomputes the
        forward kinematics and reaches every observer of the world.
        """
        if not self.apply_inputs():
            return
        self.announce_state()

    def synchronize_and_announce(self) -> None:
        """
        Write all inputs into the world state and announce the state even if no input
        wrote.

        Use this where nothing else announces, so that the observers of the world do not
        go stale.
        """
        self.apply_inputs()
        self.announce_state()

    def announce_state(self) -> None:
        """
        Hand the current world state to the observers of the world.

        Nothing is announced while the world model is being modified, because the
        observers would see an inconsistent model.
        """
        if self.world.world_is_being_modified:
            return
        self.world.notify_state_change()

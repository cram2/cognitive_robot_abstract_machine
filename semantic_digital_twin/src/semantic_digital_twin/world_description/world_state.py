from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, Iterator
from uuid import UUID

from typing_extensions import MutableMapping, List, Dict, Self, TYPE_CHECKING

import numpy as np

from krrood.symbolic_math.symbolic_math import FloatVariable
from .degree_of_freedom import DegreeOfFreedom
from ..callbacks.callback import StateChangeCallback
from ..datastructures.prefixed_name import PrefixedName
from ..exceptions import (
    DofNotInWorldStateError,
    IncorrectWorldStateValueShapeError,
    MismatchingCommandLengthError,
)
from ..spatial_types.derivatives import Derivatives

if TYPE_CHECKING:
    from ..world import World


class WorldStateView:
    """
    Returned if you access members in WorldState.
    Provides a more convenient interface to the data of a single DOF.
    """

    def __init__(self, data: np.ndarray):
        self.data = data

    def __getitem__(self, item: Derivatives) -> float:
        return self.data[item]

    def __setitem__(self, key: Derivatives, value: float) -> None:
        self.data[key] = value

    @property
    def position(self) -> float:
        return self.data[Derivatives.position]

    @position.setter
    def position(self, value: float):
        self.data[Derivatives.position] = value

    @property
    def velocity(self) -> float:
        return self.data[Derivatives.velocity]

    @velocity.setter
    def velocity(self, value: float):
        self.data[Derivatives.velocity] = value

    @property
    def acceleration(self) -> float:
        return self.data[Derivatives.acceleration]

    @acceleration.setter
    def acceleration(self, value: float):
        self.data[Derivatives.acceleration] = value

    @property
    def jerk(self) -> float:
        return self.data[Derivatives.jerk]

    @jerk.setter
    def jerk(self, value: float):
        self.data[Derivatives.jerk] = value


@dataclass
class WorldState(MutableMapping):
    """
    Tracks the state of all DOF in the world.
    Data is stored in a 4xN numpy array, such that it can be used as input for compiled functions without copying.

    This class adds a few convenience methods for manipulating this data.
    """

    _world: World = field(default=None)

    # 4 rows (pos, vel, acc, jerk), columns are joints
    data: np.ndarray = field(default_factory=lambda: np.zeros((4, 0), dtype=float))

    # list of dof ids in column order
    _ids: List[UUID] = field(default_factory=list)

    # maps dof ids -> column index
    _index: Dict[UUID, int] = field(default_factory=dict)

    version: int = 0
    """
    The version of the state. This increases whenever a change to the state of the kinematic model is made. 
    Mostly triggered by updating connection values.
    """

    state_change_callbacks: List[StateChangeCallback] = field(
        default_factory=list, repr=False
    )
    """
    Callbacks to be called when the state of the world changes.
    """

    def _notify_state_change(self) -> None:
        """
        If you have changed the state of the world, call this function to trigger necessary events and increase
        the state version.
        """
        self.version += 1
        for callback in self.state_change_callbacks:
            callback.notify()

    def _add_dof(self, uuid: UUID) -> None:
        idx = len(self._ids)
        self._ids.append(uuid)
        self._index[uuid] = idx
        # append a zero column
        new_col = np.zeros((4, 1), dtype=float)
        if self.data.shape[1] == 0:
            self.data = new_col
        else:
            self.data = np.hstack((self.data, new_col))

    def __getitem__(self, dof_id: UUID) -> WorldStateView:
        if dof_id not in self._index:
            raise DofNotInWorldStateError(dof_id)
        idx = self._index[dof_id]
        return WorldStateView(self.data[:, idx])

    def __setitem__(self, dof_id: UUID, value: np.ndarray | WorldStateView) -> None:
        if dof_id not in self._index:
            raise DofNotInWorldStateError(dof_id)
        if isinstance(value, WorldStateView):
            value = value.data
        arr = np.asarray(value, dtype=float)
        if arr.shape != (4,):
            raise IncorrectWorldStateValueShapeError(dof_id)
        idx = self._index[dof_id]
        self.data[:, idx] = arr

    def __delitem__(self, dof_id: UUID) -> None:
        if dof_id not in self._index:
            raise DofNotInWorldStateError(dof_id)
        idx = self._index.pop(dof_id)
        self._ids.pop(idx)
        # remove column from data
        self.data = np.delete(self.data, idx, axis=1)
        # rebuild indices
        for i, nm in enumerate(self._ids):
            self._index[nm] = i

    def __iter__(self) -> Iterator[UUID]:
        return iter(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __eq__(self, other: Self) -> bool:
        if self is other:
            return True

        if len(self) != len(other):
            return False

        if set(self._ids) != set(other._ids):
            return False

        return np.allclose(
            self.data,
            other.data,
            rtol=1e-8,
            atol=1e-12,
            equal_nan=True,
        )

    def keys(self) -> List[UUID]:
        return self._ids

    def items(self) -> List[tuple[UUID, np.ndarray]]:
        return [
            (dof_id, self.data[:, self._index[dof_id]].copy()) for dof_id in self._ids
        ]

    def values(self) -> List[np.ndarray]:
        return [self.data[:, self._index[dof_id]].copy() for dof_id in self._ids]

    def __contains__(self, dof_or_uuid: Union[DegreeOfFreedom, UUID]) -> bool:
        dof_id = (
            dof_or_uuid.id if isinstance(dof_or_uuid, DegreeOfFreedom) else dof_or_uuid
        )
        return dof_id in self._index

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({{ "
            + ", ".join(
                f"{n}: {list(self.data[:, i])}" for i, n in enumerate(self._ids)
            )
            + " })"
        )

    def to_position_dict(self) -> Dict[PrefixedName, float]:
        return {
            self._world.get_degree_of_freedom_by_id(dof_id).name: self[dof_id].position
            for dof_id in self._ids
        }

    @property
    def positions(self) -> np.ndarray:
        return self.data[0, :]

    @property
    def velocities(self) -> np.ndarray:
        return self.data[1, :]

    @property
    def accelerations(self) -> np.ndarray:
        return self.data[2, :]

    @property
    def jerks(self) -> np.ndarray:
        return self.data[3, :]

    def get_derivative(self, derivative: Derivatives) -> np.ndarray:
        """
        Retrieve the data for a whole derivative row.
        """
        return self.data[derivative, :]

    def set_derivative(self, derivative: Derivatives, new_state: np.ndarray):
        """
        Overwrite the data for a whole derivative row.
        Assums that the order of the DOFs is consistent.
        """
        self.data[derivative, :] = new_state

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the WorldState.
        """
        new_state = WorldState(_world=self._world)
        new_state.data = self.data.copy()
        new_state._ids = self._ids.copy()
        new_state._index = self._index.copy()
        return new_state

    def add_degree_of_freedom(self, dof: DegreeOfFreedom):
        """
        Adds a degree of freedom to the world state, initializing its position to 0 or the nearest limit.
        """
        dof.create_variables()

        lower = dof.lower_limits.position
        upper = dof.upper_limits.position
        initial_position = 0

        if lower is not None:
            initial_position = max(lower, initial_position)
        if upper is not None:
            initial_position = min(upper, initial_position)

        self._add_dof(dof.id)
        self[dof.id].position = initial_position

    def get_variables(self) -> List[FloatVariable]:
        """
        Constructs and returns a list of variables representing the state of the system. The state
        is defined in terms of positions, velocities, accelerations, and jerks for each degree
        of freedom specified in the current state.

        :raises KeyError: If a degree of freedom defined in the state does not exist in
            the `degrees_of_freedom`.
        :returns: A combined list of variables corresponding to the positions, velocities,
            accelerations, and jerks for each degree of freedom in the state.
        """
        positions = [
            self._world.get_degree_of_freedom_by_id(v_id).variables.position
            for v_id in self
        ]
        velocities = [
            self._world.get_degree_of_freedom_by_id(v_id).variables.velocity
            for v_id in self
        ]
        accelerations = [
            self._world.get_degree_of_freedom_by_id(v_id).variables.acceleration
            for v_id in self
        ]
        jerks = [
            self._world.get_degree_of_freedom_by_id(v_id).variables.jerk
            for v_id in self
        ]
        return positions + velocities + accelerations + jerks

    def _apply_control_commands(
        self, commands: np.ndarray, dt: float, derivative: Derivatives
    ):
        """
        Apply control commands to the specified derivative level, and integrate down to lower derivatives.

        :param commands: Control commands to be applied at the specified derivative
            level. The array length must match the number of free variables
            in the system.
        :param dt: Time step used for the integration of lower derivatives.
        :param derivative: The derivative level to which the control commands are
            applied.
        :return:
        """
        if len(commands) != len(self._ids):
            raise MismatchingCommandLengthError(
                expected_length=len(self._ids),
                actual_length=len(commands),
            )

        self.set_derivative(derivative, commands)

        for i in range(derivative - 1, -1, -1):
            self.set_derivative(
                i,
                self.get_derivative(i) + self.get_derivative(i + 1) * dt,
            )

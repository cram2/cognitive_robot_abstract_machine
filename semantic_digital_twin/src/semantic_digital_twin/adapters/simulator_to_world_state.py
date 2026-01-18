from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Dict, List, Optional
import numpy as np


class StateFeedbackAlreadyRunningError(Exception):
    """
    Raised when attempting to start a running state feedback loop.
    """


class AttributeSubscriptionError(Exception):
    """
    Raised when a simulator read subscription cannot be established for a connection.
    """


class AttributeReadError(Exception):
    """
    Raised when the simulator does not return the expected attributes for a subscribed entity.
    """


class ConnectionNotFoundError(Exception):
    """
    Raised when a world connection cannot be found for an entity name returned by the simulator.
    """


@dataclass
class JointReadSpecBuilder:
    """
    Build simulator read specifications for single-DoF connections.
    """

    def for_connection(self, connection) -> Dict[str, List[float]]:
        name = connection.__class__.__name__
        if name == "RevoluteConnection":
            return {"joint_angular_position": [0.0]}
        if name == "PrismaticConnection":
            return {"joint_linear_position": [0.0]}
        raise AttributeSubscriptionError(
            f"Unsupported connection type for read spec: {name}"
        )


@dataclass
class SimulatorToWorldStateSynchronizer:
    """
    Mirror simulator joint state into the semantic digital twin world.

    Provides on-demand synchronization and a low-frequency background loop.
    """

    # Late imports to avoid heavy modules at import time
    world: "World"
    sim: "MultiSim"
    poll_period_s: float = 0.05

    read_spec_builder: JointReadSpecBuilder = field(
        default_factory=JointReadSpecBuilder
    )

    _running: bool = field(default=False, init=False, repr=False)
    _stop_event: Event = field(default_factory=Event, init=False, repr=False)
    _thread: Optional[Thread] = field(default=None, init=False, repr=False)

    def initialize_subscriptions(self) -> None:
        """
        Subscribe to all active one-DoF connections. Fail if none are subscribable.
        """
        connections = [
            c
            for c in self.world.connections
            if hasattr(c, "active_dofs") and len(c.active_dofs) == 1
        ]
        read_objects: Dict[str, Dict[str, List[float]]] = {}
        for c in connections:
            read_objects[str(c.name)] = self.read_spec_builder.for_connection(c)
        if not read_objects:
            raise AttributeSubscriptionError(
                "No valid one-DoF connections to subscribe."
            )
        self.sim.set_read_objects(read_objects)

    def synchronize_once(self) -> None:
        """
        Pull latest simulator values and write them into the world, then notify a state change.
        """
        read = self.sim.get_read_objects()
        if not read:
            raise AttributeReadError(
                "Simulator returned no read objects. Did you call initialize_subscriptions()?"
            )

        updated_any = False
        for entity_name, attrs in read.items():
            if "joint_angular_position" in attrs:
                raw = attrs["joint_angular_position"]
            elif "joint_linear_position" in attrs:
                raw = attrs["joint_linear_position"]
            else:
                raise AttributeReadError(
                    f"No supported joint attribute in simulator read for '{entity_name}'."
                )

            # Extract values from either MultiverseAttribute, list, tuple, or numpy array
            if hasattr(raw, "values"):
                val_attr = getattr(raw, "values")
                values = list(val_attr() if callable(val_attr) else val_attr)
            elif isinstance(raw, (list, tuple, np.ndarray)):
                values = list(raw)
            else:
                raise AttributeReadError(
                    f"Unsupported attribute data type for '{entity_name}': {type(raw).__name__}"
                )

            if not values:
                raise AttributeReadError(
                    f"Empty attribute values returned for '{entity_name}'."
                )

            connection = self.world.get_connection_by_name(entity_name)
            if connection is None:
                raise ConnectionNotFoundError(
                    f"Connection '{entity_name}' not found in world."
                )

            arr = np.asarray(values).ravel()
            connection.position = float(arr[0])
            updated_any = True

        if updated_any:
            self.world.notify_state_change()

    def start(self) -> None:
        """
        Start a background loop that periodically synchronizes state.
        """
        if self._running:
            raise StateFeedbackAlreadyRunningError("Feedback loop already running.")
        self._stop_event.clear()
        self.initialize_subscriptions()
        self._thread = Thread(target=self._run, name="SimToWorldState", daemon=True)
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        """
        Stop the background loop.
        """
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._running = False

    def _run(self) -> None:
        """
        Background execution loop for periodic synchronization.
        """
        while not self._stop_event.is_set():
            self.synchronize_once()
            self._stop_event.wait(self.poll_period_s)

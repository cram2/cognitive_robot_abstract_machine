from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UnMonitoredContainerTypeForDescriptor(Exception):
    """
    Raised when a descriptor is used on a field with a container type that is not monitored (i.e., is not a subclass of
    MonitoredContainer). This happens when your type hint of the field is using a container type that is not supported.
    """

    clazz: type
    field_name: str
    container_type: type

    def __post_init__(self):
        super().__init__(
            f"Unmonitored container type '{self.container_type.__name__}' used for field '{self.field_name}' "
            f"in class '{self.clazz.__name__}'."
        )

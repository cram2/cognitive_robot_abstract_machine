"""The serializable description of a liquid transfer from a source into a
receiver."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

# %% transfer settings


@dataclass(frozen=True, kw_only=True)
class LiquidTransferSettings:
    """The fallback stream speed of a liquid transfer and the steepness of its
    gates."""

    exit_speed: float = 0.2
    """Fallback speed of the stream leaving the source, in metres per second,
    used where the source models no exit speed of its own."""

    height_gate_sharpness: float = 80.0
    """Logistic steepness of the gate closing as the source lip sinks to the
    receiver's opening, per receiver height of clearance."""

    overlap_gate_sharpness: float = 80.0
    """Logistic steepness of the gate closing as the stream lands away from the
    receiver's opening, per opening radius of distance."""


# %% transfer coupling


@dataclass(kw_only=True)
class LiquidTransferCoupling:
    """The parameters a receiver's symbolic inflow coupling is rebuilt from in
    any world, since the symbolic expressions themselves cannot leave the world
    they were built in."""

    source_id: UUID
    """Id of the source semantic annotation whose gated drain feeds the
    receiver."""

    settings: LiquidTransferSettings = field(default_factory=LiquidTransferSettings)
    """The fallback stream speed and gate steepness of the transfer."""

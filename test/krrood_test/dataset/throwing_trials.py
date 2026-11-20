"""
Trials of throwing a ball into a basket, where only a steep enough angle lands the ball,
to try parameter spaces, campaigns and analyses on without a simulation.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from typing_extensions import Iterator

from krrood.adapters.json_serializer import from_json, to_json
from krrood.bounds import Bounds
from krrood.entity_query_language.factories import ConditionType
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.model_registries import UniformPriorRegistry
from krrood.parametrization.trials.campaigns import (
    TrialPerformer,
    TrialSetup,
    TrialStore,
)
from krrood.parametrization.trials.parameter_spaces import ParameterSpace
from krrood.parametrization.trials.records import (
    Trial,
    TrialMeasurements,
    TrialParameters,
)

# %% the parameters of a throw


@dataclass(kw_only=True)
class ReleasePoint:
    """
    Where the hand lets go of the ball.
    """

    height: float
    """
    How high above the floor, in metres.
    """


@dataclass(kw_only=True)
class ThrowParameters(TrialParameters):
    """
    Everything a throw varies by.
    """

    angle: float
    """
    The angle above the horizon the ball leaves the hand at, in radians.
    """

    speed: float
    """
    How fast the ball leaves the hand, in m/s.
    """

    release: ReleasePoint
    """
    Where the hand lets go.
    """


@dataclass(kw_only=True)
class NamedThrowParameters(TrialParameters):
    """
    Parameters with a field that is not a number.
    """

    angle: float
    """
    The angle above the horizon the ball leaves the hand at, in radians.
    """

    thrower: str
    """
    Who throws.
    """


@dataclass(kw_only=True)
class SpeedOnlyParameters(TrialParameters):
    """
    Parameters with fewer fields than a throw has.
    """

    speed: float
    """
    How fast the ball leaves the hand, in m/s.
    """


# %% the values a throw may take


@dataclass(kw_only=True)
class ThrowSpace(ParameterSpace[ThrowParameters]):
    """
    The throws a person can make.
    """

    angle: Bounds[float] = field(default_factory=lambda: Bounds(lower=0.0, upper=1.5))
    """
    The values of :attr:`ThrowParameters.angle`.
    """

    speed: Bounds[float] = field(default_factory=lambda: Bounds(lower=1.0, upper=5.0))
    """
    The values of :attr:`ThrowParameters.speed`.
    """

    height: Bounds[float] = field(default_factory=lambda: Bounds(lower=1.0, upper=2.0))
    """
    The values of :attr:`ReleasePoint.height`.
    """

    def conditions(self, *, parameters: Match[ThrowParameters]) -> list[ConditionType]:
        return [
            *self.conditions_within(value=parameters.angle, bounds=self.angle),
            *self.conditions_within(value=parameters.speed, bounds=self.speed),
            *self.conditions_within(
                value=parameters.release.height, bounds=self.height
            ),
        ]


@dataclass(kw_only=True)
class FastThrowSpace(ThrowSpace):
    """
    The throws a person can make, of which only the fast half is tried.
    """

    def accepts(self, *, parameters: ThrowParameters) -> bool:
        """
        Unlike :meth:`ParameterSpace.accepts`, only throws of at least 3 m/s are tried.
        """
        return parameters.speed >= 3.0


# %% throwing


@dataclass(kw_only=True)
class ThrowMeasurements(TrialMeasurements):
    """
    What is measured while the ball is thrown.
    """

    reached_angle: float
    """
    The angle the ball actually left the hand at, in radians.
    """


@dataclass(kw_only=True)
class Throw(Trial[ThrowParameters, ThrowMeasurements]):
    """
    One throw, with whether the ball landed in the basket.
    """


@dataclass(kw_only=True)
class SteepThrowsLand(TrialPerformer[ThrowParameters, Throw]):
    """
    Throws that land the ball exactly when the angle the ball leaves the hand at is
    steep enough; wind turns that angle away from the one the hand aims at.
    """

    steepness: Bounds[float] = field(
        default_factory=lambda: Bounds(lower=0.6, upper=0.9)
    )
    """
    The angles that land the ball.
    """

    wind: float = 0.0
    """
    The standard deviation of the angle the wind turns the ball by, in radians.
    """

    def perform(self, *, parameters: ThrowParameters) -> Throw:
        reached_angle = float(parameters.angle + np.random.normal(scale=self.wind))
        return Throw(
            parameters=parameters,
            measurements=ThrowMeasurements(reached_angle=reached_angle),
            succeeded=self.steepness.lower <= reached_angle < self.steepness.upper,
        )


@dataclass(kw_only=True)
class ThrowsInAFile(TrialStore[Throw]):
    """
    Keeps every throw as a line of JSON at the end of a file, which several processes
    can share.
    """

    path: Path
    """
    The file.
    """

    def add(self, *, trial: Throw) -> None:
        with self.path.open("a") as file:
            file.write(json.dumps(to_json(trial)) + "\n")

    def throws(self) -> list[Throw]:
        """
        :return: The throws kept so far, in the order they were kept.
        """
        if not self.path.exists():
            return []
        return [
            from_json(json.loads(line)) for line in self.path.read_text().splitlines()
        ]


@dataclass(kw_only=True)
class ThrowingSetup(TrialSetup[ThrowParameters, Throw]):
    """
    Steep throws landing, drawn uniformly from the throws a person can make and kept in
    a file.
    """

    path: Path
    """
    The file the throws are kept in.
    """

    parameter_space: ThrowSpace = field(default_factory=ThrowSpace)
    """
    What the throws are drawn from.
    """

    model_registry: UniformPriorRegistry = field(default_factory=UniformPriorRegistry)
    """
    Draws the throws uniformly.
    """

    wind: float = 0.0
    """
    The standard deviation of the angle the wind turns each throw by, in radians.
    """

    @contextmanager
    def performing(self) -> Iterator[SteepThrowsLand]:
        yield SteepThrowsLand(wind=self.wind)

    def create_store(self) -> ThrowsInAFile:
        return ThrowsInAFile(path=self.path)

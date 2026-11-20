"""
Trials of throwing a ball into a basket, where only a steep enough angle lands the ball,
to try parameter spaces, campaigns and analyses on without a simulation.
"""

from __future__ import annotations

import dataclasses
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Iterator

from krrood.bounds import Bounds
from krrood.entity_query_language.factories import ConditionType
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.model_registries import UniformPriorRegistry
from krrood.parametrization.trials import (
    ParameterSpace,
    TrialParameters,
    TrialPerformer,
    TrialSetup,
    TrialStore,
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
        return parameters.speed >= 3.0


# %% throwing


@dataclass(kw_only=True)
class Throw:
    """
    One throw, with what came of it.
    """

    parameters: ThrowParameters
    """
    How the ball was thrown.
    """

    landed_in_the_basket: bool
    """
    Whether the ball landed in the basket.
    """


@dataclass(kw_only=True)
class SteepThrowsLand(TrialPerformer[ThrowParameters, Throw]):
    """
    Throws that land the ball exactly when the angle is steep enough.
    """

    steepness: Bounds[float] = field(
        default_factory=lambda: Bounds(lower=0.6, upper=0.9)
    )
    """
    The angles that land the ball.
    """

    def perform(self, *, parameters: ThrowParameters) -> Throw:
        return Throw(
            parameters=parameters,
            landed_in_the_basket=self.steepness.lower
            <= parameters.angle
            < self.steepness.upper,
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
            file.write(json.dumps(dataclasses.asdict(trial)) + "\n")

    def throws(self) -> list[Throw]:
        """
        :return: The throws kept so far, in the order they were kept.
        """
        if not self.path.exists():
            return []
        return [
            Throw(
                parameters=ThrowParameters(
                    angle=stored["parameters"]["angle"],
                    speed=stored["parameters"]["speed"],
                    release=ReleasePoint(**stored["parameters"]["release"]),
                ),
                landed_in_the_basket=stored["landed_in_the_basket"],
            )
            for stored in map(json.loads, self.path.read_text().splitlines())
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

    @contextmanager
    def performing(self) -> Iterator[SteepThrowsLand]:
        yield SteepThrowsLand()

    def create_store(self) -> ThrowsInAFile:
        return ThrowsInAFile(path=self.path)

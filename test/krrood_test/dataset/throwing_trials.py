"""
Trials of throwing a ball into a basket, where only a steep enough angle lands the ball,
to try parameter spaces, campaigns and analyses on without a simulation.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Iterator

from krrood.bounds import Bounds
from krrood.entity_query_language.factories import ConditionType
from krrood.entity_query_language.query.match import Match
from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.patterns.classproperty import classproperty
from krrood.parametrization.trials.campaigns import TrialPerformer, TrialSetup
from krrood.parametrization.trials.database import DatabaseTrialStore
from krrood.parametrization.trials.parameter_spaces import ParameterSpace
from krrood.parametrization.trials.starting_states import (
    StartingStateRecipe,
    StateReacher,
)
from krrood.parametrization.trials.records import (
    InformedField,
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
class ThrowBall:
    """
    The action of throwing a ball, whose defaults the throwing trials inform.
    """

    aim_angle: float = 0.3
    """
    The angle above the horizon to aim at, in radians.
    """

    release_height: float = 1.0
    """
    How high above the floor to let go of the ball, in metres.
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

    @classproperty
    def informed_fields(cls) -> list[InformedField]:
        """
        Unlike :attr:`TrialParameters.informed_fields`, the aimed angle and the release
        height inform the defaults of :class:`ThrowBall`.
        """
        return [
            InformedField(
                attribute_path="angle", owner=ThrowBall, field_name="aim_angle"
            ),
            InformedField(
                attribute_path="release.height",
                owner=ThrowBall,
                field_name="release_height",
            ),
        ]


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


@dataclass(kw_only=True)
class Court:
    """
    Where throws are made from, as the state every throw starts from.
    """

    basket_distance: float
    """
    How far the basket stands from the thrower, in metres.
    """


@dataclass(kw_only=True)
class CourtBuilder(StateReacher[Court]):
    """
    Sets the basket up at a distance that varies around the one aimed for.
    """

    aimed_distance: float
    """
    The distance the basket is meant to stand at, in metres.
    """

    placement_spread: float
    """
    The standard deviation of the distance the basket ends up at, in metres.
    """

    def reach(self) -> Court:
        return Court(
            basket_distance=float(
                np.random.normal(loc=self.aimed_distance, scale=self.placement_spread)
            )
        )


@dataclass(kw_only=True)
class CourtRecipe(StartingStateRecipe[Court]):
    """
    A court whose basket stands within given distances from the thrower.
    """

    wanted_distance: Bounds[float] = field(
        default_factory=lambda: Bounds(lower=1.9, upper=2.1)
    )
    """
    The distances of a court wanted for the throws, in metres.
    """

    placement_spread: float = 0.2
    """
    The standard deviation of the distance the basket ends up at, in metres.
    """

    @property
    def data_access_object_class(self) -> type[DataAccessObject]:
        from .ormatic_interface import CourtDAO

        return CourtDAO

    @contextmanager
    def reaching(self) -> Iterator[CourtBuilder]:
        yield CourtBuilder(
            aimed_distance=(self.wanted_distance.lower + self.wanted_distance.upper)
            / 2,
            placement_spread=self.placement_spread,
        )

    def is_wanted(self, *, state: Court) -> bool:
        return (
            self.wanted_distance.lower
            <= state.basket_distance
            < self.wanted_distance.upper
        )


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

    seconds_per_throw: float = 0.0
    """
    How long a throw takes, in seconds.
    """

    def perform(self, *, parameters: ThrowParameters) -> Throw:
        time.sleep(self.seconds_per_throw)
        reached_angle = float(parameters.angle + np.random.normal(scale=self.wind))
        return Throw(
            parameters=parameters,
            measurements=ThrowMeasurements(reached_angle=reached_angle),
            succeeded=self.steepness.lower <= reached_angle < self.steepness.upper,
        )


@dataclass(kw_only=True)
class ThrowsInADatabase(DatabaseTrialStore[Throw]):
    """
    Keeps throws in a database of the ORM interface of the krrood tests.
    """

    @property
    def data_access_object_class(self) -> type[DataAccessObject]:
        from .ormatic_interface import ThrowDAO

        return ThrowDAO


@dataclass(kw_only=True)
class ThrowingSetup(TrialSetup[ThrowParameters, Throw]):
    """
    Steep throws landing, kept in a database.
    """

    database_url: str
    """
    Where the throws are kept.
    """

    starting_state_id: int | None = None
    """
    The database id of the court the throws are made on.
    """

    performer: SteepThrowsLand = field(default_factory=SteepThrowsLand)
    """
    Throws the ball.
    """

    @contextmanager
    def performing(self) -> Iterator[SteepThrowsLand]:
        yield self.performer

    def create_store(self) -> ThrowsInADatabase:
        return ThrowsInADatabase(
            database_url=self.database_url, starting_state_id=self.starting_state_id
        )

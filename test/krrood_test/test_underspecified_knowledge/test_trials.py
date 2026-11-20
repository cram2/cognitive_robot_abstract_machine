"""
Tests of finding out by trials what makes a behaviour succeed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine

from krrood.parametrization.model_registries import UniformPriorRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from krrood.parametrization.trials.analysis import TrialAnalysis
from krrood.parametrization.trials.campaigns import TrialCampaign
from krrood.bounds import Bounds
from krrood.parametrization.trials.exceptions import (
    FieldOfUnsupportedType,
    NotEnoughMemoryForWorkers,
    NoTrials,
    OutcomeDidNotVaryInPilot,
    ParameterWithoutCentre,
)
from krrood.parametrization.trials.parameter_spaces import (
    DrawnParameters,
    NearCentresRegistry,
)
from ..dataset.throwing_trials import (
    Court,
    FastThrowSpace,
    NamedThrowParameters,
    ReleasePoint,
    SpeedOnlyParameters,
    SteepThrowsLand,
    Throw,
    ThrowingSetup,
    ThrowMeasurements,
    ThrowParameters,
    ThrowSpace,
    ThrowsInADatabase,
)

# %% the parameters of a trial


def test_the_values_of_parameters_are_keyed_by_their_attribute_paths():
    """
    A number in a nested dataclass is keyed by the path of attributes leading to it.
    """
    parameters = ThrowParameters(angle=0.7, speed=3.0, release=ReleasePoint(height=1.5))
    assert parameters.values_by_attribute_path() == {
        "angle": 0.7,
        "speed": 3.0,
        "release.height": 1.5,
    }


def test_parameters_with_a_field_that_is_no_number_have_no_values():
    """
    A field that holds text cannot be drawn, so it has no value as a parameter.
    """
    with pytest.raises(FieldOfUnsupportedType):
        NamedThrowParameters(angle=0.7, thrower="Alice").values_by_attribute_path()


def test_the_values_of_a_trial_are_keyed_by_their_paths_from_the_trial():
    """
    The parameters and the measurements of a trial are keyed by their paths from the
    trial, so a parameter and a measurement of the same name stay apart.
    """
    parameters = ThrowParameters(angle=0.7, speed=3.0, release=ReleasePoint(height=1.5))
    measurements = ThrowMeasurements(reached_angle=0.65)
    throw = Throw(parameters=parameters, measurements=measurements, succeeded=True)
    assert throw.values_by_attribute_path() == {
        **{
            f"parameters.{path}": value
            for path, value in parameters.values_by_attribute_path().items()
        },
        "measurements.reached_angle": measurements.reached_angle,
    }


# %% drawing parameters


@pytest.fixture(scope="module")
def drawn() -> list[ThrowParameters]:
    """
    Throws drawn uniformly from the throws a person can make.
    """
    np.random.seed(0)
    return ThrowSpace().drawn_parameters(
        number_of_draws=30, model_registry=UniformPriorRegistry()
    )


def test_as_many_parameters_are_drawn_as_asked_for(drawn):
    """
    Every draw is tried, since the space accepts every draw by default.
    """
    assert len(drawn) == 30


def test_every_drawn_parameter_lies_within_its_bounds(drawn):
    """
    Every parameter, the nested one included, lies between the bounds of the space.
    """
    space = ThrowSpace()
    for parameters in drawn:
        for value, bounds in [
            (parameters.angle, space.angle),
            (parameters.speed, space.speed),
            (parameters.release.height, space.height),
        ]:
            assert bounds.lower <= value < bounds.upper


def test_the_statement_has_a_variable_for_every_number_of_the_parameters():
    """
    Every number of the parameters, the nested one included, is left to be drawn.
    """
    statement = ThrowSpace().statement()
    names = UnderspecifiedParameters(statement=statement).variables
    assert {name.removeprefix("ThrowParameters.") for name in names} == set(
        ThrowParameters(
            angle=0.0, speed=0.0, release=ReleasePoint(height=0.0)
        ).values_by_attribute_path()
    )


def test_draws_the_space_does_not_accept_are_left_out():
    """
    A space that accepts only fast throws returns only fast throws.
    """
    np.random.seed(1)
    drawn = FastThrowSpace().drawn_parameters(
        number_of_draws=30, model_registry=UniformPriorRegistry()
    )
    assert 0 < len(drawn) < 30
    assert all(parameters.speed >= 3.0 for parameters in drawn)


# %% drawing near parameters that worked


def test_parameters_drawn_near_a_centre_lie_near_it():
    """
    With a small spread, every draw lies within five standard deviations of the centre
    along every parameter.
    """
    space = ThrowSpace()
    centre = ThrowParameters(angle=0.75, speed=2.0, release=ReleasePoint(height=1.4))
    registry = NearCentresRegistry(centres=[centre], spread=0.01)
    np.random.seed(2)
    drawn = space.drawn_parameters(number_of_draws=20, model_registry=registry)
    ranges = {
        "angle": space.angle.upper - space.angle.lower,
        "speed": space.speed.upper - space.speed.lower,
        "release.height": space.height.upper - space.height.lower,
    }
    expected = centre.values_by_attribute_path()

    assert len(drawn) == 20
    for parameters in drawn:
        for path, value in parameters.values_by_attribute_path().items():
            assert abs(value - expected[path]) < 5 * registry.spread * ranges[path]


def test_a_centre_without_a_value_for_a_parameter_is_refused():
    """
    A centre that gives no value for one of the statement's parameters cannot be drawn
    near.
    """
    registry = NearCentresRegistry(centres=[SpeedOnlyParameters(speed=2.0)])
    with pytest.raises(ParameterWithoutCentre):
        ThrowSpace().drawn_parameters(number_of_draws=1, model_registry=registry)


# %% keeping trials in a database


@pytest.fixture
def database_path(tmp_path) -> Path:
    """
    The file of a database of the test ORM interface, which several processes can share.
    """
    return tmp_path / "throws.sqlite"


@pytest.fixture
def database_url(database_path) -> str:
    """
    Where the database of :func:`database_path` is, as SQLAlchemy names it.
    """
    return f"sqlite:///{database_path}"


def stored_court_id(*, database_url: str, basket_distance: float) -> int:
    """
    :param database_url: Where to store the court.
    :param basket_distance: How far the basket stands from the thrower.
    :return: The database id of a court stored as a starting state.
    """
    from ..dataset.ormatic_interface import Base

    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        court = to_dao(Court(basket_distance=basket_distance))
        session.add(court)
        session.commit()
        return court.database_id


@pytest.fixture
def court_id(database_url) -> int:
    """
    The database id of the court the throws of a test start from.
    """
    return stored_court_id(database_url=database_url, basket_distance=2.0)


@pytest.fixture
def throw() -> Throw:
    """
    A throw steep enough to land.
    """
    return SteepThrowsLand().perform(
        parameters=ThrowParameters(
            angle=0.7, speed=3.0, release=ReleasePoint(height=1.5)
        )
    )


def test_a_trial_round_trips_through_the_database(database_url, court_id, throw):
    """
    A trial read back from the database equals the trial kept, marked as started from
    the store's starting state.
    """
    store = ThrowsInADatabase(database_url=database_url, starting_state_id=court_id)
    store.prepare()

    store.add(trial=throw)

    assert throw.starting_state_id == court_id
    assert store.stored_trials() == [throw]


def test_trials_are_read_back_by_the_state_they_started_from(
    database_url, court_id, throw
):
    """
    A store reads back only the trials of its own starting state.
    """
    other_court_id = stored_court_id(database_url=database_url, basket_distance=3.0)
    store = ThrowsInADatabase(database_url=database_url, starting_state_id=court_id)
    other_store = ThrowsInADatabase(
        database_url=database_url, starting_state_id=other_court_id
    )
    store.prepare()
    other_throw = SteepThrowsLand().perform(parameters=throw.parameters)

    store.add(trial=throw)
    other_store.add(trial=other_throw)

    assert store.stored_trials() == [throw]
    assert other_store.stored_trials() == [other_throw]


# %% many trials at once


def uniformly_drawn_throws(*, number_of_draws: int) -> DrawnParameters[ThrowParameters]:
    """
    :param number_of_draws: How many throws to draw.
    :return: Throws drawn uniformly from the throws a person can make.
    """
    return DrawnParameters(
        parameter_space=ThrowSpace(),
        model_registry=UniformPriorRegistry(),
        number_of_draws=number_of_draws,
    )


def test_a_single_worker_keeps_every_trial_it_tries(database_url, court_id):
    """
    A campaign of one worker keeps a trial of every parameters its source gives.
    """
    setup = ThrowingSetup(database_url=database_url, starting_state_id=court_id)
    source = uniformly_drawn_throws(number_of_draws=5)
    campaign = TrialCampaign(setup=setup, parameter_source=source)

    kept = campaign.run()

    assert kept == source.number_of_draws
    assert [throw.parameters for throw in setup.create_store().stored_trials()] == (
        source.parameters_to_try()
    )


def test_several_workers_keep_every_trial_of_their_share(database_url, court_id):
    """
    Two workers in processes of their own keep all trials in the shared database, each
    trying a share of the drawn parameters of its own.
    """
    setup = ThrowingSetup(database_url=database_url, starting_state_id=court_id)
    source = uniformly_drawn_throws(number_of_draws=8)
    campaign = TrialCampaign(setup=setup, parameter_source=source, number_of_workers=2)

    kept = campaign.run()
    angles = [throw.parameters.angle for throw in setup.create_store().stored_trials()]

    assert kept == source.number_of_draws
    assert sorted(angles) == sorted(
        parameters.angle for parameters in source.parameters_to_try()
    )


def test_workers_that_need_more_memory_than_is_free_are_refused(
    database_url, database_path
):
    """
    A campaign whose workers would need more working memory than any machine has does
    not start.
    """
    campaign = TrialCampaign(
        setup=ThrowingSetup(database_url=database_url),
        parameter_source=uniformly_drawn_throws(number_of_draws=2),
        number_of_workers=2,
        memory_per_worker=10**18,
    )
    with pytest.raises(NotEnoughMemoryForWorkers):
        campaign.run()
    assert not database_path.exists()


# %% checking the outcome before scaling up


def test_a_campaign_whose_pilot_trials_all_fail_stops_after_them(
    database_url, court_id
):
    """
    When no angle a person can throw at lands the ball, the campaign stops after the
    pilot trials and keeps only them.
    """
    setup = ThrowingSetup(
        database_url=database_url,
        starting_state_id=court_id,
        performer=SteepThrowsLand(steepness=Bounds(lower=2.0, upper=3.0)),
    )
    campaign = TrialCampaign(
        setup=setup,
        parameter_source=uniformly_drawn_throws(number_of_draws=20),
        pilot_trials=5,
        number_of_workers=2,
    )

    with pytest.raises(OutcomeDidNotVaryInPilot):
        campaign.run()

    assert len(setup.create_store().stored_trials()) == campaign.pilot_trials


def test_a_campaign_whose_pilot_trials_vary_tries_every_trial(database_url, court_id):
    """
    When some pilot throws land and some do not, the campaign goes on to try every drawn
    throw.
    """
    setup = ThrowingSetup(database_url=database_url, starting_state_id=court_id)
    source = uniformly_drawn_throws(number_of_draws=30)
    campaign = TrialCampaign(setup=setup, parameter_source=source, pilot_trials=20)

    kept = campaign.run()

    assert kept == source.number_of_draws
    assert len(setup.create_store().stored_trials()) == source.number_of_draws


# %% what made trials succeed


@pytest.fixture(scope="module")
def analysis(drawn) -> TrialAnalysis:
    """
    The analysis of the drawn throws, which land exactly when they are steep enough.
    """
    performer = SteepThrowsLand()
    return TrialAnalysis(
        trials=[performer.perform(parameters=parameters) for parameters in drawn]
    )


def test_the_success_rate_is_the_fraction_of_trials_that_succeeded(analysis):
    """
    The success rate counts every trial once.
    """
    succeeded = [trial.succeeded for trial in analysis.trials]
    assert analysis.success_rate == sum(succeeded) / len(succeeded)


def test_the_values_success_depends_on_have_the_strongest_effects(analysis):
    """
    Without wind, the aimed and the reached angle are the same and alone decide whether
    a throw lands, so their effects come first.
    """
    first, second, *_ = analysis.effects_on_success()
    assert {first.attribute_path, second.attribute_path} == {
        "parameters.angle",
        "measurements.reached_angle",
    }


def test_a_measurement_that_decides_success_has_a_stronger_effect_than_its_cause(
    drawn,
):
    """
    With wind, the reached angle alone decides whether a throw lands, so its effect is
    stronger than that of the aimed angle.
    """
    np.random.seed(3)
    performer = SteepThrowsLand(wind=0.2)
    analysis = TrialAnalysis(
        trials=[performer.perform(parameters=parameters) for parameters in drawn * 10]
    )
    [strongest, *_] = analysis.effects_on_success()
    assert strongest.attribute_path == "measurements.reached_angle"


def test_the_ranges_of_a_parameter_cover_every_trial(analysis):
    """
    Each trial falls into exactly one range of each parameter.
    """
    for effect in analysis.effects_on_success():
        assert sum(success.number_of_trials for success in effect.ranges) == len(
            analysis.trials
        )


def test_the_successful_parameters_are_those_of_the_trials_that_succeeded(analysis):
    """
    Exactly the parameters of the successful trials are returned, in order.
    """
    assert analysis.successful_parameters() == [
        trial.parameters for trial in analysis.trials if trial.succeeded
    ]


def test_an_analysis_without_trials_is_refused():
    """
    There is nothing to analyse without any trial.
    """
    with pytest.raises(NoTrials):
        TrialAnalysis(trials=[])

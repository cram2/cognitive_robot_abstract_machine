"""
Tests of finding out by trials what makes a behaviour succeed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from krrood.bounds import Bounds
from krrood.entity_query_language.factories import (
    a,
    and_,
    average,
    probability_of,
    set_of,
    variable,
)
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.feature_extraction.feature_extractor import FeatureExtractor
from krrood.parametrization.model_registries import UniformPriorRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from krrood.parametrization.probabilistic_backend import ProbabilisticBackend
from krrood.parametrization.trials.campaigns import TrialCampaign
from krrood.parametrization.trials.exceptions import (
    InformedFieldMissing,
    NoSuccessfulTrials,
    OutcomeDidNotVaryInPilot,
)
from krrood.parametrization.trials.models import TrialModel
from krrood.parametrization.trials.parameter_spaces import (
    DrawnParameters,
    GivenParameters,
)
from krrood.parametrization.trials.records import InformedField
from krrood_compute_resources.exceptions import NotEnoughMemoryForWorkers
from krrood_compute_resources.workers import WorkerPool
from ..dataset.throwing_trials import (
    Court,
    CourtRecipe,
    CourtsInADatabase,
    FastThrowSpace,
    ReleasePoint,
    SteepThrowsLand,
    Throw,
    ThrowBall,
    ThrowingSetup,
    ThrowMeasurements,
    ThrowParameters,
    ThrowSpace,
    ThrowsInADatabase,
)

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


def test_the_statement_names_its_variables_like_the_features_of_the_parameters():
    """
    Every field of the parameters, the nested one included, is left to be drawn, named
    as a model learned from parameters names it.
    """
    example = ThrowParameters(angle=0.0, speed=0.0, release=ReleasePoint(height=0.0))
    statement = ThrowSpace().statement()
    names = UnderspecifiedParameters(statement=statement).variables
    features = FeatureExtractor.from_instances(instances=[example])
    assert set(names) == set(features.create_dataframe(instances=[example]).columns)


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
    store = CourtsInADatabase(database_url=database_url)
    store.prepare()
    return store.add(state=Court(basket_distance=basket_distance))


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


# %% reaching starting states


@pytest.mark.parametrize("number_of_workers", [1, 2])
def test_only_the_states_wanted_are_kept(database_url, number_of_workers):
    """
    Of the courts set up, in this process or in workers of their own, exactly those
    whose basket stands within the wanted distances are kept.
    """
    recipe = CourtRecipe(database_url=database_url)
    np.random.seed(5)

    stored_ids = recipe.reach_and_store(
        attempts=20, worker_pool=WorkerPool(number_of_workers=number_of_workers)
    )
    store = recipe.create_store()
    courts = [store.stored_court(database_id=court_id) for court_id in stored_ids]

    assert 0 < len(courts) < 20
    assert len(set(stored_ids)) == len(stored_ids)
    assert all(recipe.is_wanted(state=court) for court in courts)


def test_trials_start_from_a_state_the_recipe_kept(database_url):
    """
    The trials of a campaign that starts from a kept court are read back as episodes of
    that court.
    """
    recipe = CourtRecipe(database_url=database_url, placement_spread=0.0)
    [court_id] = recipe.reach_and_store(attempts=1, worker_pool=WorkerPool())
    setup = ThrowingSetup(database_url=database_url, starting_state_id=court_id)
    source = uniformly_drawn_throws(number_of_draws=3)

    TrialCampaign(setup=setup, parameter_source=source).run()

    assert len(setup.create_store().stored_trials()) == source.number_of_draws


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
    campaign = TrialCampaign(
        setup=setup,
        parameter_source=source,
        worker_pool=WorkerPool(number_of_workers=2),
    )

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
        worker_pool=WorkerPool(number_of_workers=2, memory_per_worker=10**18),
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
        worker_pool=WorkerPool(number_of_workers=2),
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


# %% repeating trials


@pytest.fixture
def central_and_edge_throws() -> GivenParameters[ThrowParameters]:
    """
    A throw aimed at the middle of the steep enough angles and one aimed just above
    their lower edge.
    """
    return GivenParameters(
        parameters=[
            ThrowParameters(angle=angle, speed=3.0, release=ReleasePoint(height=1.5))
            for angle in [0.75, 0.61]
        ]
    )


def success_rates_by_angle(*, trials: list[Throw]) -> dict[float, float]:
    """
    :param trials: Throws, some of them of the same parameters.
    :return: The fraction of the throws of each aimed angle that landed.
    """
    throw = variable(Throw, domain=trials)
    angle = throw.parameters.angle
    success_rate = average(throw.succeeded)
    return {
        row[angle]: row[success_rate]
        for row in set_of(angle, success_rate).grouped_by(angle).evaluate()
    }


def test_each_parameters_is_tried_as_often_as_it_is_repeated(
    database_url, court_id, central_and_edge_throws
):
    """
    Every given parameters is tried as many times as the campaign repeats it, spread
    over the workers.
    """
    setup = ThrowingSetup(database_url=database_url, starting_state_id=court_id)
    campaign = TrialCampaign(
        setup=setup,
        parameter_source=central_and_edge_throws,
        repetitions=3,
        worker_pool=WorkerPool(number_of_workers=2),
    )

    kept = campaign.run()
    angles = [throw.parameters.angle for throw in setup.create_store().stored_trials()]

    assert kept == len(central_and_edge_throws.parameters) * campaign.repetitions
    assert sorted(angles) == sorted(
        parameters.angle
        for parameters in central_and_edge_throws.parameters
        for _ in range(campaign.repetitions)
    )


def test_the_pilot_of_repeated_parameters_tries_distinct_parameters(
    database_url, court_id
):
    """
    Without wind a throw's outcome follows from its parameters, so a pilot as long as
    the given throws varies only if it tries each of them before repeating any.
    """
    setup = ThrowingSetup(database_url=database_url, starting_state_id=court_id)
    landing_and_missing_throws = GivenParameters(
        parameters=[
            ThrowParameters(angle=angle, speed=3.0, release=ReleasePoint(height=1.5))
            for angle in [0.75, 0.3]
        ]
    )
    campaign = TrialCampaign(
        setup=setup,
        parameter_source=landing_and_missing_throws,
        repetitions=3,
        pilot_trials=len(landing_and_missing_throws.parameters),
    )

    kept = campaign.run()

    assert kept == len(landing_and_missing_throws.parameters) * campaign.repetitions


@pytest.fixture
def repeated_throws_in_the_wind(
    database_url, court_id, central_and_edge_throws
) -> list[Throw]:
    """
    The central and the edge throw, each tried forty times in the wind.
    """
    setup = ThrowingSetup(
        database_url=database_url,
        starting_state_id=court_id,
        performer=SteepThrowsLand(wind=0.1),
    )
    np.random.seed(4)
    TrialCampaign(
        setup=setup, parameter_source=central_and_edge_throws, repetitions=40
    ).run()
    return setup.create_store().stored_trials()


def test_repetitions_tell_apart_parameters_that_succeed_by_chance(
    repeated_throws_in_the_wind, central_and_edge_throws
):
    """
    With wind, a throw aimed at the middle of the steep enough angles lands more often
    than one aimed just above their edge, each rate being the fraction of its repeated
    trials that landed.
    """
    central, edge = central_and_edge_throws.parameters
    rates = success_rates_by_angle(trials=repeated_throws_in_the_wind)

    assert rates[central.angle] > rates[edge.angle]
    assert rates[central.angle] == np.mean(
        [
            throw.succeeded
            for throw in repeated_throws_in_the_wind
            if throw.parameters.angle == central.angle
        ]
    )


# %% what the model of the trials tells


@pytest.fixture(scope="module")
def many_drawn() -> list[ThrowParameters]:
    """
    Enough throws drawn uniformly to tell effects apart from chance.
    """
    np.random.seed(6)
    return ThrowSpace().drawn_parameters(
        number_of_draws=400, model_registry=UniformPriorRegistry()
    )


@pytest.fixture(scope="module")
def throws_in_the_wind(many_drawn) -> list[Throw]:
    """
    Throws turned by the wind, so that only the angle the ball actually leaves the hand
    at decides whether it lands.
    """
    np.random.seed(7)
    performer = SteepThrowsLand(wind=0.2)
    return [performer.perform(parameters=parameters) for parameters in many_drawn]


@pytest.fixture(scope="module")
def model_in_the_wind(throws_in_the_wind) -> TrialModel:
    """
    The model of the throws turned by the wind.
    """
    return TrialModel(trials=throws_in_the_wind)


def test_the_probability_of_success_is_the_fraction_of_trials_that_succeeded(
    model_in_the_wind, throws_in_the_wind
):
    """
    The model says throws land as often as the throws it was learned from did.
    """
    throw = variable(Throw, domain=throws_in_the_wind)
    probability = probability_of(throw.succeeded == True).first(  # noqa: E712
        backend=ProbabilisticBackend(model_registry=model_in_the_wind.registry)
    )
    assert probability == pytest.approx(
        np.mean([trial.succeeded for trial in throws_in_the_wind]), abs=1e-6
    )


def test_steep_throws_succeed_more_often_than_flat_ones(
    model_in_the_wind, throws_in_the_wind
):
    """
    Given the angle the ball leaves the hand at is steep enough, a throw is more likely
    to land than given it is too flat.
    """
    steepness = SteepThrowsLand().steepness
    throw = variable(Throw, domain=throws_in_the_wind)
    reached = throw.measurements.reached_angle
    backend = ProbabilisticBackend(model_registry=model_in_the_wind.registry)

    steep = probability_of(
        throw.succeeded == True,  # noqa: E712
        and_(reached >= steepness.lower, reached < steepness.upper),
    ).first(backend=backend)
    flat = probability_of(
        throw.succeeded == True,  # noqa: E712
        reached < steepness.lower,
    ).first(backend=backend)

    assert steep > flat


def landing() -> Match[Throw]:
    """
    :return: What causes a throw to land, with no cause marked, so that the learned
        structure says which features are its causes.
    """
    throw = a(Throw)(
        parameters=a(ThrowParameters)(
            angle=..., speed=..., release=a(ReleasePoint)(height=...)
        ),
        measurements=a(ThrowMeasurements)(reached_angle=...),
        succeeded=...,
    )
    throw.causes_effect(throw.succeeded == True)  # noqa: E712
    return throw


def test_only_the_features_joined_to_success_are_its_causes(model_in_the_wind):
    """
    The aimed and the reached angle are causes of landing; the speed and the release
    height, which do not change whether a throw lands, are not.
    """
    causes = model_in_the_wind.causal_backend.rank_causes(expression=landing())
    throw = variable(Throw, domain=[])
    assert {found.cause_variable.name for found in causes} == {
        throw.parameters.angle._name_,
        throw.measurements.reached_angle._name_,
    }


def test_the_measurement_that_decides_success_is_its_strongest_cause(
    model_in_the_wind,
):
    """
    Setting the reached angle within its best region makes nearly every throw land.
    """
    [strongest, *_] = model_in_the_wind.causal_backend.rank_causes(expression=landing())
    assert (
        strongest.cause_variable.name
        == variable(Throw, domain=[]).measurements.reached_angle._name_
    )
    assert strongest.effect_probability_given_region > 0.9


# %% closing the loop


@pytest.fixture(scope="module")
def drawn_throws(drawn) -> list[Throw]:
    """
    The drawn throws, which land exactly when they are steep enough.
    """
    performer = SteepThrowsLand()
    return [performer.perform(parameters=parameters) for parameters in drawn]


def test_the_next_campaign_drawn_near_the_successes_succeeds_more_often(
    drawn, drawn_throws
):
    """
    Throws drawn near the throws that landed land more often than throws drawn
    uniformly.
    """
    registry = TrialModel(trials=drawn_throws).registry_near_successes()
    np.random.seed(8)
    performer = SteepThrowsLand()
    next_throws = [
        performer.perform(parameters=parameters)
        for parameters in ThrowSpace().drawn_parameters(
            number_of_draws=len(drawn), model_registry=registry
        )
    ]

    assert np.mean([throw.succeeded for throw in next_throws]) > np.mean(
        [throw.succeeded for throw in drawn_throws]
    )


def test_drawing_near_the_successes_of_trials_that_all_failed_is_refused(drawn):
    """
    There is nothing to draw near when no throw landed.
    """
    performer = SteepThrowsLand(steepness=Bounds(lower=2.0, upper=3.0))
    trials = [performer.perform(parameters=parameters) for parameters in drawn]
    with pytest.raises(NoSuccessfulTrials):
        TrialModel(trials=trials).registry_near_successes()


def test_the_best_values_are_adopted_for_the_fields_they_inform(
    repeated_throws_in_the_wind, central_and_edge_throws
):
    """
    The values of the throw that landed most often are adopted for the fields of the
    action its parameters inform.
    """
    rates = success_rates_by_angle(trials=repeated_throws_in_the_wind)
    [best] = [
        parameters
        for parameters in central_and_edge_throws.parameters
        if rates[parameters.angle] == max(rates.values())
    ]

    space = ThrowSpace()

    adopted = {
        value.informed_field.informed._attribute_name_: value.value
        for value in space.adopted_values(parameters=best)
    }

    assert adopted == {
        informed_field.informed._attribute_name_: (
            informed_field.parameter.apply_mapping_on_external_root(instance=best)
        )
        for informed_field in space.informed_fields
    }
    assert best == central_and_edge_throws.parameters[0]


def test_an_informed_field_knows_the_default_it_would_replace():
    """
    The default an informed field has now is the one its action declares.
    """
    informed_field = InformedField(
        parameter=variable(ThrowParameters, domain=[]).angle,
        informed=variable(ThrowBall, domain=[]).aim_angle,
    )
    assert informed_field.current_default == ThrowBall().aim_angle


def test_informing_a_field_the_action_does_not_declare_is_refused():
    """
    A misspelt field, which the entity query language accepts as an attribute, is
    refused when it is declared, not when a value is adopted.
    """
    with pytest.raises(InformedFieldMissing):
        InformedField(
            parameter=variable(ThrowParameters, domain=[]).angle,
            informed=variable(ThrowBall, domain=[]).aim_angel,
        )

"""
Representing a curved region as a union of axis-aligned boxes.

The regions here are the ones a threshold on a Euclidean distance describes, and each
test pins one property the representation has to have: that it covers the region, that
spending more terms captures more of it, and that a single term reproduces exactly the
fidelity the set kind survey reports.
"""

import math

import pytest

from experiments.random_events_experiments.curved_region_approximation import (
    CoveringApproximation,
    CurvedRegion,
    EuclideanBall,
    EuclideanDisc,
    RegionApproximationExperiment,
)


def approximation(region: CurvedRegion, subdivisions: int) -> CoveringApproximation:
    """
    :param region: The region to represent.
    :param subdivisions: Number of parts each subdivided axis is cut into.
    :return: The covering representation of that region.
    """
    return CoveringApproximation(region=region, subdivisions=subdivisions)


# %% one box is the bounding box


def test_single_box_reproduces_the_fidelity_a_bounding_box_gives():
    """
    Spending one term on a region gives its bounding box, so the fidelity reported there
    is the ratio of the region to that box -- which is what the survey reports for a
    predicate it represents with one term.
    """
    ball = approximation(EuclideanBall(radius=1.0), 1)
    disc = approximation(EuclideanDisc(width=2.0, height=1.0), 1)

    assert ball.intersection_over_union() == math.pi / 6.0
    assert disc.intersection_over_union() == math.pi / 4.0


def test_single_box_costs_a_single_term():
    """
    The bounding box is one box, whichever region it bounds.
    """
    assert len(approximation(EuclideanBall(radius=1.0), 1).boxes) == 1
    assert len(approximation(EuclideanDisc(width=2.0, height=1.0), 1).boxes) == 1


# %% what spending more terms buys


def test_subdividing_captures_more_of_the_region():
    """
    Every further subdivision cuts away part of the representation that the region does
    not fill, so the fidelity rises with the terms spent and never falls.
    """
    fidelities = [
        approximation(EuclideanBall(radius=1.0), count).intersection_over_union()
        for count in (1, 2, 4, 8, 16)
    ]

    assert fidelities == sorted(fidelities)
    assert fidelities[0] < fidelities[-1]


def test_what_is_not_captured_halves_when_the_subdivisions_double():
    """
    The part of the representation the region does not fill shrinks in proportion to the
    subdivisions, so a curved condition is not approximated to some fixed loss but to
    whatever loss is paid for.
    """
    losses = [
        1.0
        - approximation(
            EuclideanDisc(width=2.0, height=1.0), count
        ).intersection_over_union()
        for count in (16, 32, 64, 128)
    ]

    for finer, coarser in zip(losses[1:], losses):
        assert 1.7 < coarser / finer < 2.0


def test_an_even_split_through_the_centre_buys_nothing():
    """
    Cutting a region into two halves through its centre leaves each half as wide as the
    whole, so the terms are spent without capturing any more of the region.

    What is paid for is subdivision away from the centre.
    """
    disc = EuclideanDisc(width=2.0, height=1.0)

    assert (
        approximation(disc, 2).intersection_over_union()
        == approximation(disc, 1).intersection_over_union()
    )


def test_a_disc_takes_one_box_per_strip_and_a_ball_one_per_strip_and_slab():
    """
    A ball has to be subdivided along two axes rather than one, since slabs alone leave
    a circular cross-section in each, so the boxes covering it grow as the square of the
    boxes covering a disc.

    ..note:: This is the subdivision the region is cut into, not what the representation
        costs: the algebra merges and splits those boxes when it makes them disjoint.
    """
    assert len(approximation(EuclideanDisc(width=2.0, height=1.0), 8).boxes) == 8
    assert len(approximation(EuclideanBall(radius=1.0), 8).boxes) == 64


# %% the representation is sound


def test_representation_covers_the_whole_region():
    """
    A covering representation holds wherever the region does, so a condition represented
    this way is never reported unsatisfiable at a state that satisfies it.
    """
    region = EuclideanDisc(width=2.0, height=1.0)
    boxes = approximation(region, 8).boxes

    for angle in range(0, 360, 15):
        for fraction in (0.0, 0.5, 0.999):
            point = (
                fraction * math.cos(math.radians(angle)),
                fraction * math.sin(math.radians(angle)),
                0.0,
            )
            assert any(
                box.simple_event.contains(point) for box in boxes
            ), f"{point} is in the disc but in none of the boxes"


def test_representation_never_claims_more_than_it_covers():
    """
    The region is contained in its representation, so the fidelity is a fraction of one
    rather than a ratio that can exceed it.
    """
    for count in (1, 2, 4, 8):
        assert approximation(EuclideanBall(radius=1.0), count).volume() >= (
            EuclideanBall(radius=1.0).volume
        )


# %% the terms are the terms the algebra holds


@pytest.mark.parametrize("subdivisions", (1, 2, 4, 8))
def test_reported_cost_is_the_number_of_terms_the_algebra_holds(subdivisions: int):
    """
    What a representation costs is the number of terms the product algebra ends up
    holding, which is not the number of boxes handed to it: making a union disjoint
    merges some boxes and splits others, so counting the boxes reports a cost the
    representation does not have.
    """
    for region in (EuclideanDisc(width=2.0, height=1.0), EuclideanBall(radius=1.0)):
        [reported] = RegionApproximationExperiment(
            regions={"Region": region}, subdivision_counts=(subdivisions,)
        ).run()

        assert reported.simple_sets == len(
            approximation(region, subdivisions).event.simple_sets
        )


def test_volume_of_the_representation_survives_being_made_disjoint():
    """
    The measured fidelity is a property of the event the product algebra holds, so
    making that event disjoint -- which is what turns a union into terms -- must not
    change the volume it encloses.
    """
    covering = approximation(EuclideanDisc(width=2.0, height=1.0), 6)

    assert covering.volume() == pytest.approx(sum(box.volume for box in covering.boxes))


# %% reported results


def test_every_region_is_reported_at_every_subdivision():
    """
    The experiment reports the whole exchange rate rather than one point of it, so a
    reader can see what a given fidelity costs.
    """
    results = RegionApproximationExperiment(
        regions={
            "Ball": EuclideanBall(radius=1.0),
            "Disc": EuclideanDisc(width=2.0, height=1.0),
        },
        subdivision_counts=(1, 2, 4),
    ).run()

    assert [(each.region, each.subdivisions) for each in results] == [
        ("Ball", 1),
        ("Ball", 2),
        ("Ball", 4),
        ("Disc", 1),
        ("Disc", 2),
        ("Disc", 4),
    ]

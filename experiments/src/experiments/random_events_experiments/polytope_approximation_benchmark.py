"""
Benchmark for :meth:`~random_events.polytope.Polytope.maximum_inner_box`,
:meth:`~random_events.polytope.Polytope.inner_box_approximation` and
:meth:`~random_events.polytope.Polytope.outer_box_approximation` on 2D and 3D
polytopes.

Polytopes are convex hulls of random point clouds sampled on/within a sphere shell,
which gives well-conditioned full-dimensional polytopes with a controllable number of
facets (more points -> more facets, up to the O(n^floor(d/2)) facet blow-up of random
points on a sphere for higher dimensions -- not relevant here since we only use d in
{2, 3}).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import tqdm
from scipy.spatial import ConvexHull

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    TypstRenderer,
)
from random_events.polytope import Polytope


def random_hull_points(
    number_of_points: int, number_of_dimensions: int, seed: int
) -> npt.NDArray[np.float64]:
    """
    :param number_of_points: Number of points to sample.
    :param number_of_dimensions: Dimensionality of each point.
    :param seed: Seed for the random number generator.
    :return: ``number_of_points`` points sampled on/within a sphere shell in
        ``number_of_dimensions`` dimensions, whose convex hull is a well-conditioned
        full-dimensional polytope.
    """
    random_number_generator = np.random.default_rng(seed)
    points = random_number_generator.normal(
        size=(number_of_points, number_of_dimensions)
    )
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= random_number_generator.uniform(0.5, 1.0, size=(number_of_points, 1))
    return points


def exact_volume(points: npt.NDArray[np.float64]) -> float:
    """
    :param points: The points a polytope's convex hull was built from.
    :return: The exact volume of the convex hull of ``points``, computed by scipy's
        ``ConvexHull`` (a deterministic simplicial decomposition) rather than
        ``Polytope.volume``, which is a randomized Monte-Carlo estimate. This is the
        ground truth :attr:`PolytopeApproximationBenchmarkResult.inner_volume_diff`
        and :attr:`PolytopeApproximationBenchmarkResult.outer_volume_diff` are measured
        against, so it needs to be exact rather than approximate: any noise in it would
        otherwise leak into both diffs and make them unreliable as a measure of the box
        approximations' own error.
    """
    return ConvexHull(points).volume


@dataclass
class PolytopeApproximationBenchmarkResult(ExperimentResult):
    """
    One measurement of :meth:`~random_events.polytope.Polytope.maximum_inner_box`,
    :meth:`~random_events.polytope.Polytope.inner_box_approximation` and
    :meth:`~random_events.polytope.Polytope.outer_box_approximation` on a single
    randomly generated polytope.
    """

    number_of_dimensions: int
    """
    Dimensionality of the measured polytope.
    """

    number_of_points: int
    """
    Number of points the polytope's convex hull was built from.
    """

    number_of_facets: int
    """
    Number of facets (inequalities) of the polytope.
    """

    volume: float
    """
    Exact volume of the polytope, see :func:`exact_volume`.
    """

    min_volume_fraction: float
    """
    ``min_volume`` passed to ``inner_box_approximation``/``outer_box_approximation``,
    as a fraction of :attr:`volume`.
    """

    maximum_inner_box_duration: float
    """
    Time spent computing the single largest inscribed box, in seconds.
    """

    inner_box_approximation_duration: float
    """
    Time spent computing the inner box approximation, in seconds.
    """

    inner_box_count: int
    """
    Number of boxes in the inner box approximation.
    """

    inner_volume_diff: float
    """
    :attr:`volume` minus the exact volume covered by the inner box approximation: the
    volume the approximation misses. Always >= 0, since the inner box approximation is
    a subset of the polytope.
    """

    outer_box_approximation_duration: float
    """
    Time spent computing the outer box approximation, in seconds.
    """

    outer_box_count: int
    """
    Number of boxes in the outer box approximation.
    """

    outer_volume_diff: float
    """
    The exact volume covered by the outer box approximation minus :attr:`volume`: the
    excess volume the approximation covers. Always >= 0, since the outer box
    approximation is a superset of the polytope.
    """


def run_benchmark(
    number_of_dimensions: int,
    number_of_points: int,
    min_volume_fraction: float,
    seed: int = 42,
) -> PolytopeApproximationBenchmarkResult:
    """
    Measure a single randomly generated polytope.

    :param number_of_dimensions: Dimensionality of the polytope to generate.
    :param number_of_points: Number of points the polytope's convex hull is built from.
    :param min_volume_fraction: ``min_volume`` passed to ``inner_box_approximation``/
        ``outer_box_approximation``, as a fraction of the polytope's volume.
    :param seed: Seed for the random number generator the polytope is sampled with.
    :return: Timing and approximation-quality measurements for this polytope.
    """
    points = random_hull_points(number_of_points, number_of_dimensions, seed)
    polytope = Polytope.from_points(points)
    volume = exact_volume(points)
    min_volume = volume * min_volume_fraction

    begin = time.perf_counter()
    polytope.maximum_inner_box()
    maximum_inner_box_duration = time.perf_counter() - begin

    begin = time.perf_counter()
    inner_event = polytope.inner_box_approximation(min_volume)
    inner_box_approximation_duration = time.perf_counter() - begin

    begin = time.perf_counter()
    outer_event = polytope.outer_box_approximation(min_volume)
    outer_box_approximation_duration = time.perf_counter() - begin

    return PolytopeApproximationBenchmarkResult(
        number_of_dimensions=number_of_dimensions,
        number_of_points=number_of_points,
        number_of_facets=polytope.A.shape[0],
        volume=volume,
        min_volume_fraction=min_volume_fraction,
        maximum_inner_box_duration=maximum_inner_box_duration,
        inner_box_approximation_duration=inner_box_approximation_duration,
        inner_box_count=len(inner_event.simple_sets),
        inner_volume_diff=volume - inner_event.size,
        outer_box_approximation_duration=outer_box_approximation_duration,
        outer_box_count=len(outer_event.simple_sets),
        outer_volume_diff=outer_event.size - volume,
    )


@dataclass
class PolytopeApproximationBenchmarkSweep:
    """
    One sweep of :func:`run_benchmark` over a range of point counts, at a fixed
    dimensionality and minimum-volume fraction.
    """

    number_of_dimensions: int
    """
    Dimensionality of every polytope generated in this sweep.
    """

    point_counts: list[int]
    """
    Point counts to run :func:`run_benchmark` with, one measurement each.
    """

    min_volume_fraction: float
    """
    ``min_volume_fraction`` passed to every measurement in this sweep.
    """


def main():
    sweeps = [
        PolytopeApproximationBenchmarkSweep(
            number_of_dimensions=2,
            point_counts=[10, 25, 50, 100, 200, 400, 800],
            min_volume_fraction=0.1,
        ),
        PolytopeApproximationBenchmarkSweep(
            number_of_dimensions=3,
            point_counts=[10, 20, 30, 50, 75, 100, 150, 200],
            min_volume_fraction=0.2,
        ),
    ]

    results = []
    for sweep in sweeps:
        description = f"{sweep.number_of_dimensions}D polytopes"
        for number_of_points in tqdm.tqdm(sweep.point_counts, desc=description):
            results.append(
                run_benchmark(
                    sweep.number_of_dimensions,
                    number_of_points,
                    sweep.min_volume_fraction,
                )
            )
    table = ExperimentsTable(results)

    print(
        TypstRenderer(table, reported_decimals=4).render_figure(
            "Timings and approximation quality of Polytope.maximum_inner_box, "
            "inner_box_approximation and outer_box_approximation on convex hulls of "
            "random point clouds in 2D and 3D, with min_volume set to "
            "min_volume_fraction times the polytope's volume. inner_volume_diff and "
            "outer_volume_diff are the exact gap between the polytope's volume and "
            "the volume covered by its box approximation: inner_box_approximation is "
            "a subset of the polytope so it can only under-cover, and "
            "outer_box_approximation is a superset so it can only over-cover."
        )
    )


if __name__ == "__main__":
    main()

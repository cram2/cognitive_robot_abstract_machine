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
import tqdm

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    TypstRenderer,
)
from random_events.polytope import Polytope


def random_hull_points(n: int, dim: int, seed: int) -> np.ndarray:
    """
    :param n: Number of points to sample.
    :param dim: Dimensionality of each point.
    :param seed: Seed for the random number generator.
    :return: ``n`` points sampled on/within a sphere shell in ``dim`` dimensions,
        whose convex hull is a well-conditioned full-dimensional polytope.
    """
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, dim))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= rng.uniform(0.5, 1.0, size=(n, 1))
    return points


def make_polytope(n: int, dim: int, seed: int) -> Polytope:
    """
    :param n: Number of points the polytope's convex hull is built from.
    :param dim: Dimensionality of the polytope.
    :param seed: Seed for the random number generator.
    :return: A random polytope, see :func:`random_hull_points`.
    """
    return Polytope.from_points(random_hull_points(n, dim, seed))


@dataclass
class PolytopeApproximationBenchmarkResult(ExperimentResult):
    """
    One measurement of :meth:`~random_events.polytope.Polytope.maximum_inner_box`,
    :meth:`~random_events.polytope.Polytope.inner_box_approximation` and
    :meth:`~random_events.polytope.Polytope.outer_box_approximation` on a single
    randomly generated polytope.
    """

    dim: int
    """
    Dimensionality of the measured polytope.
    """

    n_points: int
    """
    Number of points the polytope's convex hull was built from.
    """

    n_facets: int
    """
    Number of facets (inequalities) of the polytope.
    """

    volume: float
    """
    Monte-Carlo estimate of the polytope's volume (``Polytope.volume``).
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
    dim: int, n_points: int, min_volume_fraction: float, seed: int = 42
) -> PolytopeApproximationBenchmarkResult:
    """
    Measure a single randomly generated polytope.

    :param dim: Dimensionality of the polytope to generate.
    :param n_points: Number of points the polytope's convex hull is built from.
    :param min_volume_fraction: ``min_volume`` passed to ``inner_box_approximation``/
        ``outer_box_approximation``, as a fraction of the polytope's volume.
    :param seed: Seed for the random number generator the polytope is sampled with.
    :return: Timing and approximation-quality measurements for this polytope.
    """
    polytope = make_polytope(n_points, dim, seed)
    volume = polytope.volume
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
        dim=dim,
        n_points=n_points,
        n_facets=polytope.A.shape[0],
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


def main():
    configs = [
        # (dim, [n_points...], min_volume_fraction)
        (2, [10, 25, 50, 100], 0.1),
        (3, [10, 20, 30], 0.2),
    ]

    results = []
    for dim, sizes, min_volume_fraction in configs:
        for n_points in tqdm.tqdm(sizes, desc=f"{dim}D polytopes"):
            results.append(run_benchmark(dim, n_points, min_volume_fraction))
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

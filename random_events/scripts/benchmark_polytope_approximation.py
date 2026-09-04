"""
Benchmark for Polytope.maximum_inner_box / inner_box_approximation / outer_box_approximation
(random_events/src/random_events/polytope.py) on 2D and 3D polytopes.

Polytopes are convex hulls of random point clouds sampled on/within a sphere shell, which
gives well-conditioned full-dimensional polytopes with a controllable number of facets
(more points -> more facets, up to the O(n^floor(d/2)) facet blow-up of random points on
a sphere for higher dimensions -- not relevant here since we only use d in {2, 3}).

Usage:
    PYTHONPATH=/opt/cram/random_events/src python3 benchmark_polytope_approximation.py
"""

import time
from dataclasses import dataclass
from typing import List

import numpy as np

from random_events.polytope import Polytope


def random_hull_points(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, dim))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= rng.uniform(0.5, 1.0, size=(n, 1))
    return points


def make_polytope(n: int, dim: int, seed: int) -> Polytope:
    points = random_hull_points(n, dim, seed)
    return Polytope.from_points(points)


@dataclass
class BenchmarkResult:
    dim: int
    n_points: int
    n_facets: int
    volume: float
    min_volume_fraction: float
    maximum_inner_box_time: float
    inner_box_approximation_time: float
    inner_box_count: int
    outer_box_approximation_time: float
    outer_box_count: int


def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def run_benchmark(dim: int, n_points: int, min_volume_fraction: float, seed: int = 42) -> BenchmarkResult:
    poly = make_polytope(n_points, dim, seed)
    volume = poly.volume
    min_volume = volume * min_volume_fraction

    t_max_inner, _ = time_call(poly.maximum_inner_box)
    t_inner, inner_event = time_call(poly.inner_box_approximation, min_volume)
    t_outer, outer_event = time_call(poly.outer_box_approximation, min_volume)

    return BenchmarkResult(
        dim=dim,
        n_points=n_points,
        n_facets=poly.A.shape[0],
        volume=volume,
        min_volume_fraction=min_volume_fraction,
        maximum_inner_box_time=t_max_inner,
        inner_box_approximation_time=t_inner,
        inner_box_count=len(inner_event.simple_sets),
        outer_box_approximation_time=t_outer,
        outer_box_count=len(outer_event.simple_sets),
    )


def main():
    configs = [
        # (dim, [n_points...], min_volume_fraction)
        (2, [10, 25, 50, 100], 0.1),
        (3, [10, 20, 30], 0.2),
    ]

    results: List[BenchmarkResult] = []
    for dim, sizes, frac in configs:
        for n in sizes:
            results.append(run_benchmark(dim, n, frac))

    header = (
        f"{'dim':>3} {'n_pts':>6} {'facets':>6} {'volume':>8} {'min_vol_frac':>12} "
        f"{'max_inner_box[s]':>17} {'inner_approx[s]':>16} {'#boxes':>7} "
        f"{'outer_approx[s]':>16} {'#boxes':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.dim:>3} {r.n_points:>6} {r.n_facets:>6} {r.volume:>8.3f} {r.min_volume_fraction:>12.2f} "
            f"{r.maximum_inner_box_time:>17.4f} {r.inner_box_approximation_time:>16.4f} {r.inner_box_count:>7} "
            f"{r.outer_box_approximation_time:>16.4f} {r.outer_box_count:>7}"
        )


if __name__ == "__main__":
    main()

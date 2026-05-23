"""
Load the UCI zoo dataset as plain :class:`Animal` instances.

Returns animals with ``species=None`` (underspecified) plus the parallel list of
ground-truth :class:`Species` targets, so the RDR can be fit and scored.
"""

from __future__ import annotations

import os
import pickle

from typing_extensions import List, Optional, Tuple

from .animal import Animal, Species

#: 16 trait columns of the zoo dataset; all but ``legs`` are 0/1 booleans.
_BOOL_FIELDS = (
    "hair",
    "feathers",
    "eggs",
    "milk",
    "airborne",
    "aquatic",
    "predator",
    "toothed",
    "backbone",
    "breathes",
    "venomous",
    "fins",
    "tail",
    "domestic",
    "catsize",
)

#: Default cache location — reuses the pickles already committed for the legacy RDR tests.
DEFAULT_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "test_ripple_down_rules",
    "test_results",
    "zoo_dataset.pkl",
)


def _load_cached(cache_file: str) -> Optional[dict]:
    """Load the {features, targets, ids} parts from the per-key pickle files, or None."""
    if not cache_file.endswith(".pkl"):
        cache_file += ".pkl"
    parts = {}
    for key in ("features", "targets", "ids"):
        part_file = cache_file.replace(".pkl", f"_{key}.pkl")
        if not os.path.exists(part_file):
            return None
        with open(part_file, "rb") as f:
            parts[key] = pickle.load(f)
    return parts


def _save_cache(dataset, cache_file: str) -> None:
    if not cache_file.endswith(".pkl"):
        cache_file += ".pkl"
    parts = {
        "features": dataset.data.features,
        "targets": dataset.data.targets,
        "ids": dataset.data.ids,
    }
    for key, value in parts.items():
        with open(cache_file.replace(".pkl", f"_{key}.pkl"), "wb") as f:
            pickle.dump(value, f)


def _fetch_zoo(cache_file: Optional[str]) -> Optional[dict]:
    """Return {features, targets, ids} dataframes from cache or the UCI repo."""
    if cache_file is not None:
        cached = _load_cached(cache_file)
        if cached is not None:
            return cached
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=111)
    if dataset is None or not hasattr(dataset, "data"):
        return None
    if cache_file is not None:
        _save_cache(dataset, cache_file)
    return {
        "features": dataset.data.features,
        "targets": dataset.data.targets,
        "ids": dataset.data.ids,
    }


def load_zoo_animals(
    cache_file: Optional[str] = DEFAULT_CACHE_FILE,
) -> Tuple[List[Animal], List[Species]]:
    """
    :param cache_file: Where to cache/load the dataset; ``None`` forces a fresh fetch.
    :return: ``(animals, targets)`` where each animal has ``species=None`` and
        ``targets[i]`` is the ground-truth :class:`Species` for ``animals[i]``.
        Returns ``([], [])`` if the dataset cannot be obtained.
    """
    try:
        zoo = _fetch_zoo(cache_file)
    except (ConnectionError, ImportError):
        return [], []
    if zoo is None:
        return [], []

    features = zoo["features"]
    target_ids = zoo["targets"].values.flatten()
    names = zoo["ids"].values.flatten()

    animals: List[Animal] = []
    targets: List[Species] = []
    for i, (_, row) in enumerate(features.iterrows()):
        animals.append(
            Animal(
                name=str(names[i]),
                legs=int(row["legs"]),
                species=None,
                **{field: bool(row[field]) for field in _BOOL_FIELDS},
            )
        )
        targets.append(Species(int(target_ids[i])))
    return animals, targets

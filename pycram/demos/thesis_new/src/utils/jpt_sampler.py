"""Load per-(action, robot) Joint Probability Trees and use them as the primary
base-pose sampler for the thesis_new demos.

The JPTs in ``pycram/demos/thesis_new/jpts`` are in the original ``jpt`` library
format (``jpt.trees.JPT``) and were trained only on *successful* executions. Each
tree models the joint distribution over the geometric "cause" features that relate
the robot base pose to the target (e.g. ``robot_to_target_dist``, ``robot_yaw_rad``,
``cut_normal_approach_*``).

Rather than inverting features into a pose (the azimuth around the target is not a
feature, so that is underdetermined), the JPT is used as a *selector*: the caller
generates reachable base-pose candidates from the existing costmap, computes each
candidate's cause-feature vector, and this module scores them by likelihood under
the success distribution. The highest-likelihood candidate lies closest to the
learned success manifold.

The module stays intentionally light (no world/robot imports) so it can be loaded
and unit-tested standalone.
"""

from __future__ import annotations

import functools
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Map a demo task name to the manifest action key.
_ACTION_ALIASES = {
    "cut": "cutting",
    "cutting": "cutting",
    "mix": "mixing",
    "mixing": "mixing",
    "pour": "pouring",
    "pouring": "pouring",
    "wipe": "wiping",
    "wiping": "wiping",
}

# Demo-side robot name -> manifest (jpt-side) robot key. This is the inverse of
# ``world_setup.ROBOT_NAME_ALIASES``; kept local to avoid importing the heavy
# world_setup module (which pulls in robot descriptions).
_ROBOT_NAME_TO_MANIFEST = {
    "justin": "rollin_justin",
    "g1": "unitree_g1",
}

_JPTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "jpts")
)
_MANIFEST_PATH = os.path.join(_JPTS_DIR, "action_robot_jpt_manifest.json")

DEFAULT_MAX_CANDIDATES = 30
DEFAULT_RERANK_POOL = 40


@functools.lru_cache(maxsize=1)
def _load_manifest() -> dict:
    with open(_MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@functools.lru_cache(maxsize=None)
def _load_jpt(file_name: str):
    # Lazy import so the module imports without the jpt dependency present.
    from jpt.trees import JPT

    path = os.path.join(_JPTS_DIR, os.path.basename(file_name))
    return JPT.load(path, protocol="json")


def manifest_robot_key(robot_name: str) -> str:
    """Translate a demo-side robot name to the manifest/jpt key."""
    name = str(robot_name).strip().lower()
    return _ROBOT_NAME_TO_MANIFEST.get(name, name)


class JptPoseScorer:
    """Scores a candidate's cause-feature vector by its likelihood under a JPT.

    The score is the joint density over the manifest ``cause_cols`` (the effect
    column is marginalised out). Higher is more like the successful executions the
    tree was trained on.
    """

    def __init__(self, action: str, robot: str, entry: dict, model):
        self.action = action
        self.robot = robot
        self.cause_cols: List[str] = list(entry["cause_cols"])
        self._model = model
        # Ordered variable names of the tree; a full-width row is required by the
        # library's preprocessing even though only cause variables are scored.
        self._varnames: List[str] = list(model.varnames)
        self._cause_vars = [model.varnames[col] for col in self.cause_cols]

    def score(self, features: Dict[str, float]) -> float:
        """:return: likelihood of ``features`` (NaN if any cause value is missing/invalid)."""
        row = {name: 0.0 for name in self._varnames}
        for col in self.cause_cols:
            value = features.get(col)
            if value is None or not np.isfinite(value):
                return float("nan")
            row[col] = float(value)
        frame = pd.DataFrame([row], columns=self._varnames)
        try:
            likelihood = self._model.likelihood(frame, variables=self._cause_vars)
        except Exception:
            return float("nan")
        return float(np.asarray(likelihood).reshape(-1)[0])

    def score_many(self, feature_dicts: List[Dict[str, float]]) -> np.ndarray:
        """Vectorised :meth:`score` over many feature dicts (one likelihood call).

        :return: array of likelihoods aligned with ``feature_dicts`` (NaN where a
            cause value is missing/invalid).
        """
        if not feature_dicts:
            return np.empty(0, dtype=float)
        rows = []
        valid = np.ones(len(feature_dicts), dtype=bool)
        for index, features in enumerate(feature_dicts):
            row = {name: 0.0 for name in self._varnames}
            for col in self.cause_cols:
                value = features.get(col)
                if value is None or not np.isfinite(value):
                    valid[index] = False
                    break
                row[col] = float(value)
            rows.append(row)
        frame = pd.DataFrame(rows, columns=self._varnames)
        try:
            likelihood = np.asarray(
                self._model.likelihood(frame, variables=self._cause_vars)
            ).reshape(-1).astype(float)
        except Exception:
            return np.full(len(feature_dicts), np.nan)
        likelihood[~valid] = np.nan
        return likelihood


@functools.lru_cache(maxsize=None)
def get_pose_scorer(action: str, robot: str) -> Optional[JptPoseScorer]:
    """Return a scorer for ``(action, robot)``, or ``None`` if no JPT is available.

    A ``None`` result signals the caller to fall back to the default costmap
    behaviour (e.g. cutting has no JPT for ``stretch``).
    """
    action_key = _ACTION_ALIASES.get(str(action).strip().lower())
    if action_key is None:
        return None
    try:
        manifest = _load_manifest()
    except Exception:
        return None
    action_entry = manifest.get(action_key)
    if not action_entry:
        return None
    entry = action_entry.get(manifest_robot_key(robot))
    if entry is None:
        return None
    try:
        model = _load_jpt(entry["file_path"])
    except Exception:
        return None
    return JptPoseScorer(action_key, manifest_robot_key(robot), entry, model)


def rerank_by_score(candidates, feature_fn, scorer, *, pool_size=DEFAULT_RERANK_POOL):
    """Reorder costmap-ranked candidates by JPT likelihood (descending).

    Only the first ``pool_size`` candidates (already costmap-sorted, best first) are
    scored and reordered; the remainder is appended unchanged as a cheap fallback
    tail. Designed for ``CostmapLocation.candidate_reranker`` so the downstream
    collision search runs in JPT-preferred order without scoring the whole pool.

    :param candidates: list of pre-collision candidate poses (costmap-sorted).
    :param feature_fn: maps a candidate to its cause-feature dict.
    :param scorer: a :class:`JptPoseScorer`.
    """
    candidates = list(candidates)
    head = candidates[:pool_size]
    tail = candidates[pool_size:]
    if not head:
        return tail
    scores = scorer.score_many([feature_fn(candidate) for candidate in head])
    sort_keys = [s if math.isfinite(s) else -math.inf for s in scores]
    order = sorted(range(len(head)), key=lambda i: sort_keys[i], reverse=True)
    return [head[i] for i in order] + tail


def select_best_candidate(candidates, feature_fn, scorer, *, max_candidates=DEFAULT_MAX_CANDIDATES):
    """Pick the highest-likelihood candidate from an iterable.

    :param candidates: iterable of candidate objects (e.g. poses).
    :param feature_fn: maps a candidate to its cause-feature dict.
    :param scorer: a :class:`JptPoseScorer`.
    :return: ``(best_candidate, best_score, num_scored)``; ``best_candidate`` is
        ``None`` when nothing scored finitely.
    """
    best = None
    best_score = -math.inf
    num_scored = 0
    for candidate in candidates:
        if num_scored >= max_candidates:
            break
        score = scorer.score(feature_fn(candidate))
        num_scored += 1
        if math.isfinite(score) and score > best_score:
            best_score = score
            best = candidate
    if best is None:
        return None, float("nan"), num_scored
    return best, best_score, num_scored

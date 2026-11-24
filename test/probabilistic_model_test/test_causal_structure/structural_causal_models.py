"""
Small structural causal models with a known graph, sampled with a fixed seed, for the
structure learning tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# %% sampling


def noisy_copy(values: np.ndarray, flip_rate: float, random_state) -> np.ndarray:
    """
    :param values: Binary values.
    :param flip_rate: How often a value is flipped.
    :param random_state: Source of randomness.
    :return: The values with that share flipped.
    """
    return values ^ (random_state.random(len(values)) < flip_rate)


def season_fork(row_count: int = 5000, seed: int = 0) -> pd.DataFrame:
    """
    :return: Rows of season → ice_cream_sales, season → drowning_incidents, all binary.
    """
    random_state = np.random.default_rng(seed)
    season = random_state.integers(0, 2, row_count)
    return pd.DataFrame(
        {
            "season": season,
            "ice_cream_sales": noisy_copy(season, 0.2, random_state),
            "drowning_incidents": noisy_copy(season, 0.2, random_state),
        }
    )


def collider(row_count: int = 5000, seed: int = 0) -> pd.DataFrame:
    """
    :return: Rows of A → C ← B, all binary, C the noisy disjunction of A and B.
    """
    random_state = np.random.default_rng(seed)
    first = random_state.integers(0, 2, row_count)
    second = random_state.integers(0, 2, row_count)
    return pd.DataFrame(
        {
            "A": first,
            "B": second,
            "C": noisy_copy(first | second, 0.1, random_state),
        }
    )


def chain_with_a_copy(row_count: int = 5000, seed: int = 0) -> pd.DataFrame:
    """
    :return: Rows of A → B, with D an exact copy of B.
    """
    random_state = np.random.default_rng(seed)
    first = random_state.integers(0, 2, row_count)
    second = noisy_copy(first, 0.2, random_state)
    return pd.DataFrame({"A": first, "B": second, "D": second.copy()})


def measured_confounder(row_count: int = 20000, seed: int = 0) -> pd.DataFrame:
    """
    :return: Rows of V → X ← C → Y, X → Y, all binary: V and C fair coins, X the noisy
        disjunction of V and C, Y that of X and C.
    """
    random_state = np.random.default_rng(seed)
    instrument = random_state.integers(0, 2, row_count)
    confounder = random_state.integers(0, 2, row_count)
    cause = noisy_copy(instrument | confounder, 0.1, random_state)
    return pd.DataFrame(
        {
            "V": instrument,
            "C": confounder,
            "X": cause,
            "Y": noisy_copy(cause | confounder, 0.1, random_state),
        }
    )


def pure_hidden_confounding(row_count: int = 20000, seed: int = 0) -> pd.DataFrame:
    """
    :return: Rows of W → X ← L → Y, all binary, without the hidden L: X does not cause
        Y.
    """
    random_state = np.random.default_rng(seed)
    hidden = random_state.integers(0, 2, row_count)
    other = random_state.integers(0, 2, row_count)
    return pd.DataFrame(
        {
            "W": other,
            "X": noisy_copy(hidden | other, 0.1, random_state),
            "Y": noisy_copy(hidden, 0.1, random_state),
        }
    )


def hidden_cause_and_effect_confounder(
    row_count: int = 20000, seed: int = 0
) -> pd.DataFrame:
    """
    :return: Rows of W → X → Y with X ← L → Y, all binary, without the hidden L.
    """
    random_state = np.random.default_rng(seed)
    hidden = random_state.integers(0, 2, row_count)
    other = random_state.integers(0, 2, row_count)
    cause = noisy_copy(hidden | other, 0.1, random_state)
    return pd.DataFrame(
        {
            "W": other,
            "X": cause,
            "Y": noisy_copy(cause | hidden, 0.1, random_state),
        }
    )

"""
A survey whose fields are typed with interfaces declared in another package.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..regions import Disc, Region

# %% a class whose fields are typed with the interfaces


@dataclass(kw_only=True)
class Survey:
    """
    A survey of a region, sampled within a disc.
    """

    surveyed_region: Region
    """
    The region the survey covers.
    """

    sampling_disc: Disc
    """
    The disc within which samples are taken.
    """

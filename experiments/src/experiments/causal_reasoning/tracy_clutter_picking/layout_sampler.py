"""
Random clutter layouts for the ten-milk mock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Dict, List, Tuple

from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutterEnvironment,
    ClutterSceneLayout,
    FrictionLadder,
    ObjectCategory,
    PlacedObject,
)


@dataclass(frozen=True)
class EnvironmentDistribution:
    """
    How one kind of environment's clutters are laid out and what friction their objects
    have.

    This is what makes the environment a confounder: it drives both how crowded the
    target is and how much friction the grasp gets, so the two are correlated in the
    recorded attempts without either causing the other.
    """

    minimum_spacing: float
    """
    Smallest centre-to-centre grid spacing, in metres.
    """

    maximum_spacing: float
    """
    Largest centre-to-centre grid spacing, in metres.
    """

    friction_levels: Tuple[float, ...]
    """
    The friction coefficients an attempt in this environment draws from.
    """

    @classmethod
    def of_mock_environments(
        cls, ladder: FrictionLadder = FrictionLadder()
    ) -> Dict[ClutterEnvironment, EnvironmentDistribution]:
        """
        The mock's environments: a table leaves room between the cartons and holds any
        material, a bin packs them tightly and holds the slippery ones.

        :param ladder: The friction levels an attempt can be given.
        :return: Each environment's distribution, by environment.
        """
        return {
            ClutterEnvironment.TABLE: cls(0.09, 0.14, tuple(ladder.levels)),
            ClutterEnvironment.BIN: cls(0.065, 0.09, ladder.lowest_levels(3)),
        }


@dataclass
class ClutterLayoutSampler:
    """
    Draws random clutter layouts: a jittered grid of objects in one of the mock's
    environments, one of them picked as the target.
    """

    random_state: np.random.Generator
    """
    Source of randomness.
    """

    object_count: int = 10
    """
    How many objects each layout holds, the target included.
    """

    distributions: Dict[ClutterEnvironment, EnvironmentDistribution] = field(
        default_factory=EnvironmentDistribution.of_mock_environments
    )
    """
    The environments a layout is drawn from, each equally likely, and how each lays its
    clutter out.
    """

    cluster_centre_x: float = 0.8
    """
    Where the clutter is centred along the robot's x-axis, in metres: in front of
    Tracy's left arm, where a top-down reach has been confirmed to succeed.
    """

    cluster_centre_y: float = 0.25
    """
    Where the clutter is centred along the robot's y-axis, in metres.
    """

    columns: int = 4
    """
    How many objects stand side by side along the table's x-axis in one row of the
    cluster.
    """

    position_jitter: float = 0.008
    """
    Largest offset, in metres, an object is shifted off its grid position along each
    axis.
    """

    yaw_jitter: float = 0.3
    """
    Largest rotation, in radians, an object is turned away from the grid's own
    alignment.
    """

    grasp_face_yaws: Tuple[float, ...] = (0.0, math.pi / 2)
    """
    The gripper yaws, relative to the target, that close the fingers across one of its
    two pairs of faces.
    """

    grasp_yaw_error: float = 0.1
    """
    Largest error, in radians, the grasp's yaw is off the face pair it aims across.
    """

    def sample(self) -> ClutterSceneLayout:
        """
        :return: One random layout.
        """
        environments = list(self.distributions)
        environment = environments[self.random_state.integers(len(environments))]
        distribution = self.distributions[environment]
        spacing = float(
            self.random_state.uniform(
                distribution.minimum_spacing, distribution.maximum_spacing
            )
        )
        objects = self._grid(spacing)
        return ClutterSceneLayout(
            environment=environment,
            objects=objects,
            target_index=int(self.random_state.integers(len(objects))),
            friction_coefficient=float(
                distribution.friction_levels[
                    self.random_state.integers(len(distribution.friction_levels))
                ]
            ),
            grasp_yaw=float(
                self.grasp_face_yaws[
                    self.random_state.integers(len(self.grasp_face_yaws))
                ]
                + self.random_state.uniform(-self.grasp_yaw_error, self.grasp_yaw_error)
            ),
        )

    def _grid(self, spacing: float) -> List[PlacedObject]:
        """
        Lay the objects out on a jittered grid centred on the cluster centre.

        :param spacing: Centre-to-centre distance between grid positions, in metres.
        :return: The placed objects, row by row.
        """
        row_count = math.ceil(self.object_count / self.columns)
        x_offset = (self.columns - 1) / 2
        y_offset = (row_count - 1) / 2
        objects = []
        for index in range(self.object_count):
            row, column = divmod(index, self.columns)
            jitter_x, jitter_y = self.random_state.uniform(
                -self.position_jitter, self.position_jitter, size=2
            )
            objects.append(
                PlacedObject(
                    category=ObjectCategory.MILK,
                    x=self.cluster_centre_x
                    + (column - x_offset) * spacing
                    + float(jitter_x),
                    y=self.cluster_centre_y
                    + (row - y_offset) * spacing
                    + float(jitter_y),
                    yaw=float(
                        self.random_state.uniform(-self.yaw_jitter, self.yaw_jitter)
                    ),
                )
            )
        return objects

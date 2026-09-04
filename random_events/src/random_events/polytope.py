import itertools
from collections import deque
from typing import Optional

import numpy as np
import polytope
from ortools.linear_solver import pywraplp
from scipy.spatial import ConvexHull
from typing_extensions import Self, Tuple

from random_events.interval import closed_open
from random_events.product_algebra import Event, SimpleEvent, Continuous


class NoOptimalSolutionError(Exception):
    """
    Exception raised when the solver does not find an optimal solution.
    """

    pass


class Polytope(polytope.Polytope):
    """
    Extension of the polytope class from the polytope library.

    This class enables conversion to simple events and provides the inner box and outer
    box approximation from
    https://cse.lab.imtlucca.it/~bemporad/publications/papers/compgeom-boxes.pdf.
    """

    @classmethod
    def from_polytope(cls, polytope_: polytope.Polytope) -> Self:
        """
        Create a polytope from a polytope object.

        :param polytope_: The polytope object.
        """
        return cls(polytope_.A, polytope_.b)

    @classmethod
    def from_points(cls, points: np.ndarray) -> Self:
        """
        Create a polytope from a set of points, by computing the convex hull of the
        points and using the hull's facet equations as the polytope's inequalities.

        :param points: A numpy array with shape (n, dimensions) containing the points.
        """
        convex_hull = ConvexHull(points)
        a = convex_hull.equations[:, :-1]
        b = -convex_hull.equations[:, -1]
        return cls(a, b)

    def inner_box_approximation(self, minimum_volume: float = 0.1) -> Event:
        """
        Compute an inner box approximation of the polytope.

        Similar to algorithm 5.

        :param minimum_volume: The minimum volume (epsilon) for the approximation. If a
            box is created in the induction with lower volume than epsilon, it will not
            be split further.
        :return: The inner box approximation of the polytope as a random event.
        """
        # initialize a queue with polytopes that need to be approximated
        working_queue = deque([self])
        resulting_boxes = []

        while working_queue:
            current_polytope = working_queue.popleft()
            inner_box = current_polytope.maximum_inner_box()
            resulting_boxes.append(inner_box)

            # if the inner box is too small, we do not split it further
            if inner_box.volume < minimum_volume:
                continue

            # append the polytope without the inner box to the queue
            diff = polytope.mldivide(current_polytope, inner_box)
            working_queue.extend(
                [self.__class__.from_polytope(p) for p in diff.list_poly]
            )

        return Event.from_simple_sets(
            *[box.to_simple_event() for box in resulting_boxes]
        ).make_disjoint()

    @classmethod
    def _box_polytope_from_bounds(cls, lower: np.ndarray, upper: np.ndarray) -> Self:
        """
        Build a box-shaped polytope from already-known per-dimension bounds, and
        pre-populate its bounding-box and volume caches (`bbox`/`_volume`, read by the
        `bounding_box`/`volume` properties inherited from the `polytope` library).

        Without this, a box built via `from_box` still recomputes its bounding box
        via `2 * n_dimensions` LP solves and its volume via randomized Monte-Carlo
        sampling the first time either is accessed, even though both are already known
        exactly at construction time here.

        :param lower: The lower bound per dimension.
        :param upper: The upper bound per dimension.
        :return: The box polytope, with `.bounding_box` and `.volume` pre-cached.
        """
        lower = np.asarray(lower).reshape(-1, 1)
        upper = np.asarray(upper).reshape(-1, 1)
        result = cls.from_box(list(zip(lower.flatten(), upper.flatten())))
        result.bbox = (lower, upper)
        result._set_volume(float(np.prod(upper - lower)))
        return result

    def as_box_polytope(self) -> Self:
        """
        :return: The polytope as box polytope.
        """
        lower, upper = self.bounding_box
        return self._box_polytope_from_bounds(lower, upper)

    def is_box(
        self,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        tolerance: float = 1e-7,
    ) -> bool:
        """
        Check whether this polytope is exactly its own axis-aligned bounding box.

        The bounding box always contains the polytope, so it is a subset of the
        polytope too (making the two equal) exactly when every one of the box's
        2**n_dimensions corners already satisfies this polytope's inequalities. This
        answers the same question as ``self.as_box_polytope() <= self``, but with a
        single matrix multiplication instead of polytope's LP-based set-difference
        machinery, which dominates the runtime of `outer_box_approximation`.

        :param lower: The lower bounds of the bounding box. Computed from
            `self.bounding_box` if not given.
        :param upper: The upper bounds of the bounding box. Computed from
            `self.bounding_box` if not given.
        :param tolerance: The numerical tolerance for the inequality check.
        :return: Whether this polytope equals its own bounding box.
        """
        if lower is None or upper is None:
            lower, upper = self.bounding_box
        corners = np.array(
            list(itertools.product(*zip(lower.flatten(), upper.flatten())))
        )
        return bool(np.all(self.A @ corners.T <= self.b[:, None] + tolerance))

    def copy(self):
        return self.from_polytope(self.__copy__())

    def split_on_axis_value(self, axis: int, value: np.ndarray) -> Tuple[Self, Self]:
        """
        Split the polytope on a specific axis and value.

        :param axis: The axis to split on.
        :param value: The value to split on.
        :return: The left and right split of the polytope.
        """
        a_vector = np.zeros((1, self.A.shape[1]))
        a_vector[0, axis] = 1.0
        b_vector = value

        # construct left split
        left = self.copy()
        left.A = np.concatenate([left.A, a_vector])
        left.b = np.concatenate([left.b, b_vector])

        # construct right split
        right = self.copy()
        right.A = np.concatenate([right.A, -a_vector])
        right.b = np.concatenate([right.b, -b_vector])

        return left, right

    def outer_box_approximation(self, minimum_volume: float = 0.1) -> Event:
        """
        Compute an outer box approximation of the polytope.
        This implements Algorithm 6 in https://cse.lab.imtlucca.it/~bemporad/publications/papers/compgeom-boxes.pdf

        :param minimum_volume: The minimum volume (epsilon) for the approximation.

        :return: The outer box approximation of the polytope as a random event.
        """
        polytopes_to_split = deque([self])
        resulting_boxes = []

        while polytopes_to_split:
            current_polytope = polytopes_to_split.popleft()
            lower, upper = current_polytope.bounding_box
            volume = np.prod(upper - lower)

            # if the box is too small, or the polytope is already box-shaped, skip
            if volume < minimum_volume or current_polytope.is_box(lower, upper):
                resulting_boxes.append(current_polytope)
                continue

            # get the longest side
            side_lengths = upper - lower
            longest_side = np.argmax(side_lengths)

            # split the box in half along the longest side
            splitting_point = (lower[longest_side] + upper[longest_side]) / 2
            left, right = current_polytope.split_on_axis_value(
                longest_side, splitting_point
            )
            polytopes_to_split.extend([left, right])

        return Event.from_simple_sets(
            *[box.to_simple_event() for box in resulting_boxes]
        ).make_disjoint()

    def maximum_inner_box(self) -> Self:
        """
        Compute the maximum single inner box approximation of the polytope.

        This implements Algorithm 2 in
        https://cse.lab.imtlucca.it/~bemporad/publications/papers/compgeom-boxes.pdf

        :return: The maximum inner box of the polytope.
        """
        # calculate bounding box
        minima, maxima = self.bounding_box
        minima = minima.flatten()
        maxima = maxima.flatten()

        solver = pywraplp.Solver.CreateSolver("GLOP")

        # create variables for the dimensions of the inner box approximation (x_0, x_1, ..., x_n)
        dimension_variables = [
            solver.NumVar(minimum, maximum, f"x_{i}")
            for i, (minimum, maximum) in enumerate(zip(minima, maxima))
        ]

        # create the scale variable (lambda in the paper)
        scale = solver.NumVar(0, 1, "scale")

        # set the goal to maximize lambda
        solver.Maximize(scale)

        # create the guess for the r vector
        scale_of_box = maxima - minima

        # create the matrix A^+
        a_positive = np.maximum(0, self.A)

        # create the constraints from proposition 2
        for a, a_positive, b in zip(self.A, a_positive, self.b):
            solver.Add(
                sum(a * dimension_variables) + sum(a_positive * scale_of_box * scale)
                <= b
            )

        # solve the problem
        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            raise NoOptimalSolutionError(
                f"No optimal solution found for the bounding box {self}. "
            )

        # calculate the inner box
        box_lower = np.array([dimension.solution_value() for dimension in dimension_variables])
        box_upper = box_lower + scale_of_box * scale.solution_value()
        return self.__class__._box_polytope_from_bounds(box_lower, box_upper)

    def to_simple_event(self) -> SimpleEvent:
        """
        Convert the polytope to a simple event by using its bounding box.
        """
        minima, maxima = self.bounding_box
        minima = minima.flatten()
        maxima = maxima.flatten()
        return SimpleEvent.from_data(
            {
                Continuous(f"x_{i}"): closed_open(minimum, maximum)
                for i, (minimum, maximum) in enumerate(zip(minima, maxima))
            }
        )

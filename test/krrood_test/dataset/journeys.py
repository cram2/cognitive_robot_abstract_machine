"""
Journeys whose fields hold their stops either as a collection or as one object that can
be iterated, for reading the elements of an expression a domain is given as.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Iterator

from krrood.symbol_graph.symbol_graph import Symbol

# %% stops


@dataclass(eq=False)
class Stop(Symbol):
    """
    A place a journey passes through.
    """

    name: str
    """
    The name of the place.
    """


@dataclass(eq=False)
class Route(Symbol):
    """
    One object that can be iterated over its stops.
    """

    stops: list[Stop] = field(default_factory=list)
    """
    The stops of the route, in order.
    """

    def __iter__(self) -> Iterator[Stop]:
        """
        :return: The stops of the route, in order.
        """
        return iter(self.stops)


# %% journeys


@dataclass(eq=False)
class Journey(Symbol):
    """
    A journey along one route, with optional stops on top of it.
    """

    name: str
    """
    The name of the journey.
    """

    route: Route
    """
    The route the journey follows.
    """

    extra_stops: list[Stop] | None = None
    """
    The stops added to the route, or ``None`` when none were planned.
    """

    def every_stop(self):
        """
        :return: The stops of the route followed by the extra stops, without a declared
            return type.
        """
        return [*self.route.stops, *(self.extra_stops or [])]

from dataclasses import dataclass

from pycram.locations.base import Location, PoseGeneratorBackend


@dataclass
class GiskardLocationBackend(PoseGeneratorBackend):

    def __iter__(self):
        pass

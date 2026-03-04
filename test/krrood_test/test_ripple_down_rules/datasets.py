from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import sqlalchemy
from sqlalchemy import ForeignKey
from sqlalchemy.orm import (
    MappedAsDataclass,
    Mapped,
    mapped_column,
    relationship,
    MappedColumn,
)
from typing_extensions import Tuple, List, Set, Optional, Self, ClassVar
from ucimlrepo import fetch_ucirepo

from krrood.ripple_down_rules.datastructures.case import (
    Case,
    create_cases_from_dataframe,
)
from krrood.ripple_down_rules.datastructures.enums import Category
from krrood.ripple_down_rules.datastructures.tracked_object import TrackedObjectMixin
from krrood.ripple_down_rules.rdr_decorators import RDRDecorator


def load_cached_dataset(cache_file):
    """Loads the dataset from cache if it exists."""
    dataset = {}
    if ".pkl" not in cache_file:
        cache_file += ".pkl"
    for key in ["features", "targets", "ids"]:
        part_file = cache_file.replace(".pkl", f"_{key}.pkl")
        if not os.path.exists(part_file):
            return None
        with open(part_file, "rb") as f:
            dataset[key] = pickle.load(f)
    return dataset


def save_dataset_to_cache(dataset, cache_file):
    """Saves only essential parts of the dataset to cache."""
    dataset_to_cache = {
        "features": dataset.data.features,
        "targets": dataset.data.targets,
        "ids": dataset.data.ids,
    }

    for key, value in dataset_to_cache.items():
        with open(cache_file.replace(".pkl", f"_{key}.pkl"), "wb") as f:
            pickle.dump(dataset_to_cache[key], f)
    print("Dataset cached successfully.")


def get_dataset(dataset_id, cache_file: Optional[str] = None):
    """Fetches dataset from cache or downloads it if not available."""
    if cache_file is not None:
        if not cache_file.endswith(".pkl"):
            cache_file += ".pkl"
    dataset = load_cached_dataset(cache_file) if cache_file else None
    if dataset is None:
        print("Downloading dataset...")
        dataset = fetch_ucirepo(id=dataset_id)

        # Check if dataset is valid before caching
        if dataset is None or not hasattr(dataset, "data"):
            print("Error: Failed to fetch dataset.")
            return None

        if cache_file:
            save_dataset_to_cache(dataset, cache_file)

        dataset = {
            "features": dataset.data.features,
            "targets": dataset.data.targets,
            "ids": dataset.data.ids,
        }

    return dataset


def load_zoo_dataset(
    cache_file: Optional[str] = None,
) -> Tuple[List[Case], List[Species]]:
    """
    Load the zoo dataset.

    :param cache_file: the cache file to store the dataset or load it from.
    :return: all cases and targets.
    """
    # fetch dataset
    zoo = get_dataset(111, cache_file)

    # data (as pandas dataframes)
    X = zoo["features"]
    y = zoo["targets"]
    # get ids as list of strings
    ids = zoo["ids"].values.flatten()
    all_cases = create_cases_from_dataframe(X, name="Animal")

    category_names = [
        "mammal",
        "bird",
        "reptile",
        "fish",
        "amphibian",
        "insect",
        "molusc",
    ]
    category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
    # targets = [getattr(SpeciesCol, category_id_to_name[i]) for i in y.values.flatten()]
    targets = [Species.from_str(category_id_to_name[i]) for i in y.values.flatten()]
    return all_cases, targets


def load_zoo_cases(cache_file: Optional[str] = None) -> List[Case]:
    """
    Load the zoo dataset cases.

    :param cache_file: the cache file to store the dataset or load it from.
    :return: all cases.
    """
    all_cases, _ = load_zoo_dataset(cache_file)
    return all_cases


class Species(Category):
    mammal = "mammal"
    bird = "bird"
    reptile = "reptile"
    fish = "fish"
    amphibian = "amphibian"
    insect = "insect"
    molusc = "molusc"


class Habitat(Category):
    """
    A habitat category is a category that represents the habitat of an animal.
    """

    land = "land"
    water = "water"
    air = "air"


@dataclass(eq=False)
class PhysicalObject:
    """
    A physical object is an object that can be contained in a container.
    """

    name: str
    """
    The name of the object.
    """
    _contained_objects: List[PhysicalObject] = field(default_factory=list)
    """
    The list of objects contained in this object.
    """
    _rdr_json_dir: ClassVar[str] = os.path.join(
        os.path.dirname(__file__), "test_results"
    )
    """
    The directory where the RDR serialized JSON files are stored.
    """
    _is_a_robot_rdr: ClassVar[RDRDecorator] = RDRDecorator(
        _rdr_json_dir, (bool,), True, package_name="test"
    )
    """
    The RDR decorator that is used to determine if the object is a robot or not.
    """
    _select_parts_rdr: ClassVar[RDRDecorator] = RDRDecorator(
        _rdr_json_dir, (Self,), False, package_name="test"
    )
    """
    The RDR decorator that is used to determine if the object is a robot or not.
    """

    @property
    def contained_objects(self) -> List[PhysicalObject]:
        return self._contained_objects

    @contained_objects.setter
    def contained_objects(self, value: List[PhysicalObject]):
        self._contained_objects = value

    @_is_a_robot_rdr.decorator
    def is_a_robot(self) -> bool:
        pass

    @_select_parts_rdr.decorator
    def select_objects_that_are_parts_of_robot(
        self, objects: List[PhysicalObject], robot: Robot
    ) -> List[PhysicalObject]:
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, PhysicalObject):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


@dataclass(eq=False)
class Part(PhysicalObject): ...


@dataclass(eq=False)
class Robot(PhysicalObject):
    parts: List[Part] = field(default_factory=list)


class Base(sqlalchemy.orm.DeclarativeBase):
    pass


class HabitatTable(MappedAsDataclass, Base):
    __tablename__ = "Habitat"

    id: Mapped[int] = mapped_column(init=False, primary_key=True, autoincrement=True)
    habitat: Mapped[Habitat]
    animal_id: MappedColumn = mapped_column(ForeignKey("Animal.id"), init=False)

    def __hash__(self):
        return hash(self.habitat)

    def __str__(self):
        return f"{HabitatTable.__name__}({Habitat.__name__}.{self.habitat.name})"

    def __repr__(self):
        return self.__str__()


class MappedAnimal(MappedAsDataclass, Base):
    __tablename__ = "Animal"

    id: Mapped[int] = mapped_column(init=False, primary_key=True, autoincrement=True)
    name: Mapped[str]
    hair: Mapped[bool]
    feathers: Mapped[bool]
    eggs: Mapped[bool]
    milk: Mapped[bool]
    airborne: Mapped[bool]
    aquatic: Mapped[bool]
    predator: Mapped[bool]
    toothed: Mapped[bool]
    backbone: Mapped[bool]
    breathes: Mapped[bool]
    venomous: Mapped[bool]
    fins: Mapped[bool]
    legs: Mapped[int]
    tail: Mapped[bool]
    domestic: Mapped[bool]
    catsize: Mapped[bool]
    species: Mapped[Species] = mapped_column(nullable=True)

    habitats: Mapped[Set[HabitatTable]] = relationship(default_factory=set)


@dataclass(unsafe_hash=True)
class WorldEntity(TrackedObjectMixin):
    world: Optional[World] = field(default=None, kw_only=True, repr=False, hash=False)


@dataclass(unsafe_hash=True)
class Body(WorldEntity):
    name: str


@dataclass(unsafe_hash=True)
class Handle(Body): ...


@dataclass(unsafe_hash=True)
class Container(Body): ...


@dataclass(unsafe_hash=True)
class Connection(WorldEntity):
    parent: Body
    child: Body


@dataclass(unsafe_hash=True)
class FixedConnection(Connection): ...


@dataclass(unsafe_hash=True)
class PrismaticConnection(Connection): ...


@dataclass
class World:
    id: int = 0
    bodies: List[Body] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    views: List[View] = field(default_factory=list, repr=False)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, World):
            return False
        return self.id == other.id


@dataclass(unsafe_hash=True)
class View(WorldEntity): ...


@dataclass
class Drawer(View):
    handle: Handle
    container: Container
    correct: Optional[bool] = None

    def __hash__(self):
        return hash((self.__class__.__name__, self.handle, self.container))

    def __eq__(self, other):
        if not isinstance(other, Drawer):
            return False
        return (
            self.handle == other.handle
            and self.container == other.container
            and self.world == other.world
        )


@dataclass
class Cabinet(View):
    container: Container
    drawers: List[Drawer] = field(default_factory=list)

    def __hash__(self):
        return hash((self.__class__.__name__, self.container))

    def __eq__(self, other):
        if not isinstance(other, Cabinet):
            return False
        return (
            self.container == other.container
            and self.drawers == other.drawers
            and self.world == other.world
        )

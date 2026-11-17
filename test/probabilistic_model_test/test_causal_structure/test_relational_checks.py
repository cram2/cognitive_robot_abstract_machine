from dataclasses import dataclass

import numpy as np
import pandas as pd

from probabilistic_model.causal_structure.relational_checks import (
    ExampleToPartCheck,
    PartAttributeAdjacency,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    PartTrainingTable,
    RelationalTrainingTables,
)

# %% tables of instances with parts


@dataclass
class WettablePart:
    """
    A part that is wet or dry.
    """

    wet: int
    """
    Whether the part is wet.
    """


def room_tables(
    attribute_drives_parts: bool, instance_count: int = 150, seed: int = 0
) -> RelationalTrainingTables:
    """
    :param attribute_drives_parts: Whether the instance's ``warm`` attribute sets its
        parts' ``wet`` attribute directly, not through the count.
    :param instance_count: How many instances to draw.
    :param seed: Seed of the draw.
    :return: One row per instance, with ``warm`` and the count of its parts, and one row
        per part, with the count and ``wet``.
    """
    random_state = np.random.default_rng(seed)
    warm = random_state.integers(0, 2, instance_count)
    part_count = random_state.integers(2, 5, instance_count)
    parts, example_positions = [], []
    for position in range(instance_count):
        wet_rate = 0.8 if attribute_drives_parts and warm[position] else 0.3
        for _ in range(part_count[position]):
            parts.append(WettablePart(int(random_state.random() < wet_rate)))
            example_positions.append(position)
    return RelationalTrainingTables(
        example_rows=pd.DataFrame({"Room.warm": warm, "Room.count()": part_count}),
        part_tables={
            "parts": PartTrainingTable(
                part_instances=parts,
                rows=pd.DataFrame(
                    {
                        "Room.count()": part_count[example_positions],
                        "wet": [part.wet for part in parts],
                    }
                ),
                aggregation_names=["Room.count()"],
                example_positions=example_positions,
            )
        },
    )


def room_tables_with_noisy_copies(seed: int = 0) -> RelationalTrainingTables:
    """
    :param seed: Seed of the draw.
    :return: The tables of :func:`room_tables`, with ``warm`` setting the parts' ``wet``
        attribute directly, plus ``sunny``, which reads ``warm`` wrong a quarter of the
        time, and each part's ``dripping``, which reads ``wet`` wrong a tenth of the time.
    """
    tables = room_tables(attribute_drives_parts=True, instance_count=300, seed=seed)
    random_state = np.random.default_rng(seed + 1)
    warm = tables.example_rows["Room.warm"].to_numpy()
    tables.example_rows.insert(
        1, "Room.sunny", warm ^ (random_state.random(len(warm)) < 0.25)
    )
    part_rows = tables.part_tables["parts"].rows
    part_rows["dripping"] = part_rows["wet"] ^ (
        random_state.random(len(part_rows)) < 0.1
    )
    return tables


# %% the check


def test_attribute_driving_parts_directly_is_a_dependence():
    [dependence] = ExampleToPartCheck().dependences(
        room_tables(attribute_drives_parts=True)
    )

    assert (
        dependence.example_attribute,
        dependence.part_attribute,
        dependence.part_collection,
    ) == (
        "Room.warm",
        "wet",
        "parts",
    )


def test_attribute_independent_of_parts_is_no_dependence():
    assert (
        ExampleToPartCheck().dependences(room_tables(attribute_drives_parts=False))
        == []
    )


# %% the dependences that remain given another attribute


def test_noisy_copies_depend_on_the_parts_before_conditioning():
    tables = room_tables_with_noisy_copies()

    assert {
        (dependence.example_attribute, dependence.part_attribute)
        for dependence in ExampleToPartCheck().dependences(tables)
    } == {
        ("Room.warm", "wet"),
        ("Room.warm", "dripping"),
        ("Room.sunny", "wet"),
        ("Room.sunny", "dripping"),
    }


def test_copy_of_the_driving_attribute_is_dropped_given_it():
    tables = room_tables_with_noisy_copies()

    dependences = ExampleToPartCheck().dependences_given_another_attribute(
        tables.example_rows,
        "parts",
        tables.part_tables["parts"],
        PartAttributeAdjacency(),
    )

    assert {dependence.example_attribute for dependence in dependences} == {"Room.warm"}


def test_part_attribute_reached_through_an_adjacent_one_is_dropped_given_it():
    tables = room_tables_with_noisy_copies()

    dependences = ExampleToPartCheck().dependences_given_another_attribute(
        tables.example_rows,
        "parts",
        tables.part_tables["parts"],
        PartAttributeAdjacency({"wet": {"dripping"}, "dripping": {"wet"}}),
    )

    assert [
        (dependence.example_attribute, dependence.part_attribute)
        for dependence in dependences
    ] == [("Room.warm", "wet")]

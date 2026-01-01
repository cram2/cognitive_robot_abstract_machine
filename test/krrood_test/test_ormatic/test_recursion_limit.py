import sys

from krrood.ormatic.dao import to_dao
from ..dataset.example_classes import Person


def test_deep_person_chain_to_dao():
    """
    Test if to_dao hits recursion limits
    """
    python_recursion_limit = sys.getrecursionlimit()
    limit = 1500
    sys.setrecursionlimit(limit)

    # Create a chain of 2000 persons.
    # P0 knows [P1], P1 knows [P2], ..., P1999 knows []

    depth = 2000
    current_person = Person(name=f"Person {depth-1}")
    for i in range(depth - 2, -1, -1):
        p = Person(name=f"Person {i}", knows=[current_person])
        current_person = p

    root_person = current_person

    try:
        # This should not raise RecursionError
        dao = to_dao(root_person)
    finally:
        sys.setrecursionlimit(python_recursion_limit)

    assert dao.name == "Person 0"
    assert len(dao.knows) == 1
    assert dao.knows[0].name == "Person 1"


def test_deep_person_chain_from_dao():
    """
    Test if from_dao hits recursion limits
    """
    python_recursion_limit = sys.getrecursionlimit()
    limit = 1500
    sys.setrecursionlimit(limit)

    depth = 2000
    current_person = Person(name=f"Person {depth-1}")
    for i in range(depth - 2, -1, -1):
        p = Person(name=f"Person {i}", knows=[current_person])
        current_person = p

    root_person = current_person
    try:
        dao = to_dao(root_person)

        # This should not raise RecursionError
        reconstructed_person = dao.from_dao()

    finally:
        sys.setrecursionlimit(python_recursion_limit)

    assert reconstructed_person.name == "Person 0"
    assert len(reconstructed_person.knows) == 1
    assert reconstructed_person.knows[0].name == "Person 1"

    # Verify the depth
    curr = reconstructed_person
    count = 0
    while curr.knows:
        curr = curr.knows[0]
        count += 1
    assert count == depth - 1

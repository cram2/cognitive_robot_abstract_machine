from krrood.ormatic.dao import HasGeneric
from krrood.ormatic.exceptions import NoGenericError
from ..dataset.example_classes import Position
from ..dataset.ormatic_interface import PositionDAO


class NoGeneric(HasGeneric): ...


def test_has_generic():
    og = PositionDAO.original_class()
    assert og is Position


def test_no_generic():
    try:
        og = NoGeneric.original_class()
    except NoGenericError as e:
        assert e.clazz is NoGeneric

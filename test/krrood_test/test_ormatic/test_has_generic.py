from krrood.ormatic.dao import HasGeneric
from krrood.ormatic.exceptions import NoGenericError
from ..dataset.example_classes import KRROODPosition
from ..dataset.ormatic_interface import KRROODPositionDAO


class NoGeneric(HasGeneric): ...


def test_has_generic():
    og = KRROODPositionDAO.original_class()
    assert og is KRROODPosition


def test_no_generic():
    try:
        og = NoGeneric.original_class()
    except NoGenericError as e:
        assert e.clazz is NoGeneric

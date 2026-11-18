"""
Reading the generic argument a class binds.
"""

from krrood.patterns.exceptions import NoGenericError
from krrood.patterns.has_generic import HasGeneric
from ..dataset.example_classes import KRROODPosition


class NoGeneric(HasGeneric): ...


def test_has_generic():
    from ..dataset.ormatic_interface import KRROODPositionDAO

    og = KRROODPositionDAO.original_class()
    assert og is KRROODPosition


def test_no_generic():
    try:
        og = NoGeneric.original_class()
    except NoGenericError as e:
        assert e.clazz is NoGeneric

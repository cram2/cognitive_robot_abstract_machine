from ..dataset.example_classes import ContainerGeneration, ItemWithBackreference
from krrood.ormatic.dao import to_dao
from .test_interface import (
    KRROODPosition,
    KRROODPosition4D,
    KRROODPose,
    KRROODOrientation,
)
from ..dataset.ormatic_interface import (
    KRROODPositionDAO,
    KRROODPoseDAO,
    KRROODPosition4DDAO,
)


def test_repr_includes_scalar_columns_only():
    # Given a simple object with scalar columns only
    p = KRROODPosition(1.0, 2.0, 3.0)
    dao: KRROODPositionDAO = KRROODPositionDAO.to_dao(p)

    s = repr(dao)


def test_repr_does_not_recurse_into_relationships():
    # Given an object that has relationships
    p = KRROODPosition(1.0, 2.0, 3.0)
    o = KRROODOrientation(1.0, 2.0, 3.0, None)
    pose = KRROODPose(p, o)

    dao: KRROODPoseDAO = KRROODPoseDAO.to_dao(pose)

    s = repr(dao)


def test_repr_works_for_inherited_columns():
    # Given an inherited DAO that adds an extra column
    p4d = KRROODPosition4D(1.0, 2.0, 3.0, 4.0)
    dao: KRROODPosition4DDAO = KRROODPosition4DDAO.to_dao(p4d)

    s = repr(dao)


def test_repr_with_cyclic_references():
    i1 = ItemWithBackreference(0)
    i2 = ItemWithBackreference(1)
    container = ContainerGeneration([i1, i2])

    dao = to_dao(container)
    s = repr(dao)

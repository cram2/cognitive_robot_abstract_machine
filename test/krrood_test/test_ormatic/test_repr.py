from ..dataset.example_classes import ContainerGeneration, ItemWithBackreference
from krrood.ormatic.data_access_objects.helper import to_dao
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
    p = KRROODPosition(x=1.0, y=2.0, z=3.0)
    dao: KRROODPositionDAO = to_dao(p)

    s = repr(dao)


def test_repr_does_not_recurse_into_relationships():
    # Given an object that has relationships
    p = KRROODPosition(x=1.0, y=2.0, z=3.0)
    o = KRROODOrientation(x=1.0, y=2.0, z=3.0, w=None)
    pose = KRROODPose(position=p, orientation=o)

    dao: KRROODPoseDAO = to_dao(pose)

    s = repr(dao)


def test_repr_works_for_inherited_columns():
    # Given an inherited DAO that adds an extra column
    p4d = KRROODPosition4D(x=1.0, y=2.0, z=3.0, w=4.0)
    dao: KRROODPosition4DDAO = to_dao(p4d)

    s = repr(dao)


def test_repr_with_cyclic_references():
    i1 = ItemWithBackreference(value=0)
    i2 = ItemWithBackreference(value=1)
    container = ContainerGeneration(items=[i1, i2])

    dao = to_dao(container)
    s = repr(dao)

"""
Reproduction tests for bugs found during the ORMatic package review.
"""

import pytest
from sqlalchemy import select

from krrood.ormatic.data_access_objects.conversion_order import HoldingOrder
from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
from krrood.ormatic.data_access_objects.helper import to_dao, get_dao_class
from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState
from krrood.ormatic.exceptions import ConversionOrderCycle
from krrood.ormatic.ormatic import ORMatic
from krrood.entity_query_language.core.mapped_variable import Attribute
from ..dataset.alternative_mappings_construction_order import (
    BuildFirst,
    BuildFirstAssociation,
    BuildFirstMapping,
    EntryPointMapping,
    HoldsAnEntrypointMapping,
    OneSideOfAHoldingCycleMapping,
    OtherSideOfAHoldingCycleMapping,
    OwnsAHolder,
)
from ..dataset.example_classes import *
from ..dataset.ormatic_interface import *


def test_an_alternative_mapping_is_handed_the_domain_object_it_holds():
    """
    A mapping builds its domain object out of what it holds, so a held mapping is
    converted before the mapping holding it, even where the declared dependencies alone
    would convert the holder first.
    """
    build_first = BuildFirst("first")
    held = EntryPointMapping(build_first, BuildFirstAssociation(build_first))
    holder = HoldsAnEntrypointMapping(held)
    owner = OwnsAHolder(holder)
    state = FromDataAccessObjectState()
    # Inserted in this order, the declared dependencies alone sort the holder first.
    state._build_class_dependencies(
        [BuildFirstMapping, HoldsAnEntrypointMapping, EntryPointMapping]
    )
    state._alternative_mappings_being_referenced[held].append(
        (holder, Attribute(_attribute_name_="entrypoint", _child_=None))
    )
    state._alternative_mappings_being_referenced[holder].append(
        (owner, Attribute(_attribute_name_="holder", _child_=None))
    )

    state.convert_alternative_mappings_to_domain_objects()

    assert owner.holder.entrypoint is state.resolve_alternative_mapping(held)


def test_each_holder_of_the_same_type_is_handed_its_own_held_domain_object():
    """
    The order between two mapping types is decided once for all of their instances, and
    every holder is still handed the domain object of the mapping it holds itself.
    """
    build_first = BuildFirst("first")
    first_held = EntryPointMapping(build_first, BuildFirstAssociation(build_first))
    second_held = EntryPointMapping(build_first, BuildFirstAssociation(build_first))
    first_holder = HoldsAnEntrypointMapping(first_held)
    second_holder = HoldsAnEntrypointMapping(second_held)
    first_owner = OwnsAHolder(first_holder)
    second_owner = OwnsAHolder(second_holder)
    state = FromDataAccessObjectState()
    state._build_class_dependencies(
        [BuildFirstMapping, HoldsAnEntrypointMapping, EntryPointMapping]
    )
    for held, holder in ((first_held, first_holder), (second_held, second_holder)):
        state._alternative_mappings_being_referenced[held].append(
            (holder, Attribute(_attribute_name_="entrypoint", _child_=None))
        )
    for holder, owner in ((first_holder, first_owner), (second_holder, second_owner)):
        state._alternative_mappings_being_referenced[holder].append(
            (owner, Attribute(_attribute_name_="holder", _child_=None))
        )

    state.convert_alternative_mappings_to_domain_objects()

    assert first_owner.holder.entrypoint is state.resolve_alternative_mapping(
        first_held
    )
    assert second_owner.holder.entrypoint is state.resolve_alternative_mapping(
        second_held
    )


def test_mappings_that_hold_each_other_have_no_conversion_order():
    """
    Mappings holding one another leave no order that converts each of them after what it
    holds, so the conversion reports the cycle instead of silently picking a side.
    """
    one_side = OneSideOfAHoldingCycleMapping()
    other_side = OtherSideOfAHoldingCycleMapping(one_side)
    one_side.other_side = other_side
    state = FromDataAccessObjectState()
    state._build_class_dependencies(
        [OneSideOfAHoldingCycleMapping, OtherSideOfAHoldingCycleMapping]
    )
    state._alternative_mappings_being_referenced[one_side].append(
        (other_side, Attribute(_attribute_name_="one_side", _child_=None))
    )
    state._alternative_mappings_being_referenced[other_side].append(
        (one_side, Attribute(_attribute_name_="other_side", _child_=None))
    )

    with pytest.raises(ConversionOrderCycle) as raised:
        state.convert_alternative_mappings_to_domain_objects()

    assert {
        (constraint.earlier, constraint.later) for constraint in raised.value.cycle
    } == {
        (OneSideOfAHoldingCycleMapping, OtherSideOfAHoldingCycleMapping),
        (OtherSideOfAHoldingCycleMapping, OneSideOfAHoldingCycleMapping),
    }
    assert all(
        isinstance(constraint, HoldingOrder) for constraint in raised.value.cycle
    )


def test_a_declared_dependency_decides_where_holding_disagrees():
    """
    A mapping declaring what to wait for is converted after it even where it holds it,
    which is how the author of a pair of mappings holding each other breaks the cycle.
    """
    build_first = BuildFirstMapping("first")
    entrypoint = EntryPointMapping(
        build_first, BuildFirstAssociation(BuildFirst("first"))
    )
    build_first.backreference_to_entrypoint = entrypoint
    state = FromDataAccessObjectState()
    state._build_class_dependencies([BuildFirstMapping, EntryPointMapping])
    state._alternative_mappings_being_referenced[build_first].append(
        (entrypoint, Attribute(_attribute_name_="build_first", _child_=None))
    )
    state._alternative_mappings_being_referenced[entrypoint].append(
        (
            build_first,
            Attribute(_attribute_name_="backreference_to_entrypoint", _child_=None),
        )
    )

    state.convert_alternative_mappings_to_domain_objects()

    assert state.resolve_alternative_mapping(
        entrypoint
    ).build_first is state.resolve_alternative_mapping(build_first)


def test_shared_state_does_not_rerun_post_init(session, database, monkeypatch):
    """
    Converting an already converted root DAO again with a shared state must not re-run
    population and ``__post_init__``.
    """
    container = ContainerGeneration(
        [ItemWithBackreference(10), ItemWithBackreference(20)]
    )
    session.add(to_dao(container))
    session.commit()
    session.expunge_all()

    container_dao = session.scalars(select(ContainerGenerationDAO)).one()

    calls = []
    original_post_init = ContainerGeneration.__post_init__

    def counting_post_init(self):
        calls.append(self)
        original_post_init(self)

    monkeypatch.setattr(ContainerGeneration, "__post_init__", counting_post_init)

    state = FromDataAccessObjectState()
    container_1 = container_dao.from_dao(state)
    container_2 = container_dao.from_dao(state)

    # both conversions return the same instance
    assert container_1 is container_2
    # the container's __post_init__ must have run exactly once
    assert len(calls) == 1


def test_shared_state_converts_alternative_mappings_once(
    session, database, monkeypatch
):
    """
    Converting two root DAOs with a shared state must not duplicate class dependency
    graph nodes nor call ``to_domain_object`` repeatedly for the same alternative
    mapping instance.
    """
    entity = Entity("shared")
    to_state = ToDataAccessObjectState()
    dao_1 = to_dao(AlternativeMappingAggregator([entity], []), to_state)
    dao_2 = to_dao(AlternativeMappingAggregator([entity], []), to_state)
    session.add_all([dao_1, dao_2])
    session.commit()
    session.expunge_all()

    aggregator_daos = session.scalars(select(AlternativeMappingAggregatorDAO)).all()
    assert len(aggregator_daos) == 2

    calls = []
    original_to_domain_object = EntityMapping.to_domain_object

    def counting_to_domain_object(self):
        calls.append(self)
        return original_to_domain_object(self)

    monkeypatch.setattr(EntityMapping, "to_domain_object", counting_to_domain_object)

    state = FromDataAccessObjectState()
    aggregator_1 = aggregator_daos[0].from_dao(state)
    aggregator_2 = aggregator_daos[1].from_dao(state)

    # the shared entity is converted exactly once
    assert len(calls) == 1
    # no duplicated nodes in the class dependency graph
    node_types = list(state._class_dependencies.nodes())
    assert len(node_types) == len(set(node_types))
    # identity of the shared entity is preserved across both conversions
    assert aggregator_1.entities1[0] is aggregator_2.entities1[0]


def test_alternatively_mapped_root_in_cycle_keeps_identity(session, database):
    """
    If the root DAO is alternatively mapped and the object graph cycles back to it, the
    returned domain object must be the same instance the cycle points to.
    """
    backreference = Backreference({1: 1})
    reference = Reference(0, backreference)
    backreference.reference = reference

    session.add(to_dao(backreference))
    session.commit()
    session.expunge_all()

    queried = session.scalars(select(BackreferenceMappingDAO)).one()
    reconstructed = queried.from_dao()

    assert isinstance(reconstructed, Backreference)
    assert reconstructed.reference.backreference is reconstructed


def test_repeated_from_dao_on_alternatively_mapped_dao_with_shared_state_returns_same_object(
    session, database
):
    """
    Two ``from_dao`` calls on the same alternatively mapped DAO with a shared state must
    return the same domain object.
    """
    backreference = Backreference({1: 1})
    reference = Reference(0, backreference)
    backreference.reference = reference

    session.add(to_dao(backreference))
    session.commit()
    session.expunge_all()

    queried = session.scalars(select(BackreferenceMappingDAO)).one()
    state = FromDataAccessObjectState()
    first = queried.from_dao(state)
    second = queried.from_dao(state)
    assert first is second


def test_from_package_includes_alternative_mappings_when_not_ignoring():
    """
    ``ignore_krrood_test_classes=False`` must include alternative mappings instead of
    dropping all of them.
    """
    ormatic = ORMatic.from_package(
        packages=[],
        ormatic_interface_dependencies=[],
        ignored_classes=set(),
        type_mappings={},
        ignore_krrood_test_classes=False,
    )
    assert len(ormatic.alternative_mappings) > 0


def test_empty_collection_is_not_aliased(session, database):
    """
    An empty collection on the domain object must be a fresh container, not the DAO's
    live instrumented collection.
    """
    positions = KRROODPositions([], ["a"])
    session.add(to_dao(positions))
    session.commit()
    session.expunge_all()

    queried = session.scalars(select(KRROODPositionsDAO)).one()
    reconstructed = queried.from_dao()

    assert reconstructed.positions == []
    assert reconstructed.positions is not queried.positions

    # mutating the domain object must not touch the DAO
    reconstructed.positions.append(KRROODPosition(1, 2, 3))
    assert len(queried.positions) == 0


class _LateDomainClass:
    pass


def test_dao_lookup_recovers_after_late_dao_definition():
    """
    A failed DAO lookup must not be cached forever; defining the DAO class afterwards
    must make the lookup succeed.
    """
    assert get_dao_class(_LateDomainClass) is None

    class _LateDomainClassDAO(DataAccessObject[_LateDomainClass]):
        pass

    assert get_dao_class(_LateDomainClass) is _LateDomainClassDAO


def test_bare_generic_resolves_to_unique_concrete_subclass():
    """
    A bare generic domain class maps to an empty polymorphic base DAO while its single
    parametrization carries the data columns.

    A runtime instance of the bare generic must therefore resolve to that unique
    concrete subclass, not the empty base which would silently drop all data.
    """

    class _SingleParameterGeneric(Generic[T]):
        pass

    class _SingleParameterGenericDAO(DataAccessObject[_SingleParameterGeneric]):
        pass

    class _SingleParameterGenericFloatDAO(
        _SingleParameterGenericDAO, DataAccessObject[_SingleParameterGeneric[float]]
    ):
        pass

    assert get_dao_class(_SingleParameterGeneric) is _SingleParameterGenericFloatDAO


def test_bare_generic_with_multiple_parametrizations_stays_on_base():
    """
    When several parametrizations exist the concrete subclass is ambiguous, so the bare
    generic must resolve to the polymorphic base rather than guessing.
    """

    class _MultiParameterGeneric(Generic[T]):
        pass

    class _MultiParameterGenericDAO(DataAccessObject[_MultiParameterGeneric]):
        pass

    class _MultiParameterGenericFloatDAO(
        _MultiParameterGenericDAO, DataAccessObject[_MultiParameterGeneric[float]]
    ):
        pass

    class _MultiParameterGenericIntDAO(
        _MultiParameterGenericDAO, DataAccessObject[_MultiParameterGeneric[int]]
    ):
        pass

    assert get_dao_class(_MultiParameterGeneric) is _MultiParameterGenericDAO

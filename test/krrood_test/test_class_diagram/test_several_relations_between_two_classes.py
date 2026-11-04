import pytest

from krrood.class_diagrams.class_diagram import ClassDiagram, Inheritance

from ..dataset.example_classes import (
    DoubleKRROODPositionAggregator,
    KRROODPosition,
    KRROODPositions,
    KRROODPositionsSubclassWithAnotherKRROODPosition,
)

# %% a diagram where two fields of one class point at the same class


@pytest.fixture
def diagram() -> ClassDiagram:
    return ClassDiagram(
        [
            KRROODPosition,
            KRROODPositions,
            KRROODPositionsSubclassWithAnotherKRROODPosition,
            DoubleKRROODPositionAggregator,
        ]
    )


# %% reading the relations


def test_every_association_between_two_classes_is_keyed(diagram):
    aggregator = diagram.get_wrapped_class(DoubleKRROODPositionAggregator)

    assert diagram.get_association_keys_by_source(include_field_name=True)[
        aggregator.index
    ] == {
        association.get_key(include_field_name=True)
        for association in diagram.associations
        if association.source == aggregator
    }


def test_the_diagram_without_inherited_associations_leaves_the_diagram_unchanged(
    diagram,
):
    relations = list(diagram._dependency_graph.edges())

    diagram.to_subdiagram_without_inherited_associations()

    assert list(diagram._dependency_graph.edges()) == relations


def test_only_inherited_associations_are_left_out(diagram):
    subclass = diagram.get_wrapped_class(
        KRROODPositionsSubclassWithAnotherKRROODPosition
    )
    inherited = {
        association.wrapped_field.name
        for association in diagram.associations
        if association.source == diagram.get_wrapped_class(KRROODPositions)
    }

    subdiagram = diagram.to_subdiagram_without_inherited_associations(
        include_field_name=True
    )

    assert {
        association.wrapped_field.name
        for association in subdiagram.associations
        if association.source == subclass
    } == {
        association.wrapped_field.name
        for association in diagram.associations
        if association.source == subclass
    } - inherited


# %% changing the relations


def test_an_added_relation_is_among_the_outgoing_relations(diagram):
    position = diagram.get_wrapped_class(KRROODPosition)
    diagram.get_out_edges(position)
    relation = Inheritance(
        position, diagram.get_wrapped_class(DoubleKRROODPositionAggregator)
    )

    diagram.add_relation(relation)

    assert relation in diagram.get_out_edges(position)


def test_a_removed_relation_is_no_longer_among_the_outgoing_relations(diagram):
    position = diagram.get_wrapped_class(KRROODPosition)
    relation = Inheritance(
        position, diagram.get_wrapped_class(DoubleKRROODPositionAggregator)
    )
    diagram.add_relation(relation)
    diagram.get_out_edges(position)

    diagram.remove_relation(relation)

    assert relation not in diagram.get_out_edges(position)

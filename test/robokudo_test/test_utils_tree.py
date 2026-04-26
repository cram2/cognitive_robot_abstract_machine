from typing import Any

import py_trees
import pytest
from py_trees.composites import Sequence
from py_trees_ros.trees import BehaviourTree

from robokudo.utils.tree import (
    find_parent_of_type,
    find_children_with_name,
    get_scoped_list_of_names,
    get_scoped_name,
    add_child_to_parent,
    add_children_to_parent,
    find_root,
    fix_parent_relationship_of_childs,
    setup_with_descendants_on_behavior,
    setup_with_descendants_rk,
    behavior_iterate_except_type,
)


class DummyBehaviour(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self._setup = False

    def setup(self, **kwargs: Any):
        self._setup = True

    def update(self):
        return py_trees.common.Status.SUCCESS


class TestUtilsTree:
    def test_find_going_upwards_type(self):
        leaf_behaviour = DummyBehaviour("Behaviour")
        par = py_trees.composites.Parallel(
            "Par", policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        par.add_child(leaf_behaviour)

        top_sequence = py_trees.composites.Sequence("TopSequence", memory=True)
        top_sequence.add_child(par)

        #
        # Test from the leaf
        #
        assert (
            find_parent_of_type(leaf_behaviour, py_trees.composites.Sequence)
            == top_sequence
        )
        assert (
            find_parent_of_type(leaf_behaviour, py_trees.composites.Parallel) == par
        )  # There is no selector. This should return None.
        assert find_parent_of_type(leaf_behaviour, py_trees.composites.Selector) is None

        #
        # Test from the parallel
        #
        assert (
            find_parent_of_type(par, py_trees.composites.Sequence) == top_sequence
        )  # There is no selector. This should return None.
        assert find_parent_of_type(leaf_behaviour, py_trees.composites.Selector) is None

    def test_find_children_with_name(self):
        leaf_behaviour = DummyBehaviour("Behaviour")
        par = py_trees.composites.Parallel(
            "Par", policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        par.add_child(leaf_behaviour)

        top_sequence = py_trees.composites.Sequence("TopSequence", memory=True)
        top_sequence.add_child(par)

        assert find_children_with_name(top_sequence, "Par") == par
        assert find_children_with_name(top_sequence, "Behaviour") == leaf_behaviour
        assert find_children_with_name(top_sequence, "NonExisting") is None
        assert (
            find_children_with_name(top_sequence, "Par", direct_descendants=True) == par
        )
        assert (
            find_children_with_name(top_sequence, "Behaviour", direct_descendants=True)
            is None
        )
        assert (
            find_children_with_name(
                top_sequence, "NonExisting", direct_descendants=True
            )
            is None
        )

    def test_scoped_list_tree(self):
        leaf_behaviour = DummyBehaviour("Behaviour")
        par = py_trees.composites.Parallel(
            "Par", policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        par.add_child(leaf_behaviour)

        top_sequence = py_trees.composites.Sequence("TopSequence", memory=True)
        top_sequence.add_child(par)
        assert get_scoped_list_of_names(
            leaf_behaviour, py_trees.composites.Sequence
        ) == [top_sequence.name, leaf_behaviour.name]

        assert (
            get_scoped_name(leaf_behaviour, py_trees.composites.Sequence)
            == "TopSequence/Behaviour"
        )

    def test_scoped_list_tree_deeper(self):
        leaf_behaviour = DummyBehaviour("Behaviour")
        par = py_trees.composites.Parallel(
            "Par", policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        par.add_child(leaf_behaviour)

        mid_sequence = py_trees.composites.Sequence("MidSequence", memory=True)
        mid_sequence.add_child(par)

        top_sequence = py_trees.composites.Sequence("TopSequence", memory=True)
        top_sequence.add_child(mid_sequence)
        assert get_scoped_list_of_names(
            leaf_behaviour, py_trees.composites.Sequence
        ) == [top_sequence.name, mid_sequence.name, leaf_behaviour.name]
        assert (
            get_scoped_name(leaf_behaviour, py_trees.composites.Sequence)
            == "TopSequence/MidSequence/Behaviour"
        )

    def test_scoped_list_single_behaviour(self):
        leaf_behaviour = DummyBehaviour("Behaviour")
        assert get_scoped_list_of_names(
            leaf_behaviour, py_trees.composites.Sequence
        ) == [leaf_behaviour.name]
        assert (
            get_scoped_name(leaf_behaviour, py_trees.composites.Sequence) == "Behaviour"
        )

    def test_add_child_to_parent(self):
        par2 = py_trees.composites.Parallel(
            "Par2", policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        par1 = py_trees.composites.Parallel(
            "Par1",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[par2],
        )

        leaf_behaviour = DummyBehaviour("Behaviour")

        add_child_to_parent(par2, leaf_behaviour)

        assert len(par1.children) == 1
        assert len(par2.children) == 1
        assert par2.children[0] == leaf_behaviour

    def test_add_child_to_parent_invalid_child(self):
        par2 = py_trees.composites.Parallel(
            "Par2", policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        par1 = py_trees.composites.Parallel(
            "Par1",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[par2],
        )

        leaf_behaviour = "NotABehaviour"

        assert pytest.raises(TypeError, add_child_to_parent, par2, leaf_behaviour)

    def test_add_children_to_parent(self):
        par2 = py_trees.composites.Parallel(
            "Par2", policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        par1 = py_trees.composites.Parallel(
            "Par1",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[par2],
        )

        leaf_behaviour1 = DummyBehaviour("Behaviour1")
        leaf_behaviour2 = DummyBehaviour("Behaviour2")
        leaf_behaviour3 = DummyBehaviour("Behaviour3")

        add_children_to_parent(
            par2, [leaf_behaviour1, leaf_behaviour2, leaf_behaviour3]
        )

        assert len(par1.children) == 1
        assert len(par2.children) == 3

    def test_find_root(self):
        leaf_behaviour1 = DummyBehaviour("Behaviour1")
        leaf_behaviour2 = DummyBehaviour("Behaviour2")
        leaf_behaviour3 = DummyBehaviour("Behaviour3")
        children = [leaf_behaviour1, leaf_behaviour2, leaf_behaviour3]

        par2 = py_trees.composites.Parallel(
            "Par2",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=children,
        )
        par1 = py_trees.composites.Parallel(
            "Par1",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[par2],
        )

        for child in children:
            root = find_root(child)
            assert root == par1

    def test_fix_parent_relationship_of_childs(self):
        # Build first pipeline with children
        leaf_behaviour1 = DummyBehaviour("Behaviour1")
        leaf_behaviour2 = DummyBehaviour("Behaviour2")
        leaf_behaviour3 = DummyBehaviour("Behaviour3")
        children = [leaf_behaviour1, leaf_behaviour2, leaf_behaviour3]

        par2 = py_trees.composites.Parallel(
            "Par2",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=children,
        )

        par1 = py_trees.composites.Parallel(
            "Par1",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[par2],
        )

        # Build second pipeline with children
        seq = py_trees.composites.Sequence("Seq1", memory=True)
        add_children_to_parent(seq, children)

        for child in children:
            assert child.parent.name == seq.name

        fix_parent_relationship_of_childs(par1)

        for child in children:
            assert child.parent.name == par2.name

    def test_setup_with_descendants_on_behaviour(self):
        leaf_behaviour0 = DummyBehaviour("Behaviour0")

        leaf_behaviour1 = DummyBehaviour("Behaviour1")
        leaf_behaviour2 = DummyBehaviour("Behaviour2")
        leaf_behaviour3 = DummyBehaviour("Behaviour3")
        children = [leaf_behaviour1, leaf_behaviour2, leaf_behaviour3]

        par2 = py_trees.composites.Parallel(
            "Par2",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=children,
        )
        par1 = py_trees.composites.Parallel(
            "Par1",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[par2, leaf_behaviour0],
        )

        setup_with_descendants_on_behavior(par2)

        for child in children:
            assert child._setup == True

        assert leaf_behaviour0._setup == False, "only descendants should be setup"

    def test_setup_with_descendants_rk(self):
        leaf_behaviour0 = DummyBehaviour("Behaviour0")

        leaf_behaviour1 = DummyBehaviour("Behaviour1")
        leaf_behaviour2 = DummyBehaviour("Behaviour2")
        leaf_behaviour3 = DummyBehaviour("Behaviour3")
        children = [leaf_behaviour1, leaf_behaviour2, leaf_behaviour3]

        par2 = py_trees.composites.Parallel(
            "Par2",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=children,
        )
        par1 = py_trees.composites.Parallel(
            "Par1",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[par2, leaf_behaviour0],
        )
        tree = BehaviourTree(par1)

        setup_with_descendants_rk(tree)

        for child in children:
            assert child._setup == True
        assert leaf_behaviour0._setup == True

    def test_behaviour_iterate_except_type_basic(self):
        leaf_behaviour0 = DummyBehaviour("Behaviour0")

        leaf_behaviour1 = DummyBehaviour("Behaviour1")
        leaf_behaviour2 = DummyBehaviour("Behaviour2")
        leaf_behaviour3 = DummyBehaviour("Behaviour3")
        children = [leaf_behaviour1, leaf_behaviour2, leaf_behaviour3]

        par2 = py_trees.composites.Parallel(
            "Par2",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=children,
        )
        par1 = py_trees.composites.Parallel(
            "Par1",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[par2, leaf_behaviour0],
        )

        tree_children = [
            child.name
            for child in behavior_iterate_except_type(tree=par1, child_type=Sequence)
        ]

        assert par1.name not in tree_children

        assert par2.name in tree_children
        assert leaf_behaviour0.name in tree_children

        for child in children:
            assert child.name in tree_children

    def test_behaviour_iterate_except_type_with_tree(self):
        leaf_behaviour0 = DummyBehaviour("Behaviour0")

        leaf_behaviour1 = DummyBehaviour("Behaviour1")
        leaf_behaviour2 = DummyBehaviour("Behaviour2")
        leaf_behaviour3 = DummyBehaviour("Behaviour3")
        children = [leaf_behaviour1, leaf_behaviour2, leaf_behaviour3]

        par2 = py_trees.composites.Parallel(
            "Par2",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=children,
        )
        par1 = py_trees.composites.Parallel(
            "Par1",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[par2, leaf_behaviour0],
        )

        tree_children = [
            child.name
            for child in behavior_iterate_except_type(
                tree=par1, child_type=Sequence, include_tree=True
            )
        ]

        assert par1.name in tree_children
        assert par2.name in tree_children
        assert leaf_behaviour0.name in tree_children

        for child in children:
            assert child.name in tree_children

    def test_behaviour_iterate_except_type_direct_descendants(self):
        leaf_behaviour0 = DummyBehaviour("Behaviour0")

        leaf_behaviour1 = DummyBehaviour("Behaviour1")
        leaf_behaviour2 = DummyBehaviour("Behaviour2")
        leaf_behaviour3 = DummyBehaviour("Behaviour3")
        children = [leaf_behaviour1, leaf_behaviour2, leaf_behaviour3]

        par2 = py_trees.composites.Parallel(
            "Par2",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=children,
        )
        par1 = py_trees.composites.Parallel(
            "Par1",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[par2, leaf_behaviour0],
        )

        tree_children = [
            child.name
            for child in behavior_iterate_except_type(
                tree=par1, child_type=Sequence, direct_descendants=True
            )
        ]

        assert par1.name not in tree_children

        for child in children:
            assert child.name not in tree_children

        assert par2.name in tree_children
        assert leaf_behaviour0.name in tree_children

    def test_behaviour_iterate_except_type_exclude_type(self):
        leaf_behaviour0 = DummyBehaviour("Behaviour0")

        leaf_behaviour1 = DummyBehaviour("Behaviour1")
        leaf_behaviour2 = DummyBehaviour("Behaviour2")
        leaf_behaviour3 = DummyBehaviour("Behaviour3")
        children = [leaf_behaviour1, leaf_behaviour2, leaf_behaviour3]

        seq1 = py_trees.composites.Sequence("Seq1", memory=True, children=children)
        par1 = py_trees.composites.Parallel(
            "Par1",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[seq1, leaf_behaviour0],
        )

        tree_children = [
            child.name
            for child in behavior_iterate_except_type(tree=par1, child_type=Sequence)
        ]

        assert par1.name not in tree_children
        assert seq1.name not in tree_children

        for child in children:
            assert child.name not in tree_children

        assert leaf_behaviour0.name in tree_children

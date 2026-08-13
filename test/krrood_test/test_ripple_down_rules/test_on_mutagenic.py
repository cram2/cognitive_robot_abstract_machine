import os
from dataclasses import dataclass
from enum import Enum
from unittest import skip

from typing_extensions import List, Callable, Optional

from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.experts import Human
from krrood.ripple_down_rules.rdr import SingleClassRDR, GeneralRDR


class Element(str, Enum):
    C = "c"
    H = "h"
    O = "o"
    N = "n"
    F = "f"
    B = "b"
    I = "i"

    def __repr__(self):
        return self.name


@dataclass
class Atom:
    element: Element
    type: int
    charge: float

    def __hash__(self):
        return id(self)


@dataclass
class Bond:
    atom1: Atom
    atom2: Atom
    type: int

    def __hash__(self):
        return id(self)


@dataclass
class Molecule:
    ind1: int
    inda: int
    logp: float
    lumo: float
    mutagenic: bool

    atoms: List[Atom]
    bonds: List[Bond]

    def __hash__(self):
        return id(self)


def make_molecule_1() -> Molecule:
    atoms = [
        Atom(element=Element.C, type=22, charge=-0.117),
        Atom(element=Element.H, type=3, charge=0.142),
        Atom(element=Element.C, type=27, charge=-0.087),
        Atom(element=Element.C, type=27, charge=0.013),
        Atom(element=Element.C, type=22, charge=-0.117),
        Atom(element=Element.C, type=22, charge=-0.117),
        Atom(element=Element.H, type=3, charge=0.143),
        Atom(element=Element.H, type=3, charge=0.143),
        Atom(element=Element.C, type=22, charge=-0.117),
        Atom(element=Element.C, type=22, charge=-0.117),
        Atom(element=Element.C, type=22, charge=-0.117),
        Atom(element=Element.C, type=22, charge=-0.117),
        Atom(element=Element.C, type=22, charge=-0.117),
        Atom(element=Element.H, type=3, charge=0.142),
        Atom(element=Element.H, type=3, charge=0.143),
        Atom(element=Element.H, type=3, charge=0.142),
        Atom(element=Element.N, type=38, charge=0.812),
        Atom(element=Element.O, type=40, charge=-0.388),
        Atom(element=Element.O, type=40, charge=-0.388),
        Atom(element=Element.C, type=22, charge=-0.117),
        Atom(element=Element.C, type=195, charge=-0.087),
        Atom(element=Element.C, type=195, charge=0.013),
        Atom(element=Element.C, type=22, charge=-0.117),
        Atom(element=Element.H, type=3, charge=0.142),
        Atom(element=Element.H, type=3, charge=0.143),
        Atom(element=Element.H, type=3, charge=0.142),
    ]
    bonds = [
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[6], type=1),
        Bond(atom1=atoms[16], atom2=atoms[17], type=2),
        Bond(atom1=atoms[0], atom2=atoms[6], type=1),
        Bond(atom1=atoms[21], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[20], type=7),
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[3], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[6], type=1),
        Bond(atom1=atoms[16], atom2=atoms[0], type=1),
        Bond(atom1=atoms[16], atom2=atoms[17], type=2),
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[20], atom2=atoms[21], type=7),
        Bond(atom1=atoms[20], atom2=atoms[2], type=7),
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[0], atom2=atoms[21], type=7),
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[2], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[3], type=7),
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[6], type=1),
        Bond(atom1=atoms[2], atom2=atoms[3], type=7),
    ]

    molecule = Molecule(
        ind1=1, inda=0, logp=4.23, lumo=-1.246, mutagenic=True, atoms=atoms, bonds=bonds
    )
    return molecule


def make_molecule_2() -> Molecule:
    atoms = [
        Atom(element=Element.C, type=22, charge=-0.128),
        Atom(element=Element.H, type=3, charge=0.132),
        Atom(element=Element.C, type=29, charge=0.002),
        Atom(element=Element.C, type=22, charge=-0.128),
        Atom(element=Element.C, type=22, charge=-0.128),
        Atom(element=Element.C, type=22, charge=-0.128),
        Atom(element=Element.C, type=22, charge=0.202),
        Atom(element=Element.C, type=22, charge=-0.128),
        Atom(element=Element.H, type=3, charge=0.132),
        Atom(element=Element.H, type=3, charge=0.132),
        Atom(element=Element.H, type=3, charge=0.132),
        Atom(element=Element.C, type=22, charge=-0.128),
        Atom(element=Element.H, type=3, charge=0.132),
        Atom(element=Element.N, type=32, charge=-0.779),
        Atom(element=Element.N, type=38, charge=0.801),
        Atom(element=Element.O, type=40, charge=-0.398),
        Atom(element=Element.O, type=40, charge=-0.398),
        Atom(element=Element.H, type=1, charge=0.332),
        Atom(element=Element.H, type=1, charge=0.332),
        Atom(element=Element.C, type=22, charge=-0.128),
        Atom(element=Element.C, type=29, charge=0.002),
        Atom(element=Element.C, type=22, charge=-0.128),
        Atom(element=Element.C, type=22, charge=-0.128),
        Atom(element=Element.H, type=3, charge=0.132),
        Atom(element=Element.H, type=3, charge=0.132),
        Atom(element=Element.H, type=3, charge=0.132),
    ]

    bonds = [
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[14], atom2=atoms[15], type=2),
        Bond(atom1=atoms[6], atom2=atoms[13], type=1),
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[6], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[2], atom2=atoms[2], type=1),
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[0], atom2=atoms[2], type=7),
        Bond(atom1=atoms[13], atom2=atoms[17], type=1),
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[0], atom2=atoms[0], type=7),
        Bond(atom1=atoms[0], atom2=atoms[14], type=1),
        Bond(atom1=atoms[0], atom2=atoms[2], type=7),
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[2], atom2=atoms[0], type=7),
        Bond(atom1=atoms[13], atom2=atoms[17], type=1),
        Bond(atom1=atoms[14], atom2=atoms[15], type=2),
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[0], atom2=atoms[1], type=1),
        Bond(atom1=atoms[0], atom2=atoms[6], type=7),
        Bond(atom1=atoms[2], atom2=atoms[0], type=7),
    ]

    molecule = Molecule(
        ind1=0,
        inda=0,
        logp=2.68,
        lumo=-1.034,
        mutagenic=False,
        atoms=atoms,
        bonds=bonds,
    )
    return molecule


def get_two_molecules_model(
    draw_tree=False,
    load_answers=True,
    save_answers=False,
    filename=os.path.join(
        os.path.dirname(__file__), "test_expert_answers", "mutagenic_expert_answers"
    ),
    scenario: Optional[Callable] = None,
) -> SingleClassRDR:
    expert = Human(use_loaded_answers=load_answers)
    if load_answers:
        expert.load_answers(filename)

    molecule_1 = make_molecule_1()
    molecule_2 = make_molecule_2()
    case_queries = [
        CaseQuery(
            molecule_1,
            "mutagenic",
            (bool,),
            True,
            _target=molecule_1.mutagenic,
            case_factory=make_molecule_1,
        ),
        CaseQuery(
            molecule_2,
            "mutagenic",
            (bool,),
            True,
            _target=molecule_2.mutagenic,
            case_factory=make_molecule_2,
        ),
    ]

    rdr = SingleClassRDR()
    rdr.fit(case_queries, expert=expert, animate_tree=draw_tree, scenario=scenario)

    for case_query in case_queries:
        r = rdr.classify(case_query.case)
        if isinstance(rdr, GeneralRDR):
            assert r[case_query.attribute_name] == case_query.target_value
        else:
            assert r == case_query.target_value

    if save_answers:
        cwd = os.path.dirname(__file__)
        file = os.path.join(cwd, filename)
        expert.save_answers(file)

    return rdr


def test_two_molecules():
    rdr = get_two_molecules_model(
        draw_tree=False,
        load_answers=True,
        save_answers=False,
        scenario=test_two_molecules,
    )


def test_serialize_two_molecules_model():
    rdr = get_two_molecules_model(scenario=test_serialize_two_molecules_model)
    filename = os.path.join(
        os.path.dirname(__file__), "test_results", "two_molecules_model"
    )
    rdr.to_json_file(filename)
    loaded_rdr = type(rdr).from_json_file(filename)
    assert rdr.classify(make_molecule_1()) == loaded_rdr.classify(make_molecule_1())
    assert rdr.classify(make_molecule_2()) == loaded_rdr.classify(make_molecule_2())


def test_write_two_molecules_model_to_python():
    rdr = get_two_molecules_model(scenario=test_write_two_molecules_model_to_python)
    filename = os.path.join(
        os.path.dirname(__file__), "test_generated_rdrs", "two_molecules"
    )
    rdr._write_to_python(filename)
    loaded_rdr = rdr.get_rdr_classifier_from_python_file(filename)
    assert rdr.classify(make_molecule_1()) == loaded_rdr(make_molecule_1())
    assert rdr.classify(make_molecule_2()) == loaded_rdr(make_molecule_2())


if __name__ == "__main__":
    # test_two_molecules()
    test_write_two_molecules_model_to_python()

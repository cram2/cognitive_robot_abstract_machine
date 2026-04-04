import sys

import pytest
from typing_extensions import Type

try:
    from PyQt6.QtWidgets import QApplication
    from krrood.ripple_down_rules.user_interface.gui import RDRCaseViewer
except ImportError as e:
    QApplication = None
    RDRCaseViewer = None

from .conf.world.handles_and_containers import HandlesAndContainersWorld
from .datasets import *
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.experts import Human
from krrood.ripple_down_rules.helpers import is_matching
from krrood.ripple_down_rules.rdr import GeneralRDR

app: Optional[QApplication] = None
viewer: Optional[RDRCaseViewer] = None
use_gui: bool = False

if RDRCaseViewer is not None and QApplication is not None and use_gui:
    app = QApplication(sys.argv)
    viewer = RDRCaseViewer(save_dir="./test_generated_rdrs")


def pytest_generate_tests(metafunc):

    if metafunc.definition.originalname == "test_should_i_ask_the_expert_for_a_target":
        all_cases, all_targets = load_zoo_dataset("./test_results/zoo")

        possible_case_queries = [
            CaseQuery(all_cases[0], "species", (Species,), True),
            CaseQuery(all_cases[1], "habitat", (Habitat,), False),
        ]
        metafunc.parametrize("case_query", possible_case_queries)

        possible_conclusions = [
            [],
            None,
            True,
            False,
            set(),
            {},
            {"species": Species.mammal},
            Species.mammal,
            Habitat.land,
            {"habitat": Habitat.water},
            [Habitat.water, Habitat.land],
            {"species": Species.fish, "habitat": Habitat.water},
        ]
        metafunc.parametrize("conclusions", possible_conclusions)

        possible_update_existing = [True, False]
        metafunc.parametrize("update_existing", possible_update_existing)


@pytest.fixture
def handles_and_containers_world() -> World:
    return HandlesAndContainersWorld().create()


@pytest.fixture
def drawer_case_queries(handles_and_containers_world) -> List[CaseQuery]:
    all_possible_drawers = []
    world = handles_and_containers_world
    for handle in [body for body in world.bodies if isinstance(body, Handle)]:
        for container in [body for body in world.bodies if isinstance(body, Container)]:
            view = Drawer(handle, container, world=world)
            all_possible_drawers.append(view)
    case_queries = [
        CaseQuery(possible_drawer, "correct", (bool,), True, default_value=False)
        for possible_drawer in all_possible_drawers
    ]
    return case_queries


@pytest.fixture
def view_rdr(
    handles_and_containers_world,
    views=(Drawer, Cabinet),
    use_loaded_answers: bool = True,
    save_answers: bool = False,
    append: bool = False,
) -> GeneralRDR:
    world = handles_and_containers_world
    expert = Human(use_loaded_answers=use_loaded_answers, append=append)
    filename = os.path.join(
        os.path.dirname(__file__), "test_expert_answers", "view_rdr_expert_answers_fit"
    )
    if use_loaded_answers:
        expert.load_answers(filename)
    rdr = GeneralRDR()
    for view in views:
        rdr.fit_case(CaseQuery(world, "views", (view,), False), expert=expert)
    if save_answers:
        expert.save_answers(filename)

    found_views = rdr.classify(world)
    for view in views:
        assert len([v for v in found_views["views"] if isinstance(v, view)]) > 0

    return rdr


@pytest.fixture
def view_case_query(world: World, view_type: Type[View]) -> CaseQuery:
    """
    Create a CaseQuery for the given view type in the provided world.
    """
    return CaseQuery(world, "views", (view_type,), False)


@pytest.fixture
def drawer_rdr(
    drawer_case_queries, use_loaded_answers: bool = True, save_answers: bool = False
):
    expert = Human(use_loaded_answers=use_loaded_answers)
    filename = os.path.join(
        os.path.dirname(__file__),
        "test_expert_answers",
        "correct_drawer_rdr_expert_answers_fit",
    )
    if use_loaded_answers:
        expert.load_answers(filename)
    rdr = GeneralRDR()
    rdr.fit(drawer_case_queries, expert=expert, animate_tree=False)
    if save_answers:
        expert.save_answers(filename)
    for case_query in drawer_case_queries:
        assert is_matching(rdr.classify, case_query)
    return rdr

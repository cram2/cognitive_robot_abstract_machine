import pytest

from .conf.world.handles_and_containers import HandlesAndContainersWorld
from .datasets import *
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.experts import Human
from krrood.ripple_down_rules.helpers import is_matching
from krrood.ripple_down_rules.rdr import GeneralRDR


def pytest_generate_tests(metafunc):

    if metafunc.definition.originalname == "test_should_i_ask_the_expert_for_a_target":
        all_cases, all_targets = load_zoo_dataset("./test_results/zoo")

        possible_case_queries = [
            CaseQuery(original_case=all_cases[0], attribute_name="species", _attribute_types=(Species,), mutually_exclusive=True),
            CaseQuery(original_case=all_cases[1], attribute_name="habitat", _attribute_types=(Habitat,), mutually_exclusive=False),
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
            view = Drawer(handle=handle, container=container, world=world)
            all_possible_drawers.append(view)
    case_queries = [
        CaseQuery(original_case=possible_drawer, attribute_name="correct", _attribute_types=(bool,), mutually_exclusive=True, default_value=False)
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
        expert.load_answers(path=filename)
    rdr = GeneralRDR()
    for view in views:
        rdr.fit_case(case_query=CaseQuery(original_case=world, attribute_name="views", _attribute_types=(view,), mutually_exclusive=False), expert=expert)
    if save_answers:
        expert.save_answers(path=filename)

    found_views = rdr.classify(case=world)
    for view in views:
        assert len([v for v in found_views["views"] if isinstance(v, view)]) > 0

    return rdr


@pytest.fixture
def view_case_query(world: World, view_type: type[View]) -> CaseQuery:
    """
    Create a CaseQuery for the given view type in the provided world.
    """
    return CaseQuery(original_case=world, attribute_name="views", _attribute_types=(view_type,), mutually_exclusive=False)


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
        expert.load_answers(path=filename)
    rdr = GeneralRDR()
    rdr.fit(case_queries=drawer_case_queries, expert=expert, animate_tree=False)
    if save_answers:
        expert.save_answers(path=filename)
    for case_query in drawer_case_queries:
        assert is_matching(lambda case: rdr.classify(case=case), case_query)
    return rdr

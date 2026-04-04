import inspect
import os

from typing_extensions import List, Any, Tuple, Type, Callable, Optional

from test.krrood_test.test_ripple_down_rules.datasets import (
    Species,
    Habitat,
    load_zoo_cases,
)
from krrood.ripple_down_rules.datastructures.case import Case
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.datastructures.enums import Category
from krrood.ripple_down_rules.experts import Human
from krrood.ripple_down_rules.rdr import MultiClassRDR, SingleClassRDR, GeneralRDR
from krrood.ripple_down_rules.utils import make_set


def get_fit_scrdr(
    cases: List[Any],
    targets: List[Any],
    attribute_name: str = "species",
    attribute_type: Type = Species,
    expert_answers_dir: str = "test_expert_answers",
    expert_answers_file: str = "scrdr_expert_answers_fit",
    draw_tree: bool = False,
    load_answers: bool = True,
    save_answers: bool = False,
    update_existing_rules: bool = True,
    case_factory: Callable = load_zoo_cases,
    scenario: Optional[Callable] = None,
) -> Tuple[SingleClassRDR, List[CaseQuery]]:
    filename = os.path.join(
        os.path.dirname(__file__), "..", expert_answers_dir, expert_answers_file
    )
    expert = Human(use_loaded_answers=load_answers, answers_save_path=filename)
    if load_answers:
        expert.load_answers(filename)

    targets = [None for _ in cases] if targets is None or len(targets) == 0 else targets
    scrdr = SingleClassRDR()
    case_queries = [
        CaseQuery(
            case,
            attribute_name,
            (attribute_type,),
            True,
            _target=target,
            case_factory=case_factory,
            case_factory_idx=i,
            scenario=scenario,
        )
        for i, (case, target) in enumerate(zip(cases, targets))
    ]
    scrdr.fit(
        case_queries,
        expert=expert,
        animate_tree=draw_tree,
        update_existing_rules=update_existing_rules,
    )
    if save_answers:
        expert.save_answers(filename)
    for case_query in case_queries:
        cat = scrdr.classify(case_query.case)
        assert cat == case_query.target_value
    return scrdr, case_queries


def get_fit_mcrdr(
    cases: List[Any],
    targets: List[Any],
    attribute_name: str = "species",
    attribute_type: Type = Species,
    mutually_exclusive: bool = True,
    expert_answers_dir: str = "test_expert_answers",
    expert_answers_file: str = "mcrdr_expert_answers_stop_only_fit",
    draw_tree: bool = False,
    load_answers: bool = True,
    save_answers: bool = False,
    update_existing_rules: bool = True,
    case_factory: Callable = load_zoo_cases,
    scenario: Optional[Callable] = None,
) -> MultiClassRDR:
    filename = os.path.join(
        os.path.dirname(__file__), "..", expert_answers_dir, expert_answers_file
    )
    expert = Human(use_loaded_answers=load_answers, answers_save_path=filename)
    if load_answers:
        expert.load_answers(filename)
    targets = [None for _ in cases] if targets is None or len(targets) == 0 else targets
    mcrdr = MultiClassRDR()
    case_queries = [
        CaseQuery(
            case,
            attribute_name,
            (attribute_type,),
            mutually_exclusive,
            _target=target,
            case_factory=case_factory,
            case_factory_idx=i,
            scenario=scenario,
        )
        for i, (case, target) in enumerate(zip(cases, targets))
    ]
    mcrdr.fit(
        case_queries,
        expert=expert,
        animate_tree=draw_tree,
        update_existing_rules=update_existing_rules,
    )
    if save_answers:
        expert.save_answers(filename)
    for case_query in case_queries:
        cat = mcrdr.classify(case_query.case)
        assert make_set(cat) == make_set(case_query.target_value)
    return mcrdr


def get_fit_grdr(
    cases: List[Any],
    targets: List[Any],
    expert_answers_dir: str = "test_expert_answers",
    expert_answers_file: str = "grdr_expert_answers_fit",
    draw_tree: bool = False,
    load_answers: bool = True,
    save_answers: bool = False,
    append: bool = False,
    no_targets: bool = False,
    update_existing_rules: bool = True,
    case_factory: Callable = load_zoo_cases,
    scenario: Optional[Callable] = None,
) -> Tuple[GeneralRDR, List[dict]]:
    filename = os.path.join(
        os.path.dirname(__file__), "..", expert_answers_dir, expert_answers_file
    )
    expert = Human(
        use_loaded_answers=load_answers, append=append, answers_save_path=filename
    )
    if load_answers:
        expert.load_answers(filename)

    fit_scrdr, _ = get_fit_scrdr(cases, targets, draw_tree=False)

    grdr = GeneralRDR()
    grdr.add_rdr(fit_scrdr)

    n = 20
    true_targets = [get_habitat(x, t) for x, t in zip(cases[:n], targets[:n])]
    if no_targets:
        all_targets = [{"habitats": None} for i in range(n)]
    else:
        all_targets = true_targets
    case_queries = [
        CaseQuery(
            case,
            name,
            (Species,) if name == "species" else (Habitat,),
            True if name == "species" else False,
            _target=target,
            case_factory=case_factory,
            case_factory_idx=i,
            scenario=scenario,
        )
        for i, (case, targets) in enumerate(zip(cases[:n], all_targets))
        for name, target in targets.items()
    ]
    grdr.fit(
        case_queries,
        expert=expert,
        animate_tree=draw_tree,
        update_existing_rules=update_existing_rules,
    )
    if save_answers:
        expert.save_answers(filename)
    for case, case_targets in zip(cases[:n], true_targets):
        cat = grdr.classify(case)
        for cat_name, cat_val in cat.items():
            if cat_name in case_targets:
                assert make_set(cat_val) == make_set(case_targets[cat_name])
    return grdr, true_targets


def get_habitat(x: Case, t: Category):
    habitat = set()
    if t == Species.mammal and x["aquatic"] == 0:
        habitat = {Habitat.land}
    elif t == Species.bird:
        habitat = {Habitat.land}
        if x["airborne"] == 1:
            habitat.update({Habitat.air})
        if x["aquatic"] == 1:
            habitat.update({Habitat.water})
    elif t == Species.fish:
        habitat = {Habitat.water}
    elif t == Species.molusc:
        habitat = {Habitat.land}
        if x["aquatic"] == 1:
            habitat.update({Habitat.water})
    if len(habitat) == 0:
        return {t.__class__.__name__.lower(): t}
    else:
        return {"habitats": habitat, t.__class__.__name__.lower(): t}

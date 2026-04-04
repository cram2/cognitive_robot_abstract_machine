import os
from unittest import TestCase

import sqlalchemy.orm
from sqlalchemy import select
from sqlalchemy.orm import MappedColumn as Column
from typing_extensions import List, Sequence
import pandas as pd

from .datasets import Base, MappedAnimal, Species, get_dataset, Habitat, HabitatTable
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.experts import Human
from krrood.ripple_down_rules.rdr import SingleClassRDR, MultiClassRDR, GeneralRDR
from krrood.ripple_down_rules.utils import make_set
from .test_helpers.helpers import get_fit_scrdr


class TestAlchemyRDR(TestCase):
    session: sqlalchemy.orm.Session
    test_results_dir: str = os.path.join(os.path.dirname(__file__), "test_results")
    expert_answers_dir: str = os.path.join(
        os.path.dirname(__file__), "test_expert_answers"
    )
    cache_file: str = os.path.join(test_results_dir, "zoo_dataset.pkl")
    all_cases: Sequence[MappedAnimal]
    targets: List[Species]

    @classmethod
    def setUpClass(cls):
        # load dataset
        zoo = get_dataset(111, cls.cache_file)

        # get data and targets (as pandas dataframes)
        X = zoo["features"]
        y = zoo["targets"]
        names = zoo["ids"].values.flatten()
        X.loc[:, "name"] = names

        cls._init_session_and_insert_data(X)

        # init cases
        query = select(MappedAnimal)
        cls.all_cases = cls.session.scalars(query).all()

        # init targets
        category_names = [
            "mammal",
            "bird",
            "reptile",
            "fish",
            "amphibian",
            "insect",
            "molusc",
        ]
        category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
        cls.targets = [Species(category_id_to_name[i]) for i in y.values.flatten()]

    @classmethod
    def _init_session_and_insert_data(cls, data: pd.DataFrame):
        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = sqlalchemy.orm.Session(engine)
        session.bulk_insert_mappings(MappedAnimal, data.to_dict(orient="records"))
        session.commit()
        cls.session = session

    def test_fit_scrdr(self):
        use_loaded_answers = True
        draw_tree = False
        filename = os.path.join(self.expert_answers_dir, "scrdr_expert_answers_fit")
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        query = select(MappedAnimal)
        result = self.session.scalars(query).all()
        scrdr = SingleClassRDR()
        case_queries = [
            CaseQuery(c, "species", (Species,), True, _target=t)
            for c, t in zip(self.all_cases, self.targets)
        ]
        scrdr.fit(
            case_queries, expert=expert, animate_tree=draw_tree, session=self.session
        )

        cat = scrdr.classify(result[50])
        assert cat == self.targets[50]

    def test_fit_mcrdr_stop_only(self):
        use_loaded_answers = True
        draw_tree = False
        expert, filename = self.get_expert_and_file_name(
            use_loaded_answers, "mcrdr_expert_answers_stop_only_fit"
        )

        mcrdr = MultiClassRDR()
        case_queries = [
            CaseQuery(c, "species", (Species,), True, _target=t)
            for c, t in zip(self.all_cases, self.targets)
        ]
        mcrdr.fit(case_queries, expert=expert, animate_tree=draw_tree)

        for case_query in case_queries:
            cat = mcrdr.classify(case_query.case)
            assert make_set(cat) == make_set(case_query.target_value)

    def test_fit_grdr(self):
        use_loaded_answers = True
        save_answers = False
        draw_tree = False
        filename = os.path.join(self.expert_answers_dir, "grdr_expert_answers_fit")
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        fit_scrdr, _ = get_fit_scrdr(self.all_cases, self.targets)

        grdr = GeneralRDR()
        grdr.add_rdr(fit_scrdr)

        def get_habitat(x: MappedAnimal, t: Column) -> List[Column]:
            habitats = set()
            if t == Species.mammal and x.aquatic == 0:
                habitats = {HabitatTable(Habitat.land)}
            elif t == Species.bird:
                habitats = {HabitatTable(Habitat.land)}
                if x.airborne == 1:
                    habitats.update({HabitatTable(Habitat.air)})
                if x.aquatic == 1:
                    habitats.update({HabitatTable(Habitat.water)})
            elif t == Species.fish:
                habitats = {HabitatTable(Habitat.water)}
            elif t == Species.molusc:
                habitats = {HabitatTable(Habitat.land)}
                if x.aquatic == 1:
                    habitats.update({HabitatTable(Habitat.water)})
            if len(habitats) == 0:
                return {"species": t}
            else:
                return {"habitats": habitats, "species": t}

        n = 20
        habitat_targets = [
            get_habitat(x, t) for x, t in zip(self.all_cases[:n], self.targets[:n])
        ]
        case_queries = []
        for case, targets in zip(self.all_cases[:n], habitat_targets):
            for attr, target in targets.items():
                case_queries.append(
                    CaseQuery(
                        case,
                        attr,
                        (Species,) if attr == "species" else (HabitatTable,),
                        True if attr == "species" else False,
                        _target=target,
                    )
                )
        grdr.fit(case_queries, expert=expert, animate_tree=draw_tree)
        # for rule in grdr.start_rules:
        #     render_tree(rule, use_dot_exporter=True,
        #                 filename=self.test_results_dir + f"/grdr_{type(rule.conclusion).__name__}")

        for case, case_targets in zip(self.all_cases[:n], habitat_targets):
            cat = grdr.classify(case)
            for cat_name, cat_val in cat.items():
                assert make_set(cat_val) == make_set(case_targets[cat_name])

        if save_answers:
            cwd = os.path.dirname(__file__)
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def get_expert_and_file_name(self, use_loaded_answers: bool, filename: str):
        filename = self.expert_answers_dir + "/" + filename
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)
        return expert, filename


# tests = TestAlchemyRDR()
# tests.setUpClass()
# tests.test_fit_scrdr()
# tests.test_fit_mcrdr_stop_only()
# tests.test_fit_grdr()

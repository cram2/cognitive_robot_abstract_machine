import os
import sys
from unittest import TestCase

from typing_extensions import List, Optional

from krrood.ripple_down_rules.rules import MultiClassStopRule
from .datasets import Habitat, Species, load_zoo_cases
from .datasets import load_zoo_dataset
from krrood.ripple_down_rules.datastructures.case import Case
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.datastructures.enums import MCRDRMode
from krrood.ripple_down_rules.experts import Human
from krrood.ripple_down_rules.rdr import (
    SingleClassRDR,
    MultiClassRDR,
    GeneralRDR,
    RDRWithCodeWriter,
)
from krrood.ripple_down_rules.utils import (
    render_tree,
    make_set,
    extract_function_or_class_file,
)
from .test_helpers.helpers import get_fit_scrdr, get_fit_mcrdr, get_fit_grdr

try:
    from PyQt6.QtWidgets import QApplication
    from krrood.ripple_down_rules.user_interface.gui import RDRCaseViewer
except ImportError as e:
    RDRCaseViewer = None
    QApplication = None


class TestRDR(TestCase):
    all_cases: List[Case]
    targets: List[str]
    case_queries: List[CaseQuery]
    test_results_dir: str = os.path.join(os.path.dirname(__file__), "test_results")
    expert_answers_dir: str = os.path.join(
        os.path.dirname(__file__), "test_expert_answers"
    )
    generated_rdrs_dir: str = os.path.join(
        os.path.dirname(__file__), "test_generated_rdrs"
    )
    cache_file: str = os.path.join(test_results_dir, "zoo_dataset.pkl")
    app: Optional[QApplication] = None
    viewer: Optional[RDRCaseViewer] = None
    use_gui: bool = False

    @classmethod
    def setUpClass(cls):
        # fetch dataset
        cls.all_cases, cls.targets = load_zoo_dataset(cache_file=cls.cache_file)
        cls.case_queries = [
            CaseQuery(
                case,
                "species",
                Species,
                True,
                _target=target,
                case_factory=load_zoo_cases,
                case_factory_idx=i,
            )
            for i, (case, target) in enumerate(zip(cls.all_cases, cls.targets))
        ]
        for test_dir in [
            cls.test_results_dir,
            cls.expert_answers_dir,
            cls.generated_rdrs_dir,
        ]:
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
        if RDRCaseViewer is not None and QApplication is not None and cls.use_gui:
            cls.app = QApplication(sys.argv)
            cls.viewer = RDRCaseViewer()

    def test_classify_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = os.path.join(
            self.expert_answers_dir, "scrdr_expert_answers_classify"
        )
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        scrdr = SingleClassRDR()
        cat = scrdr.fit_case(
            self.case_queries[0], expert=expert, scenario=self.test_classify_scrdr
        )
        self.assertEqual(cat, self.targets[0])

        if save_answers:
            cwd = os.path.dirname(__file__)
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_scrdr(self):
        scrdr, _ = get_fit_scrdr(
            self.all_cases,
            self.targets,
            draw_tree=False,
            expert_answers_dir=self.expert_answers_dir,
            expert_answers_file="scrdr_expert_answers_fit",
            load_answers=True,
            save_answers=False,
            scenario=self.test_fit_scrdr,
        )
        # render_tree(scrdr.start_rule, use_dot_exporter=True,
        #             filename=self.test_results_dir + f"/scrdr")

    def test_read_scrdr_tree(self):
        original_scrdr, _ = get_fit_scrdr(self.all_cases, self.targets)
        save_dir = self.generated_rdrs_dir
        model_name = original_scrdr.save(save_dir)
        scrdr_loaded = SingleClassRDR.load(
            save_dir, original_scrdr.generated_python_file_name
        )
        model_path = os.path.join(save_dir, model_name)
        rules_root = SingleClassRDR.read_rule_tree_from_python(model_path)
        all_rules = [rules_root] + list(rules_root.descendants)
        all_og_rules = [scrdr_loaded.start_rule] + list(
            scrdr_loaded.start_rule.descendants
        )
        assert len(all_rules) == len(all_og_rules)
        for rule, og_rule in zip(all_rules, all_og_rules):
            assert (
                rule.conditions.split("conditions_")[1]
                == rule.conclusion.split("conclusion_")[1]
                == rule.uid
            )
            assert rule.uid == og_rule.uid
            if not rule.parent:
                assert not og_rule.parent
            else:
                assert rule.parent.uid == og_rule.parent.uid
            rule.name = f"{rule.conditions[:14]}\n -> {rule.conclusion[:14]} "
        render_tree(rules_root, use_dot_exporter=True, filename="scrdr_read_tree")

    def test_load_scrdr_from_python(self):
        original_scrdr, _ = get_fit_scrdr(self.all_cases, self.targets)
        save_dir = self.generated_rdrs_dir
        model_name = original_scrdr.save(save_dir)
        model_path = os.path.join(save_dir, model_name)
        scrdr_loaded = SingleClassRDR.from_python(model_path)
        for case, target in zip(self.all_cases, self.targets):
            self.assertEqual(scrdr_loaded.classify(case), target)

    def test_read_mcrdr_tree(self):
        original_mcrdr = get_fit_mcrdr(self.all_cases, self.targets)
        save_dir = self.generated_rdrs_dir
        model_name = original_mcrdr.save(save_dir, "test_read_mcrdr_tree")
        scrdr_loaded = MultiClassRDR.load(
            save_dir, model_name, package_name=f"test.test_generated_rdrs.{model_name}"
        )
        model_path = os.path.join(save_dir, model_name)
        rules_root = MultiClassRDR.read_rule_tree_from_python(model_path)
        for rule, og_rule in zip(
            [rules_root] + list(rules_root.descendants),
            [scrdr_loaded.start_rule] + list(scrdr_loaded.start_rule.descendants),
        ):
            if isinstance(rule, MultiClassStopRule):
                assert rule.conditions.split("conditions_")[1] == rule.uid
            else:
                assert (
                    rule.conditions.split("conditions_")[1]
                    == rule.conclusion.split("conclusion_")[1]
                    == rule.uid
                )
            assert rule.uid == og_rule.uid
            if not rule.parent:
                assert not og_rule.parent
            else:
                assert rule.parent.uid == og_rule.parent.uid
            if isinstance(rule, MultiClassStopRule):
                rule.name = f"{rule.conditions[:14]}\n -> {rule.conclusion} "
            else:
                rule.name = f"{rule.conditions[:14]}\n -> {rule.conclusion[:14]} "
        render_tree(rules_root, use_dot_exporter=True, filename="mcrdr_read_tree")

    def test_load_mcrdr_from_python(self):
        original_mcrdr = get_fit_mcrdr(self.all_cases, self.targets)
        save_dir = self.generated_rdrs_dir
        model_name = original_mcrdr.save(save_dir, "test_load_mcrdr_from_python")
        model_path = os.path.join(save_dir, model_name)
        mcrdr_loaded = MultiClassRDR.from_python(model_path)
        for case, target in zip(self.all_cases, self.targets):
            self.assertEqual(make_set(mcrdr_loaded.classify(case)), make_set(target))

    def test_load_grdr_from_python(self):
        original_grdr, targets = get_fit_grdr(self.all_cases, self.targets)
        save_dir = self.generated_rdrs_dir
        model_name = original_grdr.save(save_dir, "test_load_grdr_from_python")
        model_path = os.path.join(save_dir, model_name)
        grdr_loaded = GeneralRDR.from_python(model_path)
        for case, target in zip(self.all_cases[: len(targets)], targets):
            cat = grdr_loaded.classify(case)
            for cat_name, cat_val in cat.items():
                if cat_name in target:
                    self.assertEqual(make_set(cat_val), make_set(target[cat_name]))

    def test_save_load_scrdr(self):

        scrdr, _ = get_fit_scrdr(
            self.all_cases,
            self.targets,
            draw_tree=False,
            expert_answers_dir=self.expert_answers_dir,
            expert_answers_file="scrdr_expert_answers_fit",
            load_answers=True,
            scenario=self.test_save_load_scrdr,
        )
        save_dir = self.generated_rdrs_dir
        model_name = scrdr.save(save_dir)
        assert os.path.exists(save_dir)
        assert os.path.exists(os.path.join(save_dir, model_name, scrdr.metadata_folder))
        assert os.path.exists(
            os.path.join(
                save_dir, model_name, scrdr.metadata_folder, f"{model_name}.json"
            )
        )
        assert os.path.exists(os.path.join(save_dir, "__init__.py"))
        assert os.path.exists(os.path.join(save_dir, model_name, "__init__.py"))
        assert os.path.exists(
            os.path.join(save_dir, model_name, scrdr.generated_python_file_name + ".py")
        )
        assert os.path.exists(
            os.path.join(
                save_dir, model_name, scrdr.generated_python_defs_file_name + ".py"
            )
        )

        scrdr_loaded = SingleClassRDR.load(save_dir, scrdr.generated_python_file_name)
        self.assertEqual(scrdr.start_rule.uid, scrdr_loaded.start_rule.uid)
        for case, target in zip(self.all_cases, self.targets):
            cat = scrdr_loaded.classify(case)
            self.assertEqual(cat, target)

    def test_expert_incremental_save(self):
        if not os.path.exists(
            os.path.join(
                self.test_results_dir, "expert_incremental_save", "expert_answers.py"
            )
        ):
            return
            # os.remove(os.path.join(self.test_results_dir, "expert_incremental_save/expert_answers.py"))
        expert = Human(
            answers_save_path=os.path.join(
                self.test_results_dir, "expert_incremental_save", "expert_answers"
            )
        )
        cq = CaseQuery(self.all_cases[0], "species", Species, True)
        conclusion = expert.ask_for_conclusion(cq)
        assert os.path.exists(
            os.path.join(
                self.test_results_dir, "expert_incremental_save", "expert_answers.py"
            )
        )

    def test_scrdr_incremental_save_and_load(self):
        if os.path.exists(
            os.path.join(
                self.test_results_dir, "scrdr_incremental_save", "animal_species_scrdr"
            )
        ):
            scrdr = SingleClassRDR.load(
                os.path.join(self.test_results_dir, "scrdr_incremental_save"),
                "animal_species_scrdr",
            )
            scrdr.ask_always = False
        else:
            scrdr = SingleClassRDR(
                save_dir=os.path.join(self.test_results_dir, "scrdr_incremental_save")
            )
            return
        expert = Human(
            answers_save_path=os.path.join(
                self.test_results_dir, "scrdr_incremental_save", "expert_answers"
            )
        )
        cq = CaseQuery(
            self.all_cases[0],
            "species",
            Species,
            True,
            case_factory=load_zoo_cases,
            case_factory_idx=0,
            scenario=self.test_scrdr_incremental_save_and_load,
        )
        scrdr.fit_case(cq, expert=expert)
        assert not os.path.exists(
            os.path.join(
                self.test_results_dir, "scrdr_incremental_save", "expert_answers.py"
            )
        )
        assert scrdr.classify(self.all_cases[0]) == self.targets[0]

    def test_fit_scrdr_with_no_targets(self):
        # Test with no targets
        draw = False
        scrdr, case_queries = get_fit_scrdr(
            self.all_cases[:20],
            [],
            draw_tree=draw,
            expert_answers_dir=self.expert_answers_dir,
            expert_answers_file="scrdr_expert_answers_fit_no_targets",
            load_answers=True,
            save_answers=False,
            update_existing_rules=True,
            scenario=self.test_fit_scrdr_with_no_targets,
        )
        if draw:
            render_tree(
                scrdr.start_rule,
                use_dot_exporter=True,
                filename=os.path.join(self.test_results_dir, "scrdr_no_targets"),
            )

    def test_write_scrdr_no_targets_to_python_file(self):
        # Test with no targets
        scrdr, case_queries = get_fit_scrdr(
            self.all_cases[:20],
            [],
            draw_tree=False,
            expert_answers_dir=self.expert_answers_dir,
            expert_answers_file="scrdr_expert_answers_fit_no_targets",
            load_answers=True,
            save_answers=False,
            update_existing_rules=True,
            scenario=self.test_write_scrdr_no_targets_to_python_file,
        )
        model_dir = os.path.join(self.generated_rdrs_dir, "scrdr_no_targets")
        os.makedirs(model_dir, exist_ok=True)
        scrdr._write_to_python(model_dir)
        classify_species_scrdr = scrdr.get_rdr_classifier_from_python_file(model_dir)
        for case_query, target in zip(case_queries, self.targets):
            cat = classify_species_scrdr(case_query.case)
            self.assertEqual(cat, target)

    def test_fit_mcrdr_with_no_targets(self):
        # Test with no targets
        mcrdr = get_fit_mcrdr(
            self.all_cases[:20],
            [],
            draw_tree=False,
            expert_answers_dir=self.expert_answers_dir,
            expert_answers_file="mcrdr_expert_answers_fit_no_targets",
            load_answers=True,
            save_answers=False,
            update_existing_rules=True,
            scenario=self.test_fit_mcrdr_with_no_targets,
        )
        # render_tree(mcrdr.start_rule, use_dot_exporter=True,
        #             filename=self.test_results_dir + f"/mcrdr_no_targets")
        for case, target in zip(self.all_cases[:20], self.targets[:20]):
            cat = mcrdr.classify(case)
            self.assertEqual(make_set(cat), make_set(target))

    def test_write_mcrdr_no_targets_to_python_file(self):
        # Test with no targets
        mcrdr = get_fit_mcrdr(
            self.all_cases[:20],
            [],
            draw_tree=False,
            expert_answers_dir=self.expert_answers_dir,
            expert_answers_file="mcrdr_expert_answers_fit_no_targets",
            load_answers=True,
            save_answers=False,
            update_existing_rules=True,
            scenario=self.test_write_mcrdr_no_targets_to_python_file,
        )
        model_dir = os.path.join(self.generated_rdrs_dir, "mcrdr_no_targets")
        os.makedirs(model_dir, exist_ok=True)
        mcrdr._write_to_python(model_dir)
        classify_species_mcrdr = mcrdr.get_rdr_classifier_from_python_file(model_dir)
        for case, target in zip(self.all_cases[:20], self.targets[:20]):
            cat = classify_species_mcrdr(case)
            self.assertEqual(make_set(cat), make_set(target))

    # @skip("Test is not implemented yet")
    def test_fit_grdr_with_no_targets(self):
        draw_tree = False
        # Test with no targets
        grdr, all_targets = get_fit_grdr(
            self.all_cases,
            self.targets,
            draw_tree=draw_tree,
            expert_answers_dir=self.expert_answers_dir,
            expert_answers_file="grdr_expert_answers_fit_no_targets",
            load_answers=True,
            save_answers=False,
            append=False,
            no_targets=True,
            update_existing_rules=True,
            scenario=self.test_fit_grdr_with_no_targets,
        )
        if draw_tree:
            for conclusion_name, rdr in grdr.start_rules_dict.items():
                render_tree(
                    rdr.start_rule,
                    use_dot_exporter=True,
                    filename=os.path.join(
                        self.test_results_dir,
                        f"grdr_no_targets_{conclusion_name}",
                    ),
                )
        for case, case_targets in zip(self.all_cases[:20], all_targets):
            cat = grdr.classify(case)
            for cat_name, cat_val in cat.items():
                if cat_name in case_targets:
                    self.assertEqual(
                        make_set(cat_val), make_set(case_targets[cat_name])
                    )

    def test_write_grdr_no_targets_to_python_file(self):
        # Test with no targets
        grdr, all_targets = get_fit_grdr(
            self.all_cases,
            self.targets,
            draw_tree=False,
            expert_answers_dir=self.expert_answers_dir,
            expert_answers_file="grdr_expert_answers_fit_no_targets",
            load_answers=True,
            save_answers=False,
            append=False,
            no_targets=True,
            update_existing_rules=True,
            scenario=self.test_write_grdr_no_targets_to_python_file,
        )
        model_dir = os.path.join(self.generated_rdrs_dir, "grdr_no_targets")
        grdr._write_to_python(model_dir)
        classify_species_grdr = grdr.get_rdr_classifier_from_python_file(model_dir)
        for case, case_targets in zip(self.all_cases[:20], all_targets):
            cat = classify_species_grdr(case)
            for cat_name, cat_val in cat.items():
                if cat_name in case_targets:
                    self.assertEqual(
                        make_set(cat_val), make_set(case_targets[cat_name])
                    )

    def test_fit_multi_line_scrdr(self):
        n = 20
        scrdr, _ = get_fit_scrdr(
            self.all_cases[:n],
            self.targets[:n],
            draw_tree=False,
            expert_answers_dir=self.expert_answers_dir,
            expert_answers_file="scrdr_multi_line_expert_answers_fit",
            load_answers=True,
            save_answers=False,
            scenario=self.test_fit_multi_line_scrdr,
        )
        # render_tree(scrdr.start_rule, use_dot_exporter=True,
        #             filename=self.test_results_dir + f"/scrdr_multi_line")

    def test_write_multi_line_scrdr_to_python_file(self):
        n = 20
        scrdr, _ = get_fit_scrdr(
            self.all_cases[:n],
            self.targets[:n],
            draw_tree=False,
            expert_answers_dir=self.expert_answers_dir,
            expert_answers_file="scrdr_multi_line_expert_answers_fit",
            load_answers=True,
            scenario=self.test_write_multi_line_scrdr_to_python_file,
        )
        model_dir = os.path.join(self.generated_rdrs_dir, "scrdr_multi_line")
        scrdr._write_to_python(model_dir)
        classify_species_scrdr = scrdr.get_rdr_classifier_from_python_file(model_dir)
        for case, target in zip(self.all_cases[:n], self.targets[:n]):
            cat = classify_species_scrdr(case)
            self.assertEqual(cat, target)

    def test_write_scrdr_to_python_file(self):
        scrdr, _ = get_fit_scrdr(
            self.all_cases, self.targets, scenario=self.test_write_scrdr_to_python_file
        )
        model_dir = os.path.join(self.generated_rdrs_dir, "scrdr")
        scrdr._write_to_python(model_dir)
        classify_species_scrdr = scrdr.get_rdr_classifier_from_python_file(model_dir)
        for case, target in zip(self.all_cases, self.targets):
            cat = classify_species_scrdr(case)
            self.assertEqual(cat, target)

    def test_update_rdr_from_python_file(self):
        scrdr, _ = get_fit_scrdr(
            self.all_cases, self.targets, scenario=self.test_update_rdr_from_python_file
        )
        modified_model_dir = os.path.join(self.generated_rdrs_dir, "scrdr_modified")
        scrdr._write_to_python(modified_model_dir)
        main_file_path = os.path.join(
            modified_model_dir, f"{scrdr.generated_python_file_name}.py"
        )
        filepath = main_file_path.replace(".py", "_defs.py")
        func_name = f"conditions_{scrdr.start_rule.uid}"
        first_rule_conditions, line_numbers = extract_function_or_class_file(
            filepath, func_name, join_lines=False, return_line_numbers=True
        )
        self.assertEqual(
            first_rule_conditions[func_name][-1], "    return case.milk == 1"
        )
        # modify the condition to be case.milk==0
        with open(filepath, "r") as f:
            lines = f.readlines()
        lines[line_numbers[func_name][-1] - 1] = "    return case.milk == 0\n"
        with open(filepath, "w") as f:
            f.writelines(lines)
        first_rule_conditions, line_numbers = extract_function_or_class_file(
            filepath, func_name, join_lines=False, return_line_numbers=True
        )
        self.assertEqual(
            first_rule_conditions[func_name][-1], "    return case.milk == 0"
        )
        scrdr: RDRWithCodeWriter
        scrdr.update_from_python(modified_model_dir, python_file_path=main_file_path)
        self.assertEqual(
            scrdr.start_rule.conditions.user_input.strip().split("\n")[-1].strip(),
            "return case.milk == 0",
        )
        classify_species_scrdr = scrdr.get_rdr_classifier_from_python_file(
            modified_model_dir
        )
        for case, target in zip(self.all_cases, self.targets):
            if case.milk == 0:
                cat = classify_species_scrdr(case)
                self.assertEqual(cat, Species.mammal)

    def test_write_mcrdr_to_python_file(self):
        mcrdr = get_fit_mcrdr(
            self.all_cases, self.targets, scenario=self.test_write_mcrdr_to_python_file
        )
        model_dir = os.path.join(self.generated_rdrs_dir, "mcrdr")
        mcrdr._write_to_python(model_dir)
        classify_species_mcrdr = mcrdr.get_rdr_classifier_from_python_file(model_dir)
        for case, target in zip(self.all_cases, self.targets):
            cat = classify_species_mcrdr(case)
            self.assertEqual(make_set(cat), make_set(target))

    def test_write_mcrdr_multi_line_to_python_file(self):
        n = 20
        mcrdr = get_fit_mcrdr(
            self.all_cases[:n],
            self.targets[:n],
            draw_tree=False,
            expert_answers_dir=self.expert_answers_dir,
            expert_answers_file="mcrdr_multi_line_expert_answers_fit",
            load_answers=True,
            save_answers=False,
            scenario=self.test_write_mcrdr_multi_line_to_python_file,
        )
        model_dir = os.path.join(self.generated_rdrs_dir, "mcrdr_multi_line")
        mcrdr._write_to_python(model_dir)
        classify_species_mcrdr = mcrdr.get_rdr_classifier_from_python_file(model_dir)
        for case, target in zip(self.all_cases[:n], self.targets[:n]):
            cat = classify_species_mcrdr(case)
            self.assertEqual(make_set(cat), make_set(target))

    def test_write_grdr_to_python_file(self):
        grdr, all_targets = get_fit_grdr(
            self.all_cases, self.targets, scenario=self.test_write_grdr_to_python_file
        )
        model_dir = os.path.join(self.generated_rdrs_dir, "grdr")
        grdr._write_to_python(model_dir)
        classify_animal_grdr = grdr.get_rdr_classifier_from_python_file(model_dir)
        for case, case_targets in zip(self.all_cases[: len(all_targets)], all_targets):
            cat = classify_animal_grdr(case)
            for cat_name, cat_val in cat.items():
                self.assertEqual(make_set(cat_val), make_set(case_targets[cat_name]))

    def test_classify_mcrdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = os.path.join(
            self.expert_answers_dir, "mcrdr_expert_answers_classify"
        )
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        mcrdr = MultiClassRDR()
        cats = mcrdr.fit_case(
            self.case_queries[0], expert=expert, scenario=self.test_classify_mcrdr
        )

        self.assertEqual(cats[0], self.targets[0])

        if save_answers:
            cwd = os.path.dirname(__file__)
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_mcrdr_stop_only(self):
        use_loaded_answers = True
        draw_tree = False
        save_answers = False
        filename = os.path.join(
            self.expert_answers_dir, "mcrdr_expert_answers_stop_only_fit"
        )
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)
        mcrdr = MultiClassRDR()
        case_queries = self.case_queries
        mcrdr.fit(
            case_queries,
            expert=expert,
            animate_tree=draw_tree,
            scenario=self.test_fit_mcrdr_stop_only,
        )
        # render_tree(mcrdr.start_rule, use_dot_exporter=True,
        #             filename=self.test_results_dir + f"/mcrdr_stop_only")
        for case_query in case_queries:
            cat = mcrdr.classify(case_query.case)
            self.assertEqual(make_set(cat), make_set(case_query.target_value))
        if save_answers:
            cwd = os.path.dirname(__file__)
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_mcrdr_stop_plus_rule(self):
        use_loaded_answers = True
        draw_tree = False
        save_answers = False
        append = False
        filename = os.path.join(
            self.expert_answers_dir, "mcrdr_stop_plus_rule_expert_answers_fit"
        )
        expert = Human(use_loaded_answers=use_loaded_answers, append=append)
        if use_loaded_answers:
            expert.load_answers(filename)
        mcrdr = MultiClassRDR(mode=MCRDRMode.StopPlusRule)
        case_queries = self.case_queries
        mcrdr.fit(
            case_queries,
            expert=expert,
            animate_tree=draw_tree,
            scenario=self.test_fit_mcrdr_stop_plus_rule,
        )
        # render_tree(mcrdr.start_rule, use_dot_exporter=True,
        #             filename=self.test_results_dir + f"/mcrdr_stop_plus_rule")
        for case_query in case_queries:
            cat = mcrdr.classify(case_query.case)
            self.assertEqual(make_set(cat), make_set(case_query.target_value))
        if save_answers:
            cwd = os.path.dirname(__file__)
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_mcrdr_stop_plus_rule_combined(self):
        use_loaded_answers = True
        save_answers = False
        draw_tree = False
        append = False
        filename = os.path.join(
            self.expert_answers_dir, "mcrdr_stop_plus_rule_combined_expert_answers_fit"
        )
        expert = Human(use_loaded_answers=use_loaded_answers, append=append)
        if use_loaded_answers:
            expert.load_answers(filename)
        mcrdr = MultiClassRDR(mode=MCRDRMode.StopPlusRuleCombined)
        case_queries = self.case_queries
        mcrdr.fit(
            case_queries,
            expert=expert,
            animate_tree=draw_tree,
            scenario=self.test_fit_mcrdr_stop_plus_rule_combined,
        )
        # render_tree(mcrdr.start_rule, use_dot_exporter=True,
        #             filename=self.test_results_dir + f"/mcrdr_stop_plus_rule_combined")
        for case_query in case_queries:
            cat = mcrdr.classify(case_query.case)
            self.assertEqual(make_set(cat), make_set(case_query.target_value))
        if save_answers:
            cwd = os.path.dirname(__file__)
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_classify_grdr(self):
        use_loaded_answers = True
        save_answers = False
        filename = os.path.join(self.expert_answers_dir, "grdr_expert_answers_classify")
        expert = Human(use_loaded_answers=use_loaded_answers)
        if use_loaded_answers:
            expert.load_answers(filename)

        grdr = GeneralRDR()

        targets = [self.targets[0], Habitat.land]
        attribute_names = [t.__class__.__name__.lower() for t in targets]
        targets = dict(zip(attribute_names, targets))
        case_queries = [
            CaseQuery(
                self.all_cases[0],
                a,
                (type(t),),
                True if a == "species" else False,
                _target=t,
                case_factory=load_zoo_cases,
                case_factory_idx=0,
                scenario=self.test_classify_scrdr,
            )
            for a, t in targets.items()
        ]
        grdr.fit(case_queries, expert=expert)
        cats = grdr.classify(self.all_cases[0])
        for cat_name, value in cats.items():
            self.assertEqual(make_set(value), make_set(targets[cat_name]))

        if save_answers:
            cwd = os.path.dirname(__file__)
            file = os.path.join(cwd, filename)
            expert.save_answers(file)

    def test_fit_grdr(self):
        grdr, all_targets = get_fit_grdr(
            self.all_cases,
            self.targets,
            draw_tree=False,
            load_answers=True,
            save_answers=False,
            scenario=self.test_fit_grdr,
        )
        # for conclusion_name, rdr in grdr.start_rules_dict.items():
        #     render_tree(rdr.start_rule, use_dot_exporter=True,
        #                 filename=self.test_results_dir + f"/grdr_{conclusion_name}")

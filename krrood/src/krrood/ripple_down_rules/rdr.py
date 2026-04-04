from __future__ import annotations

import ast
import importlib
import os
import sys
from abc import ABC, abstractmethod
from copy import copy
from io import TextIOWrapper
from pathlib import Path
from types import NoneType, ModuleType

from krrood.ripple_down_rules.datastructures.dataclasses import CaseFactoryMetaData

from krrood.ripple_down_rules import logger
from krrood.ripple_down_rules.failures import RDRLoadError

try:
    from matplotlib import pyplot as plt

    Figure = plt.Figure
except ImportError as e:
    logger.debug(f"{e}: matplotlib is not installed")
    matplotlib = None
    Figure = None
    plt = None

from sqlalchemy.orm import DeclarativeBase as SQLTable
from typing_extensions import (
    List,
    Optional,
    Dict,
    Type,
    Union,
    Any,
    Self,
    Tuple,
    Callable,
    Set,
)

from krrood.ripple_down_rules.datastructures.callable_expression import (
    CallableExpression,
)
from krrood.ripple_down_rules.datastructures.case import (
    Case,
    CaseAttribute,
    create_case,
)
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.datastructures.enums import MCRDRMode, RDREdge
from krrood.ripple_down_rules.experts import Expert, Human
from krrood.ripple_down_rules.helpers import (
    is_matching,
    general_rdr_classify,
    get_an_updated_case_copy,
    update_case_and_conclusions_with_rule_output,
    update_case_with_conclusion_output,
)
from krrood.ripple_down_rules.rules import (
    Rule,
    SingleClassRule,
    MultiClassTopRule,
    MultiClassStopRule,
    MultiClassRefinementRule,
    MultiClassFilterRule,
)

try:
    from .user_interface.gui import RDRCaseViewer
except ImportError as e:
    RDRCaseViewer = None
from krrood.ripple_down_rules.utils import (
    draw_tree,
    make_set,
    SubclassJSONSerializer,
    make_list,
    get_type_from_string,
    is_value_conflicting,
    extract_function_or_class_file,
    extract_imports,
    get_full_class_name,
    is_iterable,
    str_to_snake_case,
    get_import_path_from_path,
    get_imports_from_types,
    render_tree,
    get_function_return_type,
    get_file_that_ends_with,
    get_and_import_python_module,
    get_and_import_python_modules_in_a_package,
    get_type_from_type_hint,
    are_results_subclass_of_types,
    update_case_in_case_query,
    copy_case,
)


class RippleDownRules(SubclassJSONSerializer, ABC):
    """
    The abstract base class for the ripple down rules classifiers.
    """

    fig: Optional[Figure] = None
    """
    The figure to draw the tree on.
    """
    expert_accepted_conclusions: Optional[List[CaseAttribute]] = None
    """
    The conclusions that the expert has accepted, such that they are not asked again.
    """
    _generated_python_file_name: Optional[str] = None
    """
    The name of the generated python file.
    """
    name: Optional[str] = None
    """
    The name of the classifier.
    """
    case_type: Optional[Type] = None
    """
    The type of the case (input) to the RDR classifier.
    """
    case_name: Optional[str] = None
    """
    The name of the case type.
    """
    metadata_folder: str = "rdr_metadata"
    """
    The folder to save the metadata of the RDR classifier.
    """
    model_name: Optional[str] = None
    """
    The name of the model. If None, the model name will be the generated python file name.
    """
    mutually_exclusive: Optional[bool] = None
    """
    Whether the output of the classification of this rdr allows only one possible conclusion or not.
    """

    def __init__(
        self,
        start_rule: Optional[Rule] = None,
        save_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        :param start_rule: The starting rule for the classifier.
        :param save_dir: The directory to save the classifier to.
        """
        self.model_name: Optional[str] = model_name
        self.save_dir: Optional[str] = save_dir
        self.start_rule: Optional[Rule] = start_rule
        self.fig: Optional[Figure] = None
        self.viewer: Optional[RDRCaseViewer] = (
            RDRCaseViewer.instances[0]
            if RDRCaseViewer and any(RDRCaseViewer.instances)
            else None
        )
        self.input_node: Optional[Rule] = None
        self.update_model()

    def update_model(self):
        """
        Update the model from the model directory if it exists.
        """
        if self.save_dir is not None and self.model_name is not None:
            model_path = os.path.join(self.save_dir, self.model_name)
            if os.path.exists(model_path):
                try:
                    self.update_from_python(model_path, update_rule_tree=True)
                except (FileNotFoundError, ModuleNotFoundError) as e:
                    pass

    def write_rdr_metadata_to_pyton_file(self, file: TextIOWrapper):
        """
        Write the metadata of the RDR classifier to a python file.

        :param file: The file to write the metadata to.
        """
        file.write(f"name = '{self.name}'\n")
        file.write(
            f"case_type = {self.case_type.__name__ if self.case_type is not None else None}\n"
        )
        file.write(f"case_name = '{self.case_name}'\n")

    def update_rdr_metadata_from_python(self, module: ModuleType):
        """
        Update the RDR metadata from the module that contains the RDR classifier function.

        :param module: The module that contains the RDR classifier function.
        """
        try:
            self.name = (
                module.name
                if hasattr(module, "name")
                else self.start_rule.conclusion_name
            )
            self.case_type = module.case_type
            self.case_name = (
                module.case_name
                if hasattr(module, "case_name")
                else f"{self.case_type.__name__}.{self.name}"
            )
        except AttributeError as e:
            logger.warning(
                f"Could not update the RDR metadata from the module {module.__name__}. "
                f"Make sure the module has the required attributes: {e}"
            )

    def render_evaluated_rule_tree(
        self, filename: str, show_full_tree: bool = False
    ) -> None:
        if show_full_tree:
            start_rule = self.start_rule if self.input_node is None else self.input_node
            render_tree(start_rule, use_dot_exporter=True, filename=filename)
        else:
            evaluated_rules = self.get_evaluated_rule_tree()
            if evaluated_rules is not None and len(evaluated_rules) > 0:
                render_tree(
                    evaluated_rules[0],
                    use_dot_exporter=True,
                    filename=filename,
                    only_nodes=evaluated_rules,
                )

    def get_contributing_rules(self) -> Optional[List[Rule]]:
        """
        Get the contributing rules of the classifier.

        :return: The contributing rules.
        """
        if self.start_rule is None:
            return None
        return [r for r in self.get_fired_rule_tree() if r.contributed]

    def get_fired_rule_tree(self) -> Optional[List[Rule]]:
        """
        Get the fired rule tree of the classifier.

        :return: The fired rule tree.
        """
        if self.start_rule is None:
            return None
        return [r for r in self.get_evaluated_rule_tree() if r.fired]

    def get_evaluated_rule_tree(self) -> Optional[List[Rule]]:
        """
        Get the evaluated rule tree of the classifier.

        :return: The evaluated rule tree.
        """
        if self.start_rule is None:
            return None
        start_rule = self.start_rule
        evaluated_rule_tree = [
            r for r in [start_rule] + list(start_rule.descendants) if r.evaluated
        ]
        return evaluated_rule_tree

    def save(
        self,
        save_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        package_name: Optional[str] = None,
    ) -> str:
        """
        Save the classifier to a file.

        :param save_dir: The directory to save the classifier to.
        :param model_name: The name of the model to save. If None, a default name is generated.
        :param package_name: The name of the package that contains the RDR classifier function, this
        is required in case of relative imports in the generated python file.
        :return: The name of the saved model.
        """
        save_dir = save_dir or self.save_dir
        if save_dir is None:
            raise ValueError(
                "The save directory cannot be None. Please provide a valid directory to save"
                " the classifier."
            )
        if not os.path.exists(save_dir + "/__init__.py"):
            os.makedirs(save_dir, exist_ok=True)
            with open(save_dir + "/__init__.py", "w") as f:
                f.write("from . import *\n")
        if model_name is not None:
            self.model_name = model_name
        elif self.model_name is None:
            self.model_name = self.generated_python_file_name
        model_dir = os.path.join(save_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        json_dir = os.path.join(model_dir, self.metadata_folder)
        os.makedirs(json_dir, exist_ok=True)
        self.to_json_file(os.path.join(json_dir, self.model_name))
        self._write_to_python(model_dir, package_name=package_name)
        return self.model_name

    @classmethod
    def load(
        cls, load_dir: str, model_name: str, package_name: Optional[str] = None
    ) -> Self:
        """
        Load the classifier from a file.

        :param load_dir: The path to the model directory to load the classifier from.
        :param model_name: The name of the model to load.
        :param package_name: The name of the package that contains the RDR classifier function, this
        is required in case of relative imports in the generated python file.
        """
        rdr: Optional[RippleDownRules] = None
        model_dir = os.path.join(load_dir, model_name)
        json_file = os.path.join(model_dir, cls.metadata_folder, model_name)
        if os.path.exists(json_file + ".json"):
            rdr = cls.from_json_file(json_file)
        try:
            if rdr is None:
                rdr = cls.from_python(model_dir, parent_package_name=package_name)
            else:
                rdr.update_from_python(model_dir, parent_package_name=package_name)
            rdr.to_json_file(json_file)
        except (FileNotFoundError, ValueError, SyntaxError, ModuleNotFoundError) as e:
            logger.warning(
                f"Could not load the python file for the model {model_name} from {model_dir}. "
                f"Make sure the file exists and is valid."
            )
            if rdr is None:
                raise RDRLoadError(
                    f"Could not load the rdr model {model_name} from {model_dir}, error is {e}"
                )
            rdr.save(
                save_dir=load_dir, model_name=model_name, package_name=package_name
            )
        rdr.save_dir = load_dir
        rdr.model_name = model_name
        return rdr

    @classmethod
    def from_python(
        cls,
        model_path: str,
        python_file_path: Optional[str] = None,
        parent_package_name: Optional[str] = None,
    ) -> Self:
        """
        Load the RDR classifier from a generated python file.

        :param model_path: The directory where the generated python file is located.
        :param python_file_path: The path to the generated python file that contains the RDR classifier function.
        :param parent_package_name: The name of the package that contains the RDR classifier function, this
        is required in case of relative imports in the generated python file.
        :return: An instance of the RDR classifier.
        """
        rdr = cls()
        rdr.update_from_python(
            model_path,
            parent_package_name=parent_package_name,
            python_file_path=python_file_path,
            update_rule_tree=True,
        )
        return rdr

    @abstractmethod
    def _write_to_python(self, model_dir: str, package_name: Optional[str] = None):
        """
        Write the tree of rules as source code to a file.

        :param model_dir: The path to the directory to write the source code to.
        :param package_name: The name of the package that contains the RDR classifier function, this
        is required in case of relative imports in the generated python file.
        """
        pass

    def fit(
        self,
        case_queries: List[CaseQuery],
        expert: Optional[Expert] = None,
        n_iter: int = None,
        animate_tree: bool = False,
        **kwargs_for_fit_case,
    ):
        """
        Fit the classifier to a batch of cases and categories.

        :param case_queries: The cases and categories to fit the classifier to.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param n_iter: The number of iterations to fit the classifier for.
        :param animate_tree: Whether to draw the tree while fitting the classifier.
        :param kwargs_for_fit_case: The keyword arguments to pass to the fit_case method.
        """
        targets = []
        if animate_tree:
            if plt is None:
                raise ImportError(
                    "matplotlib is not installed, cannot animate the tree."
                )
            plt.ion()
        i = 0
        stop_iterating = False
        num_rules: int = 0
        while not stop_iterating:
            for case_query in case_queries:
                pred_cat = self.fit_case(
                    case_query,
                    expert=expert,
                    clear_expert_answers=False,
                    **kwargs_for_fit_case,
                )
                if case_query.target is None:
                    continue
                target = {case_query.attribute_name: case_query.target(case_query.case)}
                if len(targets) < len(case_queries):
                    targets.append(target)
                match = is_matching(self.classify, case_query, pred_cat)
                if not match:
                    print(f"Predicted: {pred_cat} but expected: {target}")
                if animate_tree and len(self.start_rule.descendants) > num_rules:
                    num_rules = len(self.start_rule.descendants)
                    self.update_figures()
            i += 1
            all_predictions = [
                1 if is_matching(self.classify, case_query) else 0
                for case_query in case_queries
                if case_query.target is not None
            ]
            all_pred = sum(all_predictions)
            logger.info(f"Accuracy: {all_pred}/{len(targets)}")
            all_predicted = targets and all_pred == len(targets)
            num_iter_reached = n_iter and i >= n_iter
            stop_iterating = all_predicted or num_iter_reached
            if stop_iterating:
                break
        logger.info(f"Finished training in {i} iterations")
        if animate_tree:
            plt.ioff()
            plt.show()

    def __call__(
        self, case: Union[Case, SQLTable]
    ) -> Union[CallableExpression, Dict[str, CallableExpression]]:
        return self.classify(case)

    def classify(
        self,
        case: Union[Case, SQLTable],
        modify_case: bool = False,
        case_query: Optional[CaseQuery] = None,
    ) -> Optional[Union[CallableExpression, Dict[str, CallableExpression]]]:
        """
        Classify a case using the RDR classifier.

        :param case: The case to classify.
        :param modify_case: Whether to modify the original case attributes with the conclusion or not.
        :param case_query: The case query containing the case to classify and the target category to compare the case with.
        :return: The category that the case belongs to.
        """
        if self.start_rule is not None:
            for rule in [self.start_rule] + list(self.start_rule.descendants):
                rule.reset()
        if self.start_rule is not None and self.start_rule.parent is None:
            if self.input_node is None:
                self.input_node = type(self.start_rule)(_parent=None, uid="0")
                self.input_node.evaluated = False
                self.input_node.fired = False
            self.start_rule.parent = self.input_node
            self.start_rule.weight = RDREdge.Empty
        if self.input_node is not None:
            self.input_node.name = type(case).__name__
        return self._classify(case, modify_case=modify_case, case_query=case_query)

    @abstractmethod
    def _classify(
        self,
        case: Union[Case, SQLTable],
        modify_case: bool = False,
        case_query: Optional[CaseQuery] = None,
    ) -> Optional[Union[CallableExpression, Dict[str, CallableExpression]]]:
        """
        Classify a case.

        :param case: The case to classify.
        :param modify_case: Whether to modify the original case attributes with the conclusion or not.
        :param case_query: The case query containing the case to classify and the target category to compare the case with.
        :return: The category that the case belongs to.
        """
        pass

    def fit_case(
        self,
        case_query: CaseQuery,
        expert: Optional[Expert] = None,
        update_existing_rules: bool = True,
        scenario: Optional[Callable] = None,
        ask_now: Callable = lambda _: False,
        clear_expert_answers: bool = True,
        ask_now_target: Optional[Any] = None,
        **kwargs,
    ) -> Union[CallableExpression, Dict[str, CallableExpression]]:
        """
        Fit the classifier to a case and ask the expert for refinements or alternatives if the classification is
        incorrect by comparing the case with the target category.

        :param case_query: The query containing the case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param update_existing_rules: Whether to update the existing same conclusion type rules that already gave
        some conclusions with the type required by the case query.
        :param scenario: The scenario at which the case was created, this is used to recreate the case if needed.
        :param ask_now: Whether to ask the expert for refinements or alternatives.
        :param clear_expert_answers: Whether to clear expert answers after saving the new rule.
        :param ask_now_target: The target output for the case that is triggers the ask_now function to return True.
        :return: The category that the case belongs to.
        """
        if case_query is None:
            raise ValueError("The case query cannot be None.")

        self.name = case_query.attribute_name if self.name is None else self.name
        self.case_type = (
            case_query.case_type if self.case_type is None else self.case_type
        )
        self.case_name = (
            case_query.case_name if self.case_name is None else self.case_name
        )
        case_query.scenario = (
            scenario if case_query.scenario is None else case_query.scenario
        )
        case_query.rdr = self

        expert = expert or Human(
            answers_save_path=(
                self.save_dir + "/expert_answers" if self.save_dir else None
            )
        )
        if case_query.target is None:
            case_query_cp = copy(case_query)
            conclusions = self.classify(
                case_query_cp.case, modify_case=True, case_query=case_query_cp
            )
            should_ask = self.should_i_ask_the_expert_for_a_target(
                conclusions, case_query_cp, update_existing_rules
            )
            ask_now_value = ask_now(case_query_cp.case)
            if ask_now_value and ask_now_target:
                case_query.target = CallableExpression(
                    conclusion=ask_now_target,
                    conclusion_type=case_query.attribute_type,
                    mutually_exclusive=self.mutually_exclusive,
                )
            elif should_ask or ask_now_value:
                expert.ask_for_conclusion(case_query_cp)
                case_query.target = case_query_cp.target
            if case_query.target is None:
                return self.classify(case_query.case)

        self.update_start_rule(case_query, expert)

        fit_case_result = self._fit_case(case_query, expert=expert, **kwargs)

        if self.save_dir is not None:
            self.save()
            if clear_expert_answers:
                expert.clear_answers()

        return fit_case_result

    @staticmethod
    def should_i_ask_the_expert_for_a_target(
        conclusions: Union[Any, Dict[str, Any]],
        case_query: CaseQuery,
        update_existing: bool,
    ) -> bool:
        """
        Determine if the rdr should ask the expert for the target of a given case query.

        :param conclusions: The conclusions of the case.
        :param case_query: The query containing the case to classify.
        :param update_existing: Whether to update rules that gave the required type of conclusions.
        :return: True if the rdr should ask the expert, False otherwise.
        """
        if conclusions is None and type(None) not in case_query.core_attribute_type:
            return True
        elif is_iterable(conclusions) and len(conclusions) == 0:
            return True
        elif isinstance(conclusions, dict):
            if case_query.attribute_name not in conclusions:
                return True
            conclusions = conclusions[case_query.attribute_name]
        conclusion_values = make_list(conclusions)
        if all(
            not isinstance(cv, case_query.core_attribute_type)
            for cv in conclusion_values
        ):
            return True
        elif update_existing:
            return True
        else:
            return False

    @abstractmethod
    def _fit_case(
        self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs
    ) -> Union[CallableExpression, Dict[str, CallableExpression]]:
        """
        Fit the RDR on a case, and ask the expert for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category.

        :param case_query: The query containing the case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        pass

    @abstractmethod
    def update_start_rule(self, case_query: CaseQuery, expert: Expert):
        """
        Update the starting rule of the classifier.

        :param case_query: The case query to update the starting rule with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        pass

    def update_figures(self):
        """
        Update the figures of the classifier.
        """
        if isinstance(self, GeneralRDR):
            for i, (rdr_name, rdr) in enumerate(self.start_rules_dict.items()):
                if not rdr.fig:
                    rdr.fig = plt.figure(f"Rule {i}: {rdr_name}")
                draw_tree(rdr.start_rule, rdr.fig)
        else:
            if not self.fig:
                self.fig = plt.figure(0)
            draw_tree(self.start_rule, self.fig)

    @property
    def type_(self):
        return self.__class__

    @classmethod
    def get_json_file_path(cls, model_path: str) -> str:
        """
        Get the path to the saved json file.

        :param model_path : The path to the model directory.
        :return: The path to the saved model.
        """
        model_name = cls.get_model_name_from_model_path(model_path)
        return os.path.join(model_path, cls.metadata_folder, f"{model_name}.json")

    @classmethod
    def get_generated_cases_file_path(cls, model_path: str) -> str:
        """
        Get the path to the python file that contains the RDR classifier cases.

        :param model_path : The path to the model directory.
        :return: The path to the generated python file.
        """
        return cls.get_generated_python_file_path(model_path).replace(
            ".py", "_cases.py"
        )

    @classmethod
    def get_generated_defs_file_path(cls, model_path: str) -> str:
        """
        Get the path to the python file that contains the RDR classifier function definitions.

        :param model_path : The path to the model directory.
        :return: The path to the generated python file.
        """
        return cls.get_generated_python_file_path(model_path).replace(".py", "_defs.py")

    @classmethod
    def get_generated_python_file_path(cls, model_path: str) -> str:
        """
        Get the path to the python file that contains the RDR classifier function.

        :param model_path : The path to the model directory.
        :return: The path to the generated python file.
        """
        model_name = cls.get_model_name_from_model_path(model_path)
        return os.path.join(model_path, f"{model_name}.py")

    @classmethod
    def get_model_name_from_model_path(cls, model_path: str) -> str:
        """
        Get the model name from the model path.

        :param model_path: The path to the model directory.
        :return: The name of the model.
        """
        file_name = get_file_that_ends_with(
            model_path, f"_{cls.get_acronym().lower()}.py"
        )
        if file_name is None:
            raise FileNotFoundError(
                f"Could not find the python file for the model in the given path: {model_path}."
            )
        return file_name.replace(".py", "")

    @property
    def generated_python_file_name(self) -> str:
        if self._generated_python_file_name is None:
            self._generated_python_file_name = self._default_generated_python_file_name
        return self._generated_python_file_name

    @generated_python_file_name.setter
    def generated_python_file_name(self, value: str):
        """
        Set the generated python file name.
        :param value: The new value for the generated python file name.
        """
        self._generated_python_file_name = value

    @property
    @abstractmethod
    def _default_generated_python_file_name(self) -> str:
        """
        :return: The default generated python file name.
        """
        pass

    @abstractmethod
    def update_from_python(
        self,
        model_dir: str,
        parent_package_name: Optional[str] = None,
        python_file_path: Optional[str] = None,
        update_rule_tree: bool = False,
    ):
        """
        Update the rules from the generated python file, that might have been modified by the user.

        :param model_dir: The directory where the generated python file is located.
        :param parent_package_name: The name of the package that contains the RDR classifier function, this
        is required in case of relative imports in the generated python file.
        :param python_file_path: The path to the generated python file that contains the RDR classifier function.
        :param update_rule_tree: Whether to update the rule tree from the python file or not.
        """
        pass

    @classmethod
    def get_acronym(cls) -> str:
        """
        :return: The acronym of the classifier.
        """
        if cls.__name__ == "GeneralRDR":
            return "RDR"
        elif cls.__name__ == "MultiClassRDR":
            return "MCRDR"
        else:
            return "SCRDR"

    def get_rdr_classifier_from_python_file(
        self, package_name: str
    ) -> Callable[[Any], Any]:
        """
        :param package_name: The name of the package that contains the RDR classifier function.
        :return: The module that contains the rdr classifier function.
        """
        # remove from imports if exists first
        package_name_importable = get_import_path_from_path(package_name)
        generated_file_name = self.generated_python_file_name
        if generated_file_name is None:
            generated_file_name = Path(
                self.get_generated_python_file_path(package_name)
            ).name
        name = (
            f"{package_name_importable}.{generated_file_name}"
            if package_name_importable
            else generated_file_name
        )
        module = importlib.import_module(name)
        importlib.reload(module)
        return module.classify


class TreeBuilder(ast.NodeVisitor, ABC):
    """Parses an AST of nested if-elif statements and reconstructs the tree."""

    def __init__(self):
        self.root: Optional[Rule] = None
        self.current_parent: Optional[Rule] = None
        self.current_edge: Optional[RDREdge] = None
        self.default_conclusion: Optional[str] = None

    def visit_FunctionDef(self, node):
        """Finds the main function and starts parsing its body."""
        for stmt in node.body:
            self.visit(stmt)

    def visit_If(self, node):
        """Handles if-elif blocks and creates nodes."""
        condition = self.get_condition_name(node.test)
        if condition is None:
            return
        rule_uid = condition.split("conditions_")[1]

        new_rule_type = self.get_new_rule_type(node)
        new_node = new_rule_type(
            conditions=condition, _parent=self.current_parent, uid=rule_uid
        )
        if self.current_parent is not None:
            self.update_current_parent(new_node)

        if self.current_parent is None and self.root is None:
            self.root = new_node

        self.current_parent = new_node

        # Parse the body of the if statement
        for stmt in node.body:
            self.current_edge = self.get_refinement_edge(node)
            self.current_parent = new_node
            self.visit(stmt)

        # Parse elif/else
        for stmt in node.orelse:
            self.current_edge = self.get_alternative_edge(node)
            self.current_parent = new_node
            if isinstance(stmt, ast.If):  # elif case
                self.visit_If(stmt)
            else:  # else case (return)
                self.process_else_statement(stmt)
        self.current_parent = new_node
        self.current_edge = None

    @abstractmethod
    def process_else_statement(self, stmt: ast.AST):
        """
        Process the else statement in the if-elif-else block.

        :param stmt: The else statement to process.
        """
        pass

    @abstractmethod
    def get_refinement_edge(self, node: ast.AST) -> RDREdge:
        """
        :param node: The current AST node to determine the edge type from.
        :return: The refinement edge type.
        """
        pass

    @abstractmethod
    def get_alternative_edge(self, node: ast.AST) -> RDREdge:
        """
        :param node: The current AST node to determine the alternative edge type from.
        :return: The alternative edge type.
        """
        pass

    @abstractmethod
    def get_new_rule_type(self, node: ast.AST) -> Type[Rule]:
        """
        Get the new rule type to create.
        :param node: The current AST node to determine the rule type from.
        :return: The new rule type.
        """
        pass

    @abstractmethod
    def update_current_parent(self, new_node: Rule):
        """
        Update the current parent rule with the new node.
        :param new_node: The new node to set as the current parent.
        """
        pass

    def visit_Return(self, node):
        """Handles return statements as leaf nodes."""
        if isinstance(node.value, ast.Call):
            return_value = node.value.func.id
        else:
            return_value = ast.literal_eval(node.value)
        if self.current_parent is None:
            self.default_conclusion = return_value
        else:
            self.current_parent.conclusion = return_value

    def get_condition_name(self, node):
        """Extracts the condition function name from an AST expression."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
        return None


class SingleClassTreeBuilder(TreeBuilder):
    """Parses an AST of generated SingleClassRDR classifier and reconstructs the rdr tree."""

    def get_new_rule_type(self, node: ast.AST) -> Type[Rule]:
        return SingleClassRule

    def get_refinement_edge(self, node: ast.AST) -> RDREdge:
        return RDREdge.Refinement

    def get_alternative_edge(self, node: ast.AST) -> RDREdge:
        return RDREdge.Alternative

    def update_current_parent(self, new_node: Rule):
        if self.current_edge == RDREdge.Alternative:
            self.current_parent.alternative = new_node
        elif self.current_edge == RDREdge.Refinement:
            self.current_parent.refinement = new_node

    def process_else_statement(self, stmt: ast.AST):
        """Handles the else statement in the if-elif-else block."""
        if isinstance(stmt, ast.Return):
            self.current_parent = None
            self.visit_Return(stmt)
        else:
            raise ValueError(f"Unexpected statement in else block: {stmt}")


class MultiClassTreeBuilder(TreeBuilder):
    """Parses an AST of generated MultiClassRDR classifier and reconstructs the rdr tree."""

    def visit_If(self, stmt: ast.If):
        super().visit_If(stmt)
        if isinstance(self.current_parent, (MultiClassTopRule, MultiClassFilterRule)):
            self.current_parent.conclusion = self.current_parent.conditions.replace(
                "conditions_", "conclusion_"
            )

    def visit_Return(self, node):
        pass

    def get_new_rule_type(self, node: ast.AST) -> Type[Rule]:
        if self.current_edge == RDREdge.Refinement:
            return MultiClassStopRule
        elif self.current_edge == RDREdge.Filter:
            return MultiClassFilterRule
        elif self.current_edge in [RDREdge.Next, None]:
            return MultiClassTopRule
        elif self.current_edge == RDREdge.Alternative:
            return self.get_refinement_rule_type(node)
        else:
            raise ValueError(f"Unknown edge type: {self.current_edge}")

    def get_alternative_edge(self, node: ast.AST) -> RDREdge:
        if isinstance(self.current_parent, MultiClassTopRule):
            return RDREdge.Next
        else:
            return RDREdge.Alternative

    def get_refinement_edge(self, node: ast.AST) -> RDREdge:
        rule_type = self.get_refinement_rule_type(node)
        return self.get_refinement_edge_from_refinement_rule(rule_type)

    def get_refinement_edge_from_refinement_rule(
        self, rule_type: Type[Rule]
    ) -> RDREdge:
        """
        :param rule_type: The type of the rule to determine the refinement edge from.
        :return: The refinement edge type based on the rule type.
        """
        if isinstance(self.current_parent, MultiClassRefinementRule):
            return RDREdge.Alternative
        if rule_type == MultiClassStopRule:
            return RDREdge.Refinement
        else:
            return RDREdge.Filter

    def get_refinement_rule_type(self, node: ast.AST) -> Type[Rule]:
        """
        :param node: The current AST node to determine the rule type from.
        :return: The rule type based on the node body.
        """
        for stmt in node.body:
            if len(node.body) == 1 and isinstance(stmt, ast.Pass):
                return MultiClassStopRule
            elif isinstance(stmt, ast.If):
                return self.get_refinement_rule_type(stmt)
            else:
                return MultiClassFilterRule
        raise ValueError(
            f"Could not determine the refinement rule type from the node: {node} as it has an empty body."
        )

    def update_current_parent(self, new_node: Rule):
        if isinstance(new_node, MultiClassRefinementRule):
            if isinstance(self.current_parent, MultiClassTopRule):
                new_node.top_rule = self.current_parent
            elif hasattr(self.current_parent, "top_rule"):
                new_node.top_rule = self.current_parent.top_rule
            else:
                raise ValueError(
                    f"Could not set the top rule for the refinement rule: {new_node}"
                )
        if self.current_edge in [RDREdge.Alternative, RDREdge.Next, None]:
            self.current_parent.alternative = new_node
        elif self.current_edge in [RDREdge.Refinement, RDREdge.Filter]:
            self.current_parent.refinement = new_node

    def process_else_statement(self, stmt: ast.AST):
        """Handles the else statement in the if-elif-else block."""
        pass


class RDRWithCodeWriter(RippleDownRules, ABC):

    @classmethod
    def read_rule_tree_from_python(
        cls, model_path: str, python_file_path: Optional[str] = None
    ) -> Rule:
        """
        :param model_path: The path to the generated python file that contains the RDR classifier function.
        :param python_file_path: The path to the generated python file that contains the RDR classifier function.
        """
        if python_file_path is None:
            python_file_path = cls.get_generated_python_file_path(model_path)
        with open(python_file_path, "r") as f:
            source_code = f.read()

        tree = ast.parse(source_code)
        builder = cls.get_tree_builder_class()()

        # Find and process the function
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "classify":
                builder.visit_FunctionDef(node)

        return builder.root

    @classmethod
    @abstractmethod
    def get_tree_builder_class(cls) -> Type[TreeBuilder]:
        """
        :return: The class that builds the rule tree from the generated python file.
        This should be either SingleClassTreeBuilder or MultiClassTreeBuilder.
        """
        pass

    @property
    def all_rules(self) -> List[Rule]:
        """
        Get all rules in the classifier.

        :return: A list of all rules in the classifier.
        """
        if self.start_rule is None:
            return []
        return [
            r
            for r in [self.start_rule] + list(self.start_rule.descendants)
            if r.conditions is not None
        ]

    def update_from_python(
        self,
        model_dir: str,
        parent_package_name: Optional[str] = None,
        python_file_path: Optional[str] = None,
        update_rule_tree: bool = False,
    ):
        """
        Update the rules from the generated python file, that might have been modified by the user.

        :param model_dir: The directory where the generated python file is located.
        :param parent_package_name: The name of the package that contains the RDR classifier function, this
        is required in case of relative imports in the generated python file.
        :param python_file_path: The path to the generated python file that contains the RDR classifier function.
        :param update_rule_tree: Whether to update the rule tree from the python file or not.
        """
        if update_rule_tree:
            self.start_rule = self.read_rule_tree_from_python(
                model_dir, python_file_path=python_file_path
            )
        all_rules = self.all_rules
        condition_func_names = [
            rule.generated_conditions_function_name for rule in all_rules
        ]
        conclusion_func_names = [
            rule.generated_conclusion_function_name
            for rule in all_rules
            if not isinstance(rule, MultiClassStopRule)
        ]
        all_func_names = condition_func_names + conclusion_func_names

        main_module, defs_module, cases_module = (
            self.get_and_import_model_python_modules(
                model_dir,
                python_file_path=python_file_path,
                parent_package_name=parent_package_name,
            )
        )
        self.generated_python_file_name = Path(main_module.__file__).name.replace(
            ".py", ""
        )

        self.update_rdr_metadata_from_python(main_module)

        functions_source = extract_function_or_class_file(
            defs_module.__file__, all_func_names, include_signature=True
        )
        scope = extract_imports(defs_module.__file__, package_name=parent_package_name)

        cases_source, cases_scope = None, None
        if cases_module:
            with open(cases_module.__file__, "r") as f:
                cases_source = f.read()
            cases_scope = extract_imports(
                cases_module.__file__, package_name=parent_package_name
            )

        with open(main_module.__file__, "r") as f:
            main_source = f.read()
        main_scope = extract_imports(
            main_module.__file__, package_name=parent_package_name
        )
        attribute_name_line = [
            l for l in main_source.split("\n") if "attribute_name = " in l
        ]
        conclusion_name = None
        if len(attribute_name_line) > 0:
            conclusion_name = eval(
                attribute_name_line[0].split("=")[-1].strip(), main_scope
            )

        for rule in all_rules:
            if rule.conditions is not None:
                conditions_wrapper_func_name = rule.generated_conditions_function_name
                user_input = functions_source[conditions_wrapper_func_name]
                user_input = "\n".join(
                    user_input.split("\n")[1:]
                )  # Remove the function signature line
                rule.conditions = CallableExpression(user_input, (bool,), scope=scope)
                if cases_module:
                    try:
                        rule.corner_case_metadata = cases_module.__dict__[
                            rule.generated_corner_case_object_name
                        ]
                    except KeyError:
                        case_def_lines = [
                            l
                            for l in cases_source.split("\n")
                            if rule.generated_corner_case_object_name in l
                        ]
                        if len(case_def_lines) > 0:
                            case_def_line = case_def_lines[0].split("=")[-1].strip()
                            rule.corner_case_metadata = eval(case_def_line, cases_scope)

            if not isinstance(rule, MultiClassStopRule):
                if conclusion_name:
                    rule.conclusion_name = conclusion_name
                user_input = functions_source[rule.generated_conclusion_function_name]
                split_user_input = user_input.split("\n")
                tree = ast.parse(user_input)
                user_input = ast.unparse(tree.body[0].body)
                conclusion_func = defs_module.__dict__.get(
                    rule.generated_conclusion_function_name
                )
                if conclusion_func is None:
                    function_signature = split_user_input[0]
                    return_type_hint_str = function_signature.split("->")[-1].strip(
                        " :"
                    )
                    return_type_hint = eval(return_type_hint_str, scope)
                    conclusion_type = get_type_from_type_hint(return_type_hint)
                else:
                    conclusion_type = get_function_return_type(conclusion_func)
                rule.conclusion = CallableExpression(
                    user_input,
                    conclusion_type,
                    scope=scope,
                    mutually_exclusive=self.mutually_exclusive,
                )

    @classmethod
    def get_and_import_model_python_modules(
        cls,
        model_dir: str,
        python_file_path: Optional[str] = None,
        parent_package_name: Optional[str] = None,
    ) -> Tuple[ModuleType, ModuleType, ModuleType]:
        """
        Get and import the python modules that contain the RDR classifier function, definitions, and corner cases.

        :param model_dir: The path to the directory where the generated python files are located.
        :param python_file_path: The path to the generated python file that contains the RDR classifier function.
        :param parent_package_name: The name of the package that contains the RDR classifier function, this
        is required in case of relative imports in the generated python file.
        :return: A tuple containing the main module, defs module, and cases module.
        """
        if python_file_path is None:
            main_file_path = cls.get_generated_python_file_path(model_dir)
        else:
            main_file_path = python_file_path
        if not os.path.exists(main_file_path):
            raise ModuleNotFoundError(main_file_path)

        defs_file_path = main_file_path.replace(".py", "_defs.py")
        cases_path = main_file_path.replace(".py", "_cases.py")

        main_module, defs_module, cases_module = (
            get_and_import_python_modules_in_a_package(
                [main_file_path, defs_file_path, cases_path],
                parent_package_name=parent_package_name,
            )
        )
        return main_module, defs_module, cases_module

    @abstractmethod
    def write_rules_as_source_code_to_file(
        self,
        rule: Rule,
        file,
        parent_indent: str = "",
        defs_file: Optional[str] = None,
        cases_file: Optional[str] = None,
        package_name: Optional[str] = None,
    ):
        """
        Write the rules as source code to a file.

        :param rule: The rule to write as source code.
        :param file: The file to write the source code to.
        :param parent_indent: The indentation of the parent rule.
        :param defs_file: The file to write the definitions to.
        :param cases_file: The file to write the cases to.
        :param package_name: The name of the package that contains the RDR classifier function, this
        is required in case of relative imports in the generated python file.
        """
        pass

    def _write_to_python(self, model_dir: str, package_name: Optional[str] = None):
        """
        Write the tree of rules as source code to a file.

        :param model_dir: The path to the directory to write the source code to.
        :param package_name: The name of the package that contains the RDR classifier function, this
        is required in case of relative imports in the generated python file.
        """
        # Make sure the model directory exists and create an __init__.py file if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(model_dir + "/__init__.py"):
            with open(model_dir + "/__init__.py", "w") as f:
                f.write("from . import *\n")

        # Set the file names for the generated python files
        file_name = model_dir + f"/{self.generated_python_file_name}.py"
        defs_file_name = model_dir + f"/{self.generated_python_defs_file_name}.py"
        cases_file_name = model_dir + f"/{self.generated_python_cases_file_name}.py"

        # Get the required imports for the main file and the defs file
        main_types, defs_types, corner_cases_types = self._get_types_to_import()
        imports = get_imports_from_types(main_types, file_name, package_name)
        defs_imports = get_imports_from_types(defs_types, defs_file_name, package_name)
        corner_cases_imports = get_imports_from_types(
            corner_cases_types, cases_file_name, package_name
        )

        defs_imports.append(f"from krrood.ripple_down_rules import *")
        # Add the imports to the defs file
        with open(defs_file_name, "w") as f:
            f.write("\n".join(defs_imports) + "\n\n\n")

        # Add the imports to the cases file
        case_factory_import = get_imports_from_types(
            [CaseFactoryMetaData], cases_file_name, package_name
        )
        corner_cases_imports.extend(case_factory_import)
        with open(cases_file_name, "w") as cases_f:
            cases_f.write("# This file contains the corner cases for the rules.\n")
            cases_f.write("\n".join(corner_cases_imports) + "\n\n\n")

        # Add the imports, the attributes, and the function definition to the main file
        func_def = f"def classify(case: {self.case_type.__name__}, **kwargs) -> {self.conclusion_type_hint}:\n"
        with open(file_name, "w") as f:
            imports.append(f"from .{self.generated_python_defs_file_name} import *")
            f.write("\n".join(imports) + "\n\n\n")
            f.write(f"attribute_name = '{self.attribute_name}'\n")
            f.write(
                f"conclusion_type = ({', '.join([ct.__name__ for ct in self.conclusion_type])},)\n"
            )
            f.write(f"mutually_exclusive = {self.mutually_exclusive}\n")
            self.write_rdr_metadata_to_pyton_file(f)
            f.write(f"\n\n{func_def}")
            f.write(
                f"{' ' * 4}if not isinstance(case, Case):\n"
                f"{' ' * 4}    case = create_case(case, max_recursion_idx=3)\n"
                f"{' ' * 4}else:\n"
                f"{' ' * 4}    case = copy_case(case)\n"
            )

        # Write the rules as source code to the main file
        self.write_rules_as_source_code_to_file(
            self.start_rule,
            file_name,
            " " * 4,
            defs_file=defs_file_name,
            cases_file=cases_file_name,
            package_name=package_name,
        )

    @property
    @abstractmethod
    def conclusion_type_hint(self) -> str:
        """
        :return: The type hint of the conclusion of the rdr as a string.
        """
        pass

    def _get_types_to_import(
        self,
    ) -> Tuple[Set[Union[Type, Callable]], Set[Type], Set[Type]]:
        """
        :return: The types of the main, defs, and corner cases files of the RDR classifier that will be imported.
        """
        defs_types = set()
        cases_types = set()
        for rule in [self.start_rule] + list(self.start_rule.descendants):
            if not rule.conditions:
                continue
            for scope in [rule.conditions.scope, rule.conclusion.scope]:
                if scope is None:
                    continue
                scope_values = set()
                for scope_value in scope.values():
                    try:
                        hash(scope_value)
                        scope_values.add(scope_value)
                    except TypeError:
                        continue
                defs_types.update(scope_values)
            corner_case_types = rule.get_corner_case_types_to_import()
            if corner_case_types is not None:
                cases_types.update(corner_case_types)
        defs_types.add(self.case_type)
        main_types = set()
        main_types.add(self.case_type)
        main_types.update(make_set(self.conclusion_type))
        main_types.update({Union, Optional})
        defs_types.add(Union)
        main_types.update({Case, create_case, copy_case})
        # main_types = main_types.difference(defs_types)
        return main_types, defs_types, cases_types

    @property
    def _default_generated_python_file_name(self) -> Optional[str]:
        """
        :return: The default generated python file name.
        """
        if self.start_rule is None or self.start_rule.conclusion is None:
            return None
        return f"{str_to_snake_case(self.case_name)}_{self.attribute_name}_{self.get_acronym().lower()}"

    @property
    def generated_python_defs_file_name(self) -> str:
        return f"{self.generated_python_file_name}_defs"

    @property
    def generated_python_cases_file_name(self) -> str:
        return f"{self.generated_python_file_name}_cases"

    @property
    def conclusion_type(self) -> Tuple[Type]:
        """
        :return: The type of the conclusion of the RDR classifier.
        """
        all_types = []
        for rule in self.all_rules:
            all_types.extend(list(rule.conclusion.conclusion_type))
        return tuple(set(all_types))

    @property
    def attribute_name(self) -> str:
        """
        :return: The name of the attribute that the classifier is classifying.
        """
        return self.start_rule.conclusion_name

    def _to_json(self) -> Dict[str, Any]:
        return {
            "start_rule": self.start_rule.to_json(),
            "generated_python_file_name": self.generated_python_file_name,
            "name": self.name,
            "case_type": (
                get_full_class_name(self.case_type)
                if self.case_type is not None
                else None
            ),
            "case_name": self.case_name,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create an instance of the class from a json
        """
        start_rule = cls.start_rule_type().from_json(data["start_rule"])
        new_rdr = cls(start_rule=start_rule)
        if "generated_python_file_name" in data:
            new_rdr.generated_python_file_name = data["generated_python_file_name"]
        if "name" in data:
            new_rdr.name = data["name"]
        if "case_type" in data:
            new_rdr.case_type = get_type_from_string(data["case_type"])
        if "case_name" in data:
            new_rdr.case_name = data["case_name"]
        return new_rdr

    @staticmethod
    @abstractmethod
    def start_rule_type() -> Type[Rule]:
        """
        :return: The type of the starting rule of the RDR classifier.
        """
        pass


class SingleClassRDR(RDRWithCodeWriter):
    mutually_exclusive: bool = True
    """
    The output of the classification of this rdr negates all other possible outputs, there can only be one true value.
    """

    def __init__(self, default_conclusion: Optional[Any] = None, **kwargs):
        """
        :param start_rule: The starting rule for the classifier.
        :param default_conclusion: The default conclusion for the classifier if no rules fire.
        """
        super(SingleClassRDR, self).__init__(**kwargs)
        self.default_conclusion: Optional[Any] = default_conclusion

    @classmethod
    def get_tree_builder_class(cls) -> Type[TreeBuilder]:
        return SingleClassTreeBuilder

    def _fit_case(
        self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs
    ) -> Union[CaseAttribute, CallableExpression, None]:
        """
        Classify a case, and ask the user for refinements or alternatives if the classification is incorrect by
        comparing the case with the target category if provided.

        :param case_query: The case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The category that the case belongs to.
        """
        if (
            case_query.default_value is not None
            and self.default_conclusion != case_query.default_value
        ):
            self.default_conclusion = case_query.default_value

        pred = self.evaluate(case_query.case)
        if (not pred.fired and self.default_conclusion is None) or pred.conclusion(
            case_query.case
        ) != case_query.target_value:
            expert.ask_for_conditions(case_query, pred)
            pred.fit_rule(case_query)

        return self.classify(case_query.case)

    def update_start_rule(self, case_query: CaseQuery, expert: Expert):
        """
        Update the starting rule of the classifier.

        :param case_query: The case query to update the starting rule with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if not self.start_rule:
            expert.ask_for_conditions(case_query)
            self.start_rule = SingleClassRule.from_case_query(case_query)

    def _classify(
        self,
        case: Case,
        modify_case: bool = False,
        case_query: Optional[CaseQuery] = None,
    ) -> Optional[Any]:
        """
        Classify a case by recursively evaluating the rules until a rule fires or the last rule is reached.

        :param case: The case to classify.
        :param modify_case: Whether to modify the original case attributes with the conclusion or not.
        :param case_query: The case query containing the case and the target category to compare the case with.
        """
        pred = self.evaluate(case)
        conclusion = (
            pred.conclusion(case)
            if pred is not None and pred.fired
            else self.default_conclusion
        )
        if pred is not None and pred.fired:
            pred.contributed = True
            pred.last_conclusion = conclusion
            if case_query is not None:
                pred.contributed_to_case_query = True
        if pred is not None and pred.fired and case_query is not None:
            if (
                pred.corner_case_metadata is None
                and conclusion is not None
                and type(conclusion) in case_query.core_attribute_type
            ):
                pred.corner_case_metadata = CaseFactoryMetaData.from_case_query(
                    case_query
                )
        return conclusion

    def evaluate(self, case: Case) -> SingleClassRule:
        """
        Evaluate the starting rule on a case.
        """
        matched_rule = self.start_rule(case) if self.start_rule is not None else None
        return matched_rule if matched_rule is not None else self.start_rule

    def _write_to_python(self, model_dir: str, package_name: Optional[str] = None):
        super()._write_to_python(model_dir, package_name=package_name)
        with open(model_dir + f"/{self.generated_python_file_name}.py", "a") as f:
            f.write(f"{' ' * 4}else:\n{' ' * 8}return {self.default_conclusion}\n")

    def write_rules_as_source_code_to_file(
        self,
        rule: SingleClassRule,
        filename: str,
        parent_indent: str = "",
        defs_file: Optional[str] = None,
        cases_file: Optional[str] = None,
        package_name: Optional[str] = None,
    ):
        """
        Write the rules as source code to a file.
        """
        if rule.conditions:
            rule.write_corner_case_as_source_code(cases_file, package_name=package_name)
            if_clause = rule.write_condition_as_source_code(parent_indent, defs_file)
            with open(filename, "a") as file:
                file.write(if_clause)
            if rule.refinement:
                self.write_rules_as_source_code_to_file(
                    rule.refinement,
                    filename,
                    parent_indent + "    ",
                    defs_file=defs_file,
                    cases_file=cases_file,
                    package_name=package_name,
                )

            conclusion_call = rule.write_conclusion_as_source_code(
                parent_indent, defs_file
            )
            with open(filename, "a") as file:
                file.write(conclusion_call)

            if rule.alternative:
                self.write_rules_as_source_code_to_file(
                    rule.alternative,
                    filename,
                    parent_indent,
                    defs_file=defs_file,
                    cases_file=cases_file,
                    package_name=package_name,
                )

    @property
    def conclusion_type_hint(self) -> str:
        all_types = set(list(self.conclusion_type) + [type(self.default_conclusion)])
        if NoneType in all_types:
            return f"Optional[{', '.join([t.__name__ for t in all_types if t is not NoneType])}]"
        return f"Union[{', '.join([t.__name__ for t in all_types])}]"

    def _get_types_to_import(self) -> Tuple[Set[Type], Set[Type], Set[Type]]:
        main_types, def_types, case_types = super()._get_types_to_import()
        main_types.add(type(self.default_conclusion))
        def_types.add(type(self.default_conclusion))
        if self.default_conclusion is None:
            main_types.add(Optional)
            def_types.add(Optional)
        return main_types, def_types, case_types

    @property
    def conclusion_type(self) -> Tuple[Type]:
        if self.default_conclusion is not None:
            return (type(self.default_conclusion),)
        return super().conclusion_type

    @staticmethod
    def start_rule_type() -> Type[Rule]:
        """
        :return: The type of the starting rule of the RDR classifier.
        """
        return SingleClassRule


class MultiClassRDR(RDRWithCodeWriter):
    """
    A multi class ripple down rules classifier, which can draw multiple conclusions for a case.
    This is done by going through all rules and checking if they fire or not, and adding stopping rules if needed,
    when wrong conclusions are made to stop these rules from firing again for similar cases.
    """

    evaluated_rules: Optional[List[Rule]] = None
    """
    The evaluated rules in the classifier for one case.
    """
    conclusions: Optional[List[CaseAttribute]] = None
    """
    The conclusions that the case belongs to.
    """
    stop_rule_conditions: Optional[CallableExpression] = None
    """
    The conditions of the stopping rule if needed.
    """
    mutually_exclusive: bool = False
    """
    The output of the classification of this rdr allows for more than one true value as conclusion.
    """

    def __init__(
        self,
        start_rule: Optional[MultiClassTopRule] = None,
        mode: MCRDRMode = MCRDRMode.StopOnly,
        **kwargs,
    ):
        """
        :param start_rule: The starting rules for the classifier.
        :param mode: The mode of the classifier, either StopOnly or StopPlusRule, or StopPlusRuleCombined.
        """
        super(MultiClassRDR, self).__init__(start_rule, **kwargs)
        self.mode: MCRDRMode = mode

    @classmethod
    def get_tree_builder_class(cls) -> Type[TreeBuilder]:
        return MultiClassTreeBuilder

    def _classify(
        self,
        case: Union[Case, SQLTable],
        modify_case: bool = False,
        case_query: Optional[CaseQuery] = None,
    ) -> Set[Any]:
        evaluated_rule = self.start_rule
        self.conclusions = []
        case_cp = copy_case(case) if not modify_case else case
        while evaluated_rule:
            next_rule = evaluated_rule(case_cp)
            if evaluated_rule.fired:
                rule_conclusion = evaluated_rule.conclusion(case_cp)
                if rule_conclusion:
                    rule_conclusion = make_set(rule_conclusion) - make_set(
                        self.conclusions
                    )
                if (
                    evaluated_rule.corner_case_metadata is None
                    and case_query is not None
                ):
                    if (
                        rule_conclusion is not None
                        and len(make_list(rule_conclusion)) > 0
                        and any(
                            ct in case_query.core_attribute_type
                            for ct in map(type, make_list(rule_conclusion))
                        )
                    ):
                        evaluated_rule.corner_case_metadata = (
                            CaseFactoryMetaData.from_case_query(case_query)
                        )
                if rule_conclusion is not None and any(make_list(rule_conclusion)):
                    evaluated_rule.contributed = True
                    evaluated_rule.last_conclusion = rule_conclusion
                    if case_query is not None:
                        rule_conclusion_types = set(
                            map(type, make_list(rule_conclusion))
                        )
                        if are_results_subclass_of_types(
                            rule_conclusion_types, case_query.core_attribute_type
                        ):
                            evaluated_rule.contributed_to_case_query = True
                self.add_conclusion(rule_conclusion)
                if rule_conclusion:
                    if case_query and modify_case:
                        update_case_in_case_query(case_query, rule_conclusion)
                    else:
                        update_case_with_conclusion_output(
                            case_cp,
                            rule_conclusion,
                            self.attribute_name,
                            self.conclusion_type,
                            self.mutually_exclusive,
                        )
            evaluated_rule = next_rule
        return make_set(self.conclusions)

    def _fit_case(
        self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs
    ) -> Set[Union[CaseAttribute, CallableExpression, None]]:
        """
        Classify a case, and ask the user for stopping rules or classifying rules if the classification is incorrect
         or missing by comparing the case with the target category if provided.

        :param case_query: The query containing the case to classify and the target category to compare the case with.
        :param expert: The expert to ask for differentiating features as new rule conditions or for extra conclusions.
        :return: The conclusions that the case belongs to.
        """
        self.conclusions = []
        self.stop_rule_conditions = None
        evaluated_rule = self.start_rule
        target_value = make_set(case_query.target_value)
        while evaluated_rule:
            next_rule = evaluated_rule(case_query.case)
            if evaluated_rule.fired:
                rule_conclusion = evaluated_rule.conclusion(case_query.case)
                if not make_set(rule_conclusion).issubset(target_value):
                    # Rule fired and conclusion is different from target
                    self.stop_wrong_conclusion_else_add_it(
                        case_query, expert, evaluated_rule
                    )
                else:
                    # Rule fired and target is correct or there is no target to compare
                    self.add_conclusion(rule_conclusion)

            if not next_rule:
                if not make_set(target_value).issubset(make_set(self.conclusions)):
                    # Nothing fired and there is a target that should have been in the conclusions
                    self.add_rule_for_case(case_query, expert)
                    # Have to check all rules again to make sure only this new rule fires
                    next_rule = self.start_rule
            evaluated_rule = next_rule
        return self.conclusions

    def write_rules_as_source_code_to_file(
        self,
        rule: Union[MultiClassTopRule, MultiClassStopRule],
        filename: str,
        parent_indent: str = "",
        defs_file: Optional[str] = None,
        cases_file: Optional[str] = None,
        package_name: Optional[str] = None,
    ):
        if rule == self.start_rule:
            with open(filename, "a") as file:
                file.write(f"{parent_indent}conclusions = set()\n")
        if rule.conditions:
            rule.write_corner_case_as_source_code(cases_file, package_name=package_name)
            if_clause = rule.write_condition_as_source_code(parent_indent, defs_file)
            with open(filename, "a") as file:
                file.write(if_clause)
            conclusion_indent = parent_indent
            if hasattr(rule, "refinement") and rule.refinement:
                self.write_rules_as_source_code_to_file(
                    rule.refinement,
                    filename,
                    parent_indent + "    ",
                    defs_file=defs_file,
                    cases_file=cases_file,
                    package_name=package_name,
                )
                conclusion_indent = parent_indent + " " * 4
                with open(filename, "a") as file:
                    file.write(f"{conclusion_indent}else:\n")

            conclusion_call = rule.write_conclusion_as_source_code(
                conclusion_indent, defs_file
            )
            with open(filename, "a") as file:
                file.write(conclusion_call)

            if rule.alternative:
                self.write_rules_as_source_code_to_file(
                    rule.alternative,
                    filename,
                    parent_indent,
                    defs_file=defs_file,
                    cases_file=cases_file,
                    package_name=package_name,
                )
            elif isinstance(rule, MultiClassTopRule):
                with open(filename, "a") as file:
                    file.write(f"{parent_indent}return conclusions\n")

    @property
    def conclusion_type_hint(self) -> str:
        conclusion_types = [
            ct.__name__ for ct in self.conclusion_type if ct not in [list, set]
        ]
        if len(conclusion_types) == 1:
            return f"Set[{conclusion_types[0]}]"
        else:
            return f"Set[Union[{', '.join(conclusion_types)}]]"

    def _get_types_to_import(
        self,
    ) -> Tuple[Set[Union[Type, Callable]], Set[Type], Set[Type]]:
        main_types, defs_types, cases_types = super()._get_types_to_import()
        main_types.update(
            {get_an_updated_case_copy, update_case_and_conclusions_with_rule_output}
        )
        main_types.update({Set, make_set})
        defs_types.update({List, Set})
        return main_types, defs_types, cases_types

    def update_start_rule(self, case_query: CaseQuery, expert: Expert):
        """
        Update the starting rule of the classifier.

        :param case_query: The case query to update the starting rule with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if not self.start_rule:
            conditions = expert.ask_for_conditions(case_query)
            self.start_rule: MultiClassTopRule = MultiClassTopRule.from_case_query(
                case_query
            )

    @property
    def last_top_rule(self) -> Optional[MultiClassTopRule]:
        """
        Get the last top rule in the tree.
        """
        if not self.start_rule.furthest_alternative:
            return self.start_rule
        else:
            return self.start_rule.furthest_alternative[-1]

    def stop_wrong_conclusion_else_add_it(
        self, case_query: CaseQuery, expert: Expert, evaluated_rule: MultiClassTopRule
    ):
        """
        Stop a wrong conclusion by adding a stopping rule.
        """
        rule_conclusion = evaluated_rule.conclusion(case_query.case)
        stop: bool = False
        add_filter_rule: bool = False
        if is_value_conflicting(rule_conclusion, case_query.target_value):
            if make_set(case_query.target_value).issubset(rule_conclusion):
                add_filter_rule = True
            else:
                stop = True
        elif make_set(case_query.core_attribute_type).issubset(
            make_set(evaluated_rule.conclusion.conclusion_type)
        ):
            if make_set(case_query.target_value).issubset(rule_conclusion):
                add_filter_rule = True

        if not stop:
            self.add_conclusion(rule_conclusion)
        if stop or add_filter_rule:
            refinement_type = MultiClassStopRule if stop else MultiClassFilterRule
            self.stop_or_filter_conclusion(
                case_query, expert, evaluated_rule, refinement_type=refinement_type
            )

    def stop_or_filter_conclusion(
        self,
        case_query: CaseQuery,
        expert: Expert,
        evaluated_rule: MultiClassTopRule,
        refinement_type: Type[MultiClassRefinementRule] = MultiClassStopRule,
    ):
        """
        Stop a conclusion by adding a stopping rule.

        :param case_query: The case query to stop the conclusion for.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :param evaluated_rule: The evaluated rule to ask the expert about.
        :param refinement_type: The refinement type to use.
        """
        conditions = expert.ask_for_conditions(case_query, evaluated_rule)
        evaluated_rule.fit_rule(case_query, refinement_type=refinement_type)
        if refinement_type is MultiClassStopRule:
            if self.mode == MCRDRMode.StopPlusRule:
                self.stop_rule_conditions = conditions
            if self.mode == MCRDRMode.StopPlusRuleCombined:
                new_top_rule_conditions = conditions.combine_with(
                    evaluated_rule.conditions
                )
                case_query.conditions = new_top_rule_conditions
                self.add_top_rule(case_query)

    def add_rule_for_case(self, case_query: CaseQuery, expert: Expert):
        """
        Add a rule for a case that has not been classified with any conclusion.

        :param case_query: The case query to add the rule for.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if self.stop_rule_conditions and self.mode == MCRDRMode.StopPlusRule:
            conditions = self.stop_rule_conditions
            self.stop_rule_conditions = None
            case_query.conditions = conditions
        else:
            conditions = expert.ask_for_conditions(case_query)
        self.add_top_rule(case_query)

    def add_conclusion(self, rule_conclusion: List[Any]) -> None:
        """
        Add the conclusion of the evaluated rule to the list of conclusions.

        :param rule_conclusion: The conclusion of the evaluated rule, which can be a single conclusion
         or a set of conclusions.
        """
        conclusion_types = [type(c) for c in self.conclusions]
        if type(rule_conclusion) not in conclusion_types:
            self.conclusions.extend(make_list(rule_conclusion))
        else:
            same_type_conclusions = [
                c for c in self.conclusions if type(c) == type(rule_conclusion)
            ]
            combined_conclusion = (
                rule_conclusion
                if isinstance(rule_conclusion, set)
                else {rule_conclusion}
            )
            combined_conclusion = copy(combined_conclusion)
            for c in same_type_conclusions:
                combined_conclusion.update(c if isinstance(c, set) else make_set(c))
                self.conclusions.remove(c)
            self.conclusions.extend(make_list(combined_conclusion))

    def add_top_rule(self, case_query: CaseQuery):
        """
        Add a top rule to the classifier, which is a rule that is always checked and is part of the start_rules list.

        :param case_query: The case query to add the top rule for.
        """
        self.start_rule.alternative = MultiClassTopRule.from_case_query(case_query)

    @staticmethod
    def start_rule_type() -> Type[Rule]:
        """
        :return: The type of the starting rule of the RDR classifier.
        """
        return MultiClassTopRule


class GeneralRDR(RippleDownRules):
    """
    A general ripple down rules classifier, which can draw multiple conclusions for a case, but each conclusion is part
    of a set of mutually exclusive conclusions. Whenever a conclusion is made, the classification restarts from the
    starting rule, and all the rules that belong to the class of the made conclusion are not checked again. This
    continues until no more rules can be fired. In addition, previous conclusions can be used as conditions or input to
    the next classification/cycle.
    Another possible mode is to have rules that are considered final, when fired, inference will not be restarted,
     and only a refinement can be made to the final rule, those can also be used in another SCRDR of their own that
     gets called when the final rule fires.
    """

    def __init__(
        self,
        category_rdr_map: Optional[
            Dict[str, Union[SingleClassRDR, MultiClassRDR]]
        ] = None,
        **kwargs,
    ):
        """
        :param category_rdr_map: A map of case attribute names to ripple down rules classifiers,
        where each category is a parent category that has a set of mutually exclusive (in case of SCRDR) child
        categories, e.g. {'species': SCRDR, 'habitats': MCRDR}, where 'species' and 'habitats' are attribute names
        for a case of type Animal, while SCRDR and MCRDR are SingleClass and MultiClass ripple down rules classifiers.
        Species can have values like Mammal, Bird, Fish, etc. which are mutually exclusive, while Habitat can have
        values like Land, Water, Air, etc., which are not mutually exclusive due to some animals living more than one
        habitat.
        """
        self.start_rules_dict: Dict[str, Union[SingleClassRDR, MultiClassRDR]] = (
            category_rdr_map if category_rdr_map else {}
        )
        super(GeneralRDR, self).__init__(**kwargs)
        self.all_figs: List[Figure] = [sr.fig for sr in self.start_rules_dict.values()]

    @classmethod
    def from_python(
        cls,
        model_dir: str,
        python_file_path: Optional[str] = None,
        parent_package_name: Optional[str] = None,
    ) -> Self:
        """
        Create an instance of the class from a python file.

        :param model_dir: The path to the directory containing the python file.
        :param python_file_path: The path to the python file, if not provided, it will be generated from the model_dir.
        :param parent_package_name: The name of the package that contains the RDR classifier function, this
        is required in case of relative imports in the generated python file.
        :return: An instance of the class.
        """
        grdr = cls()
        grdr.update_from_python(
            model_dir,
            parent_package_name=parent_package_name,
            python_file_path=python_file_path,
            update_rule_tree=True,
        )
        return grdr

    @classmethod
    def get_rdr_type_from_acronym(
        cls, acronym: str
    ) -> Type[Union[SingleClassRDR, MultiClassRDR]]:
        """
        Get the type of the ripple down rules classifier from the acronym.

        :param acronym: The acronym of the ripple down rules classifier.
        :return: The type of the ripple down rules classifier.
        """
        acronym = acronym.lower()
        if acronym == "scrdr":
            return SingleClassRDR
        elif acronym == "mcrdr":
            return MultiClassRDR
        else:
            raise ValueError(f"Unknown RDR type acronym: {acronym}")

    def add_rdr(
        self,
        rdr: Union[SingleClassRDR, MultiClassRDR],
        case_query: Optional[CaseQuery] = None,
    ):
        """
        Add a ripple down rules classifier to the map of classifiers.

        :param rdr: The ripple down rules classifier to add.
        :param case_query: The case query to add the classifier for.
        """
        name = case_query.attribute_name if case_query else rdr.name
        self.start_rules_dict[name] = rdr

    @property
    def start_rule(self) -> Optional[Union[SingleClassRule, MultiClassTopRule]]:
        return self.start_rules[0] if self.start_rules_dict else None

    @start_rule.setter
    def start_rule(self, value: Union[SingleClassRDR, MultiClassRDR]):
        if value:
            self.start_rules_dict[value.attribute_name] = value

    @property
    def start_rules(self) -> List[Union[SingleClassRule, MultiClassTopRule]]:
        return [rdr.start_rule for rdr in self.start_rules_dict.values()]

    def _classify(
        self,
        case: Any,
        modify_case: bool = False,
        case_query: Optional[CaseQuery] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Classify a case by going through all RDRs and adding the categories that are classified, and then restarting
        the classification until no more categories can be added.

        :param case: The case to classify.
        :param modify_case: Whether to modify the original case or create a copy and modify it.
        :param case_query: The case query containing the case and the target category to compare the case with.
        :return: The categories that the case belongs to.
        """
        return general_rdr_classify(
            self.start_rules_dict,
            case,
            modify_original_case=modify_case,
            case_query=case_query,
        )

    def _fit_case(
        self, case_query: CaseQuery, expert: Optional[Expert] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Fit the GRDR on a case, if the target is a new type of category, a new RDR is created for it,
        else the existing RDR of that type will be fitted on the case, and then classification is done and all
        concluded categories are returned. If the category is mutually exclusive, an SCRDR is created, else an MCRDR.
        In case of SCRDR, multiple conclusions of the same type replace each other, in case of MCRDR, they are added if
        they are accepted by the expert, and the attribute of that category is represented in the case as a set of
        values.

        :param case_query: The query containing the case to classify and the target category to compare the case
        with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        :return: The categories that the case belongs to.
        """
        case_query_cp = copy(case_query)
        self.classify(case_query_cp.case, modify_case=True)
        case_query_cp.update_target_value()

        self.start_rules_dict[case_query_cp.attribute_name].fit_case(
            case_query_cp, expert, **kwargs
        )

        return self.classify(case_query.case)

    def update_start_rule(self, case_query: CaseQuery, expert: Expert):
        """
        Update the starting rule of the classifier.

        :param case_query: The case query to update the starting rule with.
        :param expert: The expert to ask for differentiating features as new rule conditions.
        """
        if case_query.attribute_name not in self.start_rules_dict:
            new_rdr = self.initialize_new_rdr_for_attribute(case_query)
            self.add_rdr(new_rdr, case_query)

    @staticmethod
    def initialize_new_rdr_for_attribute(case_query: CaseQuery):
        """
        Initialize the appropriate RDR type for the target.
        """
        return (
            SingleClassRDR(default_conclusion=case_query.default_value)
            if case_query.mutually_exclusive
            else MultiClassRDR()
        )

    def _to_json(self) -> Dict[str, Any]:
        return {
            "start_rules": {
                name: rdr.to_json() for name, rdr in self.start_rules_dict.items()
            },
            "generated_python_file_name": self.generated_python_file_name,
            "name": self.name,
            "case_type": (
                get_full_class_name(self.case_type)
                if self.case_type is not None
                else None
            ),
            "case_name": self.case_name,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> GeneralRDR:
        """
        Create an instance of the class from a json
        """
        start_rules_dict = {}
        for k, v in data["start_rules"].items():
            start_rules_dict[k] = get_type_from_string(v["_type"]).from_json(v)
        new_rdr = cls(category_rdr_map=start_rules_dict)
        if "generated_python_file_name" in data:
            new_rdr.generated_python_file_name = data["generated_python_file_name"]
        if "name" in data:
            new_rdr.name = data["name"]
        if "case_type" in data:
            new_rdr.case_type = get_type_from_string(data["case_type"])
        if "case_name" in data:
            new_rdr.case_name = data["case_name"]
        return new_rdr

    def update_from_python(
        self,
        model_dir: str,
        parent_package_name: Optional[str] = None,
        python_file_path: Optional[str] = None,
        update_rule_tree: bool = False,
    ) -> None:
        """
        Update the rules from the generated python file, that might have been modified by the user.

        :param model_dir: The directory where the model is stored.
        :param parent_package_name: The name of the package that contains the RDR classifier function, this
        is required in case of relative imports in the generated python file.
        :param python_file_path: The path to the python file, if not provided, it will be generated from the model_dir.
        :param update_rule_tree: Whether to update the rule tree from the python file or not.
        """
        if update_rule_tree:
            if python_file_path is None:
                main_python_file_path = self.get_generated_python_file_path(model_dir)
            else:
                main_python_file_path = python_file_path
            main_module = get_and_import_python_module(
                main_python_file_path, parent_package_name=parent_package_name
            )
            classifiers_dict = main_module.classifiers_dict
            self.start_rules_dict = {}
            for rdr_name, rdr_module in classifiers_dict.items():
                rdr_acronym = rdr_module.__name__.split("_")[-1]
                rdr_type = self.get_rdr_type_from_acronym(rdr_acronym)
                rdr_model_path = main_python_file_path.replace(
                    "_rdr.py", f"_{rdr_name}_{rdr_acronym}.py"
                )
                rdr = rdr_type.from_python(
                    model_dir,
                    python_file_path=rdr_model_path,
                    parent_package_name=parent_package_name,
                )
                self.start_rules_dict[rdr_name] = rdr

            self.update_rdr_metadata_from_python(main_module)
        else:
            for rdr in self.start_rules_dict.values():
                rdr.update_from_python(
                    model_dir, parent_package_name=parent_package_name
                )

    def _write_to_python(
        self, model_dir: str, package_name: Optional[str] = None
    ) -> None:
        """
        Write the tree of rules as source code to a file.

        :param model_dir: The directory where the model is stored.
        :param package_name: The name of the package that contains the RDR classifier function.
        """
        for rdr in self.start_rules_dict.values():
            rdr._write_to_python(model_dir, package_name=package_name)
        func_def = f"def classify(case: {self.case_type.__name__}, **kwargs) -> {self.conclusion_type_hint}:\n"
        file_path = model_dir + f"/{self.generated_python_file_name}.py"
        with open(file_path, "w") as f:
            f.write(
                self._get_imports(file_path=file_path, package_name=package_name)
                + "\n\n"
            )
            self.write_rdr_metadata_to_pyton_file(f)
            f.write("classifiers_dict = dict()\n")
            for rdr_key, rdr in self.start_rules_dict.items():
                f.write(
                    f"classifiers_dict['{rdr_key}'] = {self.rdr_key_to_function_name(rdr_key)}\n"
                )
            f.write("\n\n")
            f.write(func_def)
            f.write(
                f"{' ' * 4}if not isinstance(case, Case):\n"
                f"{' ' * 4}    case = create_case(case, max_recursion_idx=3)\n"
                ""
            )
            f.write(
                f"{' ' * 4}return general_rdr_classify(classifiers_dict, case, **kwargs)\n"
            )

    @property
    def _default_generated_python_file_name(self) -> Optional[str]:
        """
        :return: The default generated python file name.
        """
        if self.start_rule is None or self.start_rule.conclusion is None:
            return None
        return f"{str_to_snake_case(self.case_name)}_rdr".lower()

    @property
    def conclusion_type_hint(self) -> str:
        return f"Dict[str, Any]"

    def _get_imports(
        self, file_path: Optional[str] = None, package_name: Optional[str] = None
    ) -> str:
        """
        Get the imports needed for the generated python file.

        :param file_path: The path to the file where the imports will be written, if None, the imports will be absolute.
        :param package_name: The name of the package that contains the RDR classifier function, this
        is required in case of relative imports in the generated python file.
        :return: The imports needed for the generated python file.
        """
        all_types = set()
        # add type hints
        all_types.update({Dict, Any})
        # import rdr type
        all_types.add(general_rdr_classify)
        # add case type
        all_types.update({Case, create_case, self.case_type})
        # get the imports from the types
        imports = get_imports_from_types(
            all_types, target_file_path=file_path, package_name=package_name
        )
        # add rdr python generated functions.
        for rdr_key, rdr in self.start_rules_dict.items():
            imports.append(
                f"from . import {rdr.generated_python_file_name} as {self.rdr_key_to_function_name(rdr_key)}"
            )
        return "\n".join(imports)

    @staticmethod
    def rdr_key_to_function_name(rdr_key: str) -> str:
        """
        Convert the RDR key to a function name.

        :param rdr_key: The RDR key to convert.
        :return: The function name.
        """
        return rdr_key.replace(".", "_").lower() + "_classifier"

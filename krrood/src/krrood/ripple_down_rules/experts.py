from __future__ import annotations

import ast
import importlib
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from textwrap import dedent, indent
from typing_extensions import Tuple, Dict

from typing_extensions import Optional, TYPE_CHECKING, List

from krrood.ripple_down_rules.datastructures.callable_expression import CallableExpression
from krrood.ripple_down_rules.datastructures.case import show_current_and_corner_cases
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.datastructures.enums import PromptFor
from krrood.ripple_down_rules.user_interface.template_file_creator import TemplateFileCreator
from krrood.ripple_down_rules.utils import extract_imports, extract_function_or_class_file, get_imports_from_scope, get_class_file_path

try:
    from .user_interface.gui import RDRCaseViewer
except ImportError as e:
    RDRCaseViewer = None
from krrood.ripple_down_rules.user_interface.prompt import UserPrompt

if TYPE_CHECKING:
    from .rdr import Rule


class Expert(ABC):
    """
    The Abstract Expert class, all experts should inherit from this class.
    An expert is a class that can provide differentiating features and conclusions for a case when asked.
    The expert can compare a case with a corner case and provide the differentiating features and can also
    provide one or multiple conclusions for a case.
    """
    all_expert_answers: Optional[List] = None
    """
    A list of all expert answers, used for testing purposes.
    """
    use_loaded_answers: bool = False
    """
    A flag to indicate if the expert should use loaded answers or not.
    """

    def __init__(self, use_loaded_answers: bool = False,
                 append: bool = False,
                 answers_save_path: Optional[str] = None):
        self.all_expert_answers = []
        self.use_loaded_answers = use_loaded_answers
        self.answers_save_path = answers_save_path
        if answers_save_path is not None and os.path.exists(answers_save_path + '.py'):
            if use_loaded_answers:
                self.load_answers(answers_save_path)
            if not append:
                os.remove(answers_save_path + '.py')
        self.append = True

    @abstractmethod
    def ask_for_conditions(self, case_query: CaseQuery, last_evaluated_rule: Optional[Rule] = None) \
            -> CallableExpression:
        """
        Ask the expert to provide the differentiating features between two cases or unique features for a case
        that doesn't have a corner case to compare to.

        :param case_query: The case query containing the case to classify and the required target.
        :param last_evaluated_rule: The last evaluated rule.
        :return: The differentiating features as new rule conditions.
        """
        pass

    @abstractmethod
    def ask_for_conclusion(self, case_query: CaseQuery) -> Optional[CallableExpression]:
        """
        Ask the expert to provide a relational conclusion for the case.

        :param case_query: The case query containing the case to find a conclusion for.
        :return: A callable expression that can be called with a new case as an argument.
        """

    def clear_answers(self, path: Optional[str] = None):
        """
        Clear the expert answers.

        :param path: The path to clear the answers from. If None, the answers will be cleared from the
                     answers_save_path attribute.
        """
        if path is None and self.answers_save_path is None:
            raise ValueError("No path provided to clear expert answers, either provide a path or set the "
                             "answers_save_path attribute.")
        if path is None:
            path = self.answers_save_path
        if os.path.exists(path + '.py'):
            os.remove(path + '.py')
        self.all_expert_answers = []

    def save_answers(self, path: Optional[str] = None, expert_answers: Optional[List[Tuple[Dict, str]]] = None):
        """
        Save the expert answers to a file.

        :param path: The path to save the answers to.
        :param expert_answers: The expert answers to save.
        """
        expert_answers = expert_answers if expert_answers else self.all_expert_answers
        if not any(expert_answers):
            return
        if path is None and self.answers_save_path is None:
            raise ValueError("No path provided to save expert answers, either provide a path or set the "
                             "answers_save_path attribute.")
        if path is None:
            path = self.answers_save_path
        self._save_to_python(path, expert_answers=expert_answers)

    def _save_to_json(self, path: str):
        """
        Save the expert answers to a JSON file.

        :param path: The path to save the answers to.
        """
        all_answers = self.all_expert_answers
        if self.append and os.path.exists(path + '.json'):
            # read the file and append the new answers
            with open(path + '.json', "r") as f:
                old_answers = json.load(f)
                all_answers = old_answers + all_answers
        with open(path + '.json', "w") as f:
            json.dump(all_answers, f)

    def _save_to_python(self, path: str, expert_answers: Optional[List[Tuple[Dict, str]]] = None):
        """
        Save the expert answers to a Python file.

        :param path: The path to save the answers to.
        :param expert_answers: The expert answers to save.
        """
        expert_answers = expert_answers if expert_answers else self.all_expert_answers
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name + '/__init__.py'):
            os.makedirs(dir_name, exist_ok=True)
            with open(dir_name + '/__init__.py', 'w') as f:
                f.write('# This is an empty init file to make the directory a package.\n')
        # Current file data
        current_file_data = None
        if os.path.exists(path + '.py'):
            with open(path + '.py', 'r') as f:
                current_file_data = f.read()
        action = 'a' if self.append and current_file_data is not None else 'w'
        with open(path + '.py', action) as f:
            for scope, func_source in expert_answers:
                if len(scope) > 0:
                    imports = '\n'.join(get_imports_from_scope(scope)) + '\n\n\n'
                else:
                    imports = ''
                if func_source is None:
                    func_source = 'pass  # No user input provided for this case.\n'
                f.write(imports + func_source + '\n' + '\n\n\n\'===New Answer===\'\n\n\n')

    def load_answers(self, path: Optional[str] = None):
        """
        Load the expert answers from a file.

        :param path: The path to load the answers from.
        """
        if path is None and self.answers_save_path is None:
            raise ValueError("No path provided to load expert answers from, either provide a path or set the "
                             "answers_save_path attribute.")
        if path is None:
            path = self.answers_save_path
        if os.path.exists(path + '.py'):
            self._load_answers_from_python(path)
        elif os.path.exists(path + '.json'):
            self._load_answers_from_json(path)

    def _load_answers_from_json(self, path: str):
        """
        Load the expert answers from a JSON file.

        :param path: The path to load the answers from.
        """
        with open(path + '.json', "r") as f:
            all_answers = json.load(f)
        self.all_expert_answers = [({}, answer) for answer in all_answers]

    def _load_answers_from_python(self, path: str):
        """
        Load the expert answers from a Python file.

        :param path: The path to load the answers from.
        """
        file_path = path + '.py'
        with open(file_path, "r") as f:
            all_answers = f.read().split('\n\n\n\'===New Answer===\'\n\n\n')[:-1]
        all_function_sources = extract_function_or_class_file(file_path, [], as_list=True)
        for i, answer in enumerate(all_answers):
            answer = answer.strip('\n').strip()
            if 'def ' not in answer and 'pass' in answer:
                self.all_expert_answers.append(({}, None))
                continue
            scope = extract_imports(tree=ast.parse(answer))
            func_name = all_function_sources[i].split('def ')[1].split('(')[0]
            function_source = all_function_sources[i].replace(func_name,
                                                              CallableExpression.encapsulating_function_name)
            self.all_expert_answers.append((scope, function_source))


class AI(Expert):
    """
    The AI Expert class, an expert that uses AI to provide differentiating features and conclusions.
    """

    def __init__(self, **kwargs):
        """
        Initialize the AI expert.
        """
        super().__init__(**kwargs)
        self.user_prompt = UserPrompt()

    def ask_for_conditions(self, case_query: CaseQuery,
                           last_evaluated_rule: Optional[Rule] = None) \
            -> CallableExpression:
        prompt_str = self.get_prompt_for_ai(case_query, PromptFor.Conditions)
        print(prompt_str)
        sys.exit()

    def ask_for_conclusion(self, case_query: CaseQuery) -> Optional[CallableExpression]:
        prompt_str = self.get_prompt_for_ai(case_query, PromptFor.Conclusion)
        output_type_source = self.get_output_type_class_source(case_query)
        prompt_str = f"\n\n\nOutput type(s) class source:\n{output_type_source}\n\n" + prompt_str
        print(prompt_str)
        sys.exit()

    def get_output_type_class_source(self, case_query: CaseQuery) -> str:
        """
        Get the output type class source for the AI expert.

        :param case_query: The case query containing the case to classify.
        :return: The output type class source.
        """
        output_types = case_query.core_attribute_type

        def get_class_source(cls):
            cls_source_file = get_class_file_path(cls)
            found_class_source = extract_function_or_class_file(cls_source_file, function_names=[cls.__name__],
                                                                is_class=True,
                                                                as_list=True)[0]
            class_signature = found_class_source.split('\n')[0]
            if '(' in class_signature:
                parent_class_names = list(map(lambda x: x.strip(),
                                              class_signature.split('(')[1].split(')')[0].split(',')))
                parent_classes = [importlib.import_module(cls.__module__).__dict__.get(cls_name.strip())
                                  for cls_name in parent_class_names]
            else:
                parent_classes = []
            if is_dataclass(cls):
                found_class_source = f"@dataclass\n{found_class_source}"
            return '\n'.join([get_class_source(pcls) for pcls in parent_classes] + [found_class_source])

        found_class_sources = []
        for output_type in output_types:
            found_class_sources.append(get_class_source(output_type))
        found_class_sources = '\n\n\n'.join(found_class_sources)
        return found_class_sources

    def get_prompt_for_ai(self, case_query: CaseQuery, prompt_for: PromptFor) -> str:
        """
        Get the prompt for the AI expert.

        :param case_query: The case query containing the case to classify.
        :param prompt_for: The type of prompt to get.
        :return: The prompt for the AI expert.
        """
        # data_to_show = show_current_and_corner_cases(case_query.case)
        data_to_show = f"\nCase ({case_query.case_name}):\n {case_query.case.__dict__}"
        template_file_creator = TemplateFileCreator(case_query, prompt_for=prompt_for)
        boilerplate_code = template_file_creator.build_boilerplate_code()
        initial_prompt_str = data_to_show + "\n\n" + boilerplate_code + "\n\n"
        return self.user_prompt.build_prompt_str_for_ai(case_query, prompt_for=prompt_for,
                                                        initial_prompt_str=initial_prompt_str)


class Human(Expert):
    """
    The Human Expert class, an expert that asks the human to provide differentiating features and conclusions.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Human expert.
        """
        super().__init__(**kwargs)
        self.user_prompt = UserPrompt()

    def ask_for_conditions(self, case_query: CaseQuery,
                           last_evaluated_rule: Optional[Rule] = None) \
            -> CallableExpression:
        data_to_show = None
        if (not self.use_loaded_answers or len(self.all_expert_answers) == 0) and self.user_prompt.viewer is None:
            data_to_show = show_current_and_corner_cases(case_query.case,
                                                         {case_query.attribute_name: case_query.target_value},
                                                         last_evaluated_rule=last_evaluated_rule)
        return self._get_conditions(case_query, data_to_show)

    def _get_conditions(self, case_query: CaseQuery, data_to_show: Optional[str] = None) \
            -> CallableExpression:
        """
        Ask the expert to provide the differentiating features between two cases or unique features for a case
        that doesn't have a corner case to compare to.

        :param case_query: The case query containing the case to classify.
        :return: The differentiating features as new rule conditions.
        """
        user_input = None
        if self.use_loaded_answers and len(self.all_expert_answers) == 0 and self.append:
            self.use_loaded_answers = False
        if self.use_loaded_answers:
            try:
                loaded_scope, user_input = self.all_expert_answers.pop(0)
            except IndexError:
                self.use_loaded_answers = False
        if user_input is not None:
            case_query.scope.update(loaded_scope)
            condition = CallableExpression(user_input, bool, scope=case_query.scope)
            if self.answers_save_path is not None and not any(loaded_scope):
                self.convert_json_answer_to_python_answer(case_query, user_input, condition, PromptFor.Conditions)
        else:
            user_input, condition = self.user_prompt.prompt_user_for_expression(case_query, PromptFor.Conditions,
                                                                                prompt_str=data_to_show)
        if user_input in ['exit', 'quit']:
            sys.exit()
        if not self.use_loaded_answers:
            self.all_expert_answers.append((condition.scope, user_input))
            if self.answers_save_path is not None:
                self.save_answers()
        case_query.conditions = condition
        return condition

    def convert_json_answer_to_python_answer(self, case_query: CaseQuery, user_input: str,
                                             callable_expression: CallableExpression,
                                             prompt_for: PromptFor):
        tfc = TemplateFileCreator(case_query, prompt_for=prompt_for)
        code = tfc.build_boilerplate_code()
        if user_input.startswith('def'):
            user_input = '\n'.join(user_input.split('\n')[1:])
            user_input = indent(dedent(user_input), " " * 4).strip()
            code = code.replace('pass', user_input)
        else:
            code = code.replace('pass', f"return {user_input}")
        self.save_answers(expert_answers=[({}, code)])

    def ask_for_conclusion(self, case_query: CaseQuery) -> Optional[CallableExpression]:
        """
        Ask the expert to provide a conclusion for the case.

        :param case_query: The case query containing the case to find a conclusion for.
        :return: The conclusion for the case as a callable expression.
        """
        expression: Optional[CallableExpression] = None
        expert_input: Optional[str] = None
        if self.use_loaded_answers and len(self.all_expert_answers) == 0 and self.append:
            self.use_loaded_answers = False
        if self.use_loaded_answers:
            try:
                loaded_scope, expert_input = self.all_expert_answers.pop(0)
                if expert_input is not None:
                    case_query.scope.update(loaded_scope)
                    expression = CallableExpression(expert_input, case_query.attribute_type,
                                                    scope=case_query.scope,
                                                    mutually_exclusive=case_query.mutually_exclusive)
                    if self.answers_save_path is not None and not any(loaded_scope):
                        self.convert_json_answer_to_python_answer(case_query, expert_input, expression,
                                                                  PromptFor.Conclusion)
            except IndexError:
                self.use_loaded_answers = False
        if not self.use_loaded_answers:
            data_to_show = None
            if self.user_prompt.viewer is None:
                data_to_show = show_current_and_corner_cases(case_query.case)
            expert_input, expression = self.user_prompt.prompt_user_for_expression(case_query, PromptFor.Conclusion,
                                                                                   prompt_str=data_to_show)
            if expert_input is None:
                self.all_expert_answers.append(({}, None))
            elif expert_input not in ['exit', 'quit']:
                self.all_expert_answers.append((expression.scope, expert_input))
            if self.answers_save_path is not None and expert_input not in ['exit', 'quit']:
                self.save_answers()
        if expert_input in ['exit', 'quit']:
            sys.exit()
        case_query.target = expression
        return expression

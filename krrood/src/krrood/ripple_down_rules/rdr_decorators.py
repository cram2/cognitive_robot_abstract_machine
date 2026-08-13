"""
This file contains decorators for the RDR (Ripple Down Rules) framework. Where each type of RDR has a decorator
that can be used with any python function such that this function can benefit from the incremental knowledge acquisition
of the RDRs.
"""
import inspect
import os.path
from dataclasses import dataclass, field
from functools import wraps

from typing_extensions import Callable, Optional, Type, Tuple, Dict, Any, List

from krrood.ripple_down_rules.datastructures.dataclasses import CaseFactoryMetaData
from krrood.ripple_down_rules.experts import Expert, Human
from krrood.ripple_down_rules.rdr import GeneralRDR
from krrood.ripple_down_rules.utils import get_origin_type_of_function_output, get_arg_type_of_function_output, fill_in_missing_kwargs

try:
    from .user_interface.gui import RDRCaseViewer
except ImportError:
    RDRCaseViewer = None
from krrood.ripple_down_rules.utils import get_func_rdr_model_name, make_set, \
    make_list
from krrood.ripple_down_rules.helpers import create_case_from_method, create_case_query_from_method


@dataclass(unsafe_hash=True)
class RDRDecorator:
    models_dir: str
    """
    The directory to save the RDR models in.
    """
    output_type: Tuple[Type, ...]
    """
    The type(s) of the output produced by the RDR model (The type(s) of the queried attribute).
    """
    mutual_exclusive: bool
    """
    Whether the queried attribute is mutually exclusive (i.e. allows for only one possible value) or not.
    """
    fit: bool = field(default=True)
    """
    Whether to run in fitting mode and prompt the expert or just classify using existing rules.
    """
    expert: Optional[Expert] = field(default=None)
    """
    The expert instance, this is used by the rdr to prompt for answers.
    """
    update_existing_rules: bool = field(default=True)
    """
    When in fitting mode, whether to ask for answers for existing rules as well or not.
    """
    package_name: Optional[str] = field(default=None)
    """
    The name of the user python package where the RDR model will be saved, this is useful for generating relative
    imports in the generated rdr model files.
    """
    use_generated_classifier: bool = field(default=False)
    """
    Whether to use the generated classifier files of the rdr model instead of the RDR instance itself, this is useful
    when you want to debug inside the rules.
    """
    ask_now: Callable[Dict[str, Any], bool] = field(default=lambda _: False)
    """
    A user provided callable function that outputs a boolean indicating when to ask the expert for an answer. The input
    to the `ask_now` function is a dictionary with the original function arguments, while arguments like `self` and
    `cls` are passed as a special key `self_` or `cls_` respectively.
    """
    ask_now_target: Optional[Any] = field(default=None)
    """
    A target for the case that triggers the ask now function.
    """
    fitting_decorator: Optional[Callable] = field(default=lambda f: f)
    """
    A user provided decorator that wraps the `py:meth:ripple_down_rules.rdr.RippleDownRules.fit_case` method which is 
    used when in fitting mode, this is useful when you want special logic pre and post the fitting operation, you can
    for example freeze your system during fitting such that you have a stable state that you can query and use while 
    writing and testing your answer/rule.
    """
    generate_dot_file: bool = field(default=False)
    """
    Whether to generate a dynamic dot file representing the state of the rule tree each time the rdr is queried, showing
    which rules fired and which rules didn't get evaluated, ...etc.
    """
    regenerate_model: bool = field(default=False)
    """
    Whether to regenerate the rdr model from scratch or not, this is useful when rdr package has been updated.
    """
    model_name: Optional[str] = field(default=None)
    """
    The name of the rdr model, this gets auto generated from the function signature and the class/file it is contained
    in.
    """
    rdr: GeneralRDR = field(init=False, default=None)
    """
    The ripple down rules instance of the decorator class.
    """
    parsed_output_type: List[Type] = field(init=False, default_factory=list)
    """
    The output of a post processing done on the output types, for example converting typing module types 
    (i.e. type hints) to python types.
    """
    origin_type: Optional[Type] = field(init=False, default=None)
    """
    The origin of the type hint of the attribute, useful in the case of not mutually exclusive attributes to map the 
    result to the specified container type (e.g. a list instead of a set which is the default container type for rdr
    output).
    """
    output_name: str = field(init=False, default='output_')
    """
    This is used to refer to the output value of the decorated function, this is used as part of the case as input to 
    the rdr model, but is never used in the rule logic to prevent cycles from happening. The correct way to use the 
    output of an rdr is through refinement rules which happens automatically by the rdr prompting for refinements.
    """
    _not_none_output_found: bool = field(init=False, default=False)
    """
    This is a flag that indicates that a not None output for the rdr has been inferred, this is used to update the 
    generated dot file if it is set to `True`.
    """
    case_factory_metadata: CaseFactoryMetaData = field(init=False, default_factory=CaseFactoryMetaData)
    """
    Metadata that contains the case factory method, and the scenario that is being run during the case query.
    """
    generated_classifier: Optional[Callable] = field(init=False, default=None)
    """
    The generated classifier file of the rdr model, this is used when `use_generated_classifier` is set to `True`.
    """

    def decorator(self, func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[Any]:

            kwargs = fill_in_missing_kwargs(func, **kwargs)

            if self.origin_type is None and not self.mutual_exclusive:
                self.origin_type = get_origin_type_of_function_output(func)

            if len(self.parsed_output_type) == 0:
                self.parsed_output_type = self.parse_output_type(func, self.output_type, *args)
                if self.origin_type:
                    self.parsed_output_type.append(self.origin_type)

            if self.model_name is None:
                self.initialize_rdr_model_name_and_load(func)

            func_output = {self.output_name: func(*args, **kwargs)}

            case, case_dict = create_case_from_method(func, func_output, *args, **kwargs)

            @self.fitting_decorator
            def fit():
                if self.expert is None:
                    self.expert = Human(answers_save_path=self.models_dir + f'/{self.model_name}/expert_answers')
                case_query = create_case_query_from_method(
                    func, func_output,
                    self.parsed_output_type,
                    self.mutual_exclusive,
                    args, kwargs,
                    case=case, case_dict=case_dict,
                    scenario=self.case_factory_metadata.scenario,
                    this_case_target_value=self.case_factory_metadata.this_case_target_value)
                output = self.rdr.fit_case(case_query, expert=self.expert,
                                           update_existing_rules=self.update_existing_rules,
                                           ask_now=self.ask_now, ask_now_target=self.ask_now_target)
                return output

            if self.fit and not self.use_generated_classifier:
                output = fit()
            else:
                if self.use_generated_classifier:
                    if self.generated_classifier is None:
                        model_path = os.path.join(self.models_dir, self.model_name)
                        self.generated_classifier = self.rdr.get_rdr_classifier_from_python_file(model_path)
                    output = self.generated_classifier(case)
                else:
                    output = self.rdr.classify(case)
                    if self.generate_dot_file:
                        eval_rule_tree = self.rdr.get_evaluated_rule_tree()
                        if not self._not_none_output_found or (eval_rule_tree and len(eval_rule_tree) > 1):
                            self.rdr.render_evaluated_rule_tree(self.models_dir + f'/{self.model_name}',
                                                                show_full_tree=True)
                        if eval_rule_tree and len(eval_rule_tree) > 1:
                            self._not_none_output_found = True

            if self.output_name in output:
                if self.origin_type == list:
                    return make_list(output[self.output_name])
                return output[self.output_name]
            else:
                return func_output[self.output_name]

        wrapper._rdr_decorator_instance = self

        return wrapper

    def initialize_rdr_model_name_and_load(self, func: Callable) -> None:
        self.model_name = get_func_rdr_model_name(func, include_file_name=True)
        self.load()
        if self.regenerate_model:
            self.rdr.save(self.models_dir, self.model_name)

    def parse_output_type(self, func: Callable, output_type: Any, *args) -> List[Type]:
        parsed_output_type = []
        for ot in make_set(output_type):
            ot = get_arg_type_of_function_output(func, ot, *args, package_name=self.package_name)
            parsed_output_type.append(ot)
        return parsed_output_type

    def load(self):
        """
        Load the RDR model from the specified directory, otherwise create a new one.
        """
        self.rdr = GeneralRDR(save_dir=self.models_dir, model_name=self.model_name)


def fit_rdr_func(scenario: Callable, rdr_decorated_func: Callable, *func_args, **func_kwargs) -> None:
    rdr_decorated_func._rdr_decorator_instance.case_factory_metadata = CaseFactoryMetaData(scenario=scenario)
    rdr_decorated_func(*func_args, **func_kwargs)

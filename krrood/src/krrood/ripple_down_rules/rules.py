from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import NoneType
from uuid import uuid4

from anytree import Node
from typing_extensions import List, Optional, Self, Union, Dict, Any, Tuple, Type, Set

from krrood.ripple_down_rules.datastructures.callable_expression import CallableExpression
from krrood.ripple_down_rules.datastructures.case import Case
from krrood.ripple_down_rules.datastructures.dataclasses import CaseFactoryMetaData, CaseQuery
from krrood.ripple_down_rules.datastructures.enums import RDREdge, Stop
from krrood.ripple_down_rules.helpers import get_an_updated_case_copy
from krrood.ripple_down_rules.utils import SubclassJSONSerializer, conclusion_to_json, get_full_class_name, get_type_from_string


@dataclass
class Rule(SubclassJSONSerializer, ABC):
    conditions: Optional[CallableExpression] = field(default=None)
    """
    The conditions of the rule, which is a callable expression that takes a case and returns a boolean.
    """
    conclusion: Optional[CallableExpression] = field(default=None)
    """
    The conclusion of the rule, which is a callable expression that takes a case and returns a value.
    """
    _parent: Optional[Rule] = field(default=None)
    """
    The parent rule of this rule in the ripple down rules tree.
    """
    corner_case: Optional[Any] = field(default=None)
    """
    The corner case for which this rule was created.
    """
    _weight: Optional[RDREdge] = field(default=None)
    """
    The weight of the rule, which is the type of edge connecting the rule to its parent.
    """
    conclusion_name: Optional[str] = field(default=None)
    """
    The name of the conclusion of the rule, which is used to identify the conclusion
    """
    uid: str = field(default_factory=lambda: str(uuid4().int))
    """
    A unique id for the rule using uuid4
    """
    corner_case_metadata: Optional[CaseFactoryMetaData] = field(default=None)
    """
    Metadata about the corner case, such as the factory that created it or the scenario it is based on.
    """
    json_serialization: Optional[Dict[str, Any]] = field(init=False, default=None)
    """
    The JSON serialization of the rule, which is used to serialize the rule to JSON.
    """
    _name: Optional[str] = field(init=False, default=None)
    """
    The name of the rule, which is the names of the conditions and the conclusion
    """
    evaluated: bool = field(init=False, default=False)
    """
    Whether the rule has been evaluated or not (i.e. whether the rule has been reached during evaluation of the ripple 
    down rules tree and the conditions have been checked).
    """
    last_conclusion: Optional[Any] = field(init=False, default=None)
    """
    The last conclusion of the rule, which is the conclusion of the rule when it was last evaluated.
    """
    contributed: bool = field(init=False, default=False)
    """
    Whether the rule has contributed by a value, meaning that it has fired and the conclusion has been added to the case.
    """
    contributed_to_case_query: bool = field(init=False, default=False)
    """
    Whether the rule has contributed to the case query, meaning that it has fired and the conclusion is relevant to the
    case query.
    """
    fired: Optional[bool] = field(init=False, default=None)
    """
    Whether the rule has fired or not.
    """
    mutually_exclusive: bool = field(init=False, default=True)
    """
    Whether the rule is mutually exclusive with other rules.
    """
    node: Optional[Node] = field(init=False, default=None)

    def __post_init__(self):
        self.node = Node(self.name, parent=self.parent.node if self.parent else None)
        self.node.weight = self.weight.value if self.weight else None
        self.node._rdr_rule = self

    @property
    def descendants(self) -> List[Rule]:
        """
        :return: the descendants of this rule, which are the rules that are children of this rule in the ripple down
         rules tree.
        """
        return [child._rdr_rule for child in self.node.descendants]

    @property
    def children(self) -> List[Rule]:
        """
        :return: the children of this rule, which are the rules that are direct children of this rule in the ripple down
         rules tree.
        """
        return [child._rdr_rule for child in self.node.children]

    @property
    def parent(self):
        """
        :return: The parent rule of this rule.
        """
        return self._parent

    @parent.setter
    def parent(self, new_parent: Optional[Rule]):
        """
        Set the parent rule of this rule.
        :param new_parent: The new parent rule to set.
        """
        self._parent = new_parent
        if self.node:
            self.node.parent = new_parent.node

    @property
    def weight(self) -> RDREdge:
        return self._weight

    @weight.setter
    def weight(self, new_weight: RDREdge):
        """
        Set the weight of the rule, which is the type of edge connecting the rule to its parent.
        :param new_weight: The new weight to set.
        """
        self._weight = new_weight
        if self.node:
            self.node.weight = new_weight.value

    def get_an_updated_case_copy(self, case: Case) -> Case:
        """
        :param case: The case to copy and update.
        :return: A copy of the case updated with this rule conclusion.
        """
        return get_an_updated_case_copy(case, self.conclusion, self.conclusion_name, self.conclusion.conclusion_type,
                                        self.mutually_exclusive)

    def reset(self):
        self.evaluated = False
        self.fired = False
        self.contributed = False
        self.contributed_to_case_query = False
        self.last_conclusion = None

    @property
    def color(self) -> str:
        if self.evaluated:
            if self.contributed_to_case_query:
                return "green"
            elif self.contributed:
                return "yellow"
            elif self.fired:
                return "orange"
            else:
                return "red"
        else:
            return "white"

    @classmethod
    def from_case_query(cls, case_query: CaseQuery, parent: Optional[Rule] = None) -> Rule:
        """
        Create a SingleClassRule from a CaseQuery.

        :param case_query: The CaseQuery to create the rule from.
        :param parent: The parent rule of this rule.
        :return: A SingleClassRule instance.
        """
        corner_case_metadata = CaseFactoryMetaData.from_case_query(case_query)
        return cls(conditions=case_query.conditions, conclusion=case_query.target,
                   corner_case=case_query.case, _parent=parent,
                   corner_case_metadata=corner_case_metadata,
                   conclusion_name=case_query.attribute_name)

    def _post_detach(self, parent):
        """
        Called after this node is detached from the tree, useful when drawing the tree.

        :param parent: The parent node from which this node was detached.
        """
        self.weight = None

    def __call__(self, x: Case) -> Self:
        return self.evaluate(x)

    def evaluate(self, x: Case) -> Rule:
        """
        Check if the rule or its refinement or its alternative match the case,
        by checking if the conditions are met, then return the rule that matches the case.

        :param x: The case to evaluate the rule on.
        :return: The rule that fired or the last evaluated rule if no rule fired.
        """
        self.evaluated = True
        if not self.conditions:
            raise ValueError("Rule has no conditions")
        self.fired = self.conditions(x)
        return self.evaluate_next_rule(x)

    @abstractmethod
    def evaluate_next_rule(self, x: Case):
        """
        Evaluate the next rule after this rule is evaluated.
        """
        pass

    def write_corner_case_as_source_code(self, cases_file: str, package_name: Optional[str] = None) -> None:
        """
        Write the source code representation of the corner case of the rule to a file.

        :param cases_file: The file to write the corner case to.
        :param package_name: The package name to use for relative imports.
        """
        if self.corner_case_metadata is None:
            return
        with open(cases_file, 'a') as f:
            f.write(f"corner_case_{self.uid} = {self.corner_case_metadata}" + "\n\n\n")

    def get_corner_case_types_to_import(self) -> Set[Type]:
        """
        Get the types that need to be imported for the corner case of the rule.
        """
        if self.corner_case_metadata is None:
            return
        types_to_import = set()
        if self.corner_case_metadata.factory_method is not None:
            types_to_import.add(self.corner_case_metadata.factory_method)
        if self.corner_case_metadata.scenario is not None:
            types_to_import.add(self.corner_case_metadata.scenario)
        if self.corner_case_metadata.case_conf is not None:
            types_to_import.add(self.corner_case_metadata.case_conf)
        return types_to_import

    def write_conclusion_as_source_code(self, parent_indent: str = "", defs_file: Optional[str] = None) -> str:
        """
        Get the source code representation of the conclusion of the rule.

        :param parent_indent: The indentation of the parent rule.
        :param defs_file: The file to write the conclusion to if it is a definition.
        :return: The source code representation of the conclusion of the rule.
        """
        if self.conclusion.user_input is not None:
            conclusion = self.conclusion.user_input
        else:
            conclusion = self.conclusion.conclusion
        conclusion_func, conclusion_func_call = self.get_conclusion_as_source_code(conclusion,
                                                                                   parent_indent=parent_indent)
        if conclusion_func is not None:
            with open(defs_file, 'a') as f:
                f.write(conclusion_func.strip() + "\n\n\n")
        return conclusion_func_call

    @property
    def generated_conclusion_function_name(self) -> str:
        return f"conclusion_{self.uid}"

    @property
    def generated_conditions_function_name(self) -> str:
        return f"conditions_{self.uid}"

    @property
    def generated_corner_case_object_name(self) -> str:
        return f"corner_case_{self.uid}"

    def get_conclusion_as_source_code(self, conclusion: Any, parent_indent: str = "") -> Tuple[Optional[str], str]:
        """
        Convert the conclusion of a rule to source code.

        :param conclusion: The conclusion to convert to source code.
        :param parent_indent: The indentation of the parent rule.
        :return: The source code of the conclusion as a tuple of strings, one for the function and one for the call.
        """
        if "def " in conclusion:
            # This means the conclusion is a definition that should be written and then called
            conclusion_lines = conclusion.split('\n')
            # use regex to replace the function name
            new_function_name = f"def {self.generated_conclusion_function_name}"
            conclusion_lines[0] = re.sub(r"def (\w+)", new_function_name, conclusion_lines[0])
            # add type hint
            if not self.conclusion.mutually_exclusive:
                type_names = [t.__name__ for t in self.conclusion.conclusion_type if t not in [list, set]]
                if len(type_names) == 1:
                    hint = f"List[{type_names[0]}]"
                else:
                    hint = f"List[Union[{', '.join(type_names)}]]"
            else:
                if NoneType in self.conclusion.conclusion_type:
                    type_names = [t.__name__ for t in self.conclusion.conclusion_type if t is not NoneType]
                    hint = f"Optional[{', '.join(type_names)}]"
                elif len(self.conclusion.conclusion_type) == 1:
                    hint = self.conclusion.conclusion_type[0].__name__
                else:
                    type_names = [t.__name__ for t in self.conclusion.conclusion_type]
                    hint = f"Union[{', '.join(type_names)}]"
            conclusion_lines[0] = conclusion_lines[0].replace("):", f") -> {hint}:")
            func_call = f"{parent_indent}    return {new_function_name.replace('def ', '')}(case)\n"
            return "\n".join(conclusion_lines).strip(' '), func_call
        else:
            raise ValueError(f"Conclusion format is not valid, it should contain a function definition."
                             f" Instead got:\n{conclusion}\n")

    def write_condition_as_source_code(self, parent_indent: str = "", defs_file: Optional[str] = None) -> str:
        """
        Get the source code representation of the conditions of the rule.

        :param parent_indent: The indentation of the parent rule.
        :param defs_file: The file to write the conditions to if they are a definition.
        """
        if_clause = self._if_statement_source_code_clause()
        if "def " in self.conditions.user_input:
            if defs_file is None:
                raise ValueError("Cannot write conditions to source code as definitions python file was not given.")
            # This means the conditions are a definition that should be written and then called
            conditions_lines = self.conditions.user_input.split('\n')
            # use regex to replace the function name
            new_function_name = f"def {self.generated_conditions_function_name}"
            conditions_lines[0] = re.sub(r"def (\w+)", new_function_name, conditions_lines[0])
            # add type hint
            conditions_lines[0] = conditions_lines[0].replace('):', ') -> bool:')
            def_code = "\n".join(conditions_lines)
            with open(defs_file, 'a') as f:
                f.write(def_code.strip() + "\n\n\n")
            return f"\n{parent_indent}{if_clause} {new_function_name.replace('def ', '')}(case):\n"
        else:
            raise ValueError(f"Conditions format is not valid, it should contain a function definition"
                             f" Instead got:\n{self.conditions.user_input}\n")

    @abstractmethod
    def _if_statement_source_code_clause(self) -> str:
        pass

    def _to_json(self) -> Dict[str, Any]:
        json_serialization = {"_type": get_full_class_name(type(self)),
                              "conditions": self.conditions.to_json(),
                              "conclusion": conclusion_to_json(self.conclusion),
                              "parent": self.parent.json_serialization if self.parent else None,
                              "conclusion_name": self.conclusion_name,
                              "weight": self.weight.value,
                              "uid": self.uid}
        return json_serialization

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Rule:
        loaded_rule = cls(conditions=CallableExpression.from_json(data["conditions"]),
                          conclusion=CallableExpression.from_json(data["conclusion"]),
                          _parent=cls.from_json(data["parent"]),
                          conclusion_name=data["conclusion_name"],
                          _weight=RDREdge.from_value(data["weight"]),
                          uid=data["uid"])
        return loaded_rule

    @property
    def name(self):
        """
        Get the name of the rule, which is the conditions and the conclusion.
        """
        return self._name if self._name is not None else self.__str__()

    @name.setter
    def name(self, new_name: str):
        """
        Set the name of the rule.
        """
        self._name = new_name
        self.node.name = new_name

    @property
    def semantic_condition_name(self) -> Optional[str]:
        """
        Get the name of the conditions of the rule, which is the user input of the conditions.
        """
        if isinstance(self.conditions, CallableExpression):
            return self.expression_name(self.conditions)
        return None

    @property
    def semantic_conclusion_name(self) -> Optional[str]:
        """
        Get the name of the conclusion of the rule, which is the user input of the conclusion.
        """
        if isinstance(self.conclusion, CallableExpression):
            return self.expression_name(self.conclusion)
        return None

    @staticmethod
    def expression_name(expression: CallableExpression) -> str:
        """
        Get the name of the expression, which is the user input of the expression if it exists,
        otherwise it is the conclusion or conditions of the rule.
        """
        if expression.user_defined_name is not None and expression.user_defined_name != expression.encapsulating_function_name:
            return expression.user_defined_name.strip()
        func_name = expression.user_input.split('(')[0].replace('def ',
                                                                '').strip() if "def " in expression.user_input else None
        if func_name is not None and func_name != expression.encapsulating_function_name:
            return func_name
        elif expression.user_input:
            return expression.user_input.strip()
        else:
            return str(expression)

    def __str__(self, sep="\n"):
        """
        Get the string representation of the rule, which is the conditions and the conclusion.
        """
        return f"{self.semantic_condition_name}{sep}=> {self.semantic_conclusion_name}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return other.uid == self.uid

    def __hash__(self):
        return hash(self.uid)


@dataclass
class HasAlternativeRule:
    """
    A mixin class for rules that have an alternative rule.
    """
    _alternative: Optional[Rule] = field(init=False, default=None)
    """
    The alternative rule of the rule, which is evaluated when the rule doesn't fire.
    """
    furthest_alternative: Optional[List[Rule]] = field(init=False, default=None)
    """
    The furthest alternative rule of the rule, which is the last alternative rule in the chain of alternative rules.
    """
    all_alternatives: Optional[List[Rule]] = field(init=False, default=None)
    """
    All alternative rules of the rule, which is all the alternative rules in the chain of alternative rules.
    """

    @property
    def alternative(self) -> Optional[Rule]:
        return self._alternative

    @alternative.setter
    def alternative(self, new_rule: Rule):
        """
        Set the alternative rule of the rule. It is important that no rules should be retracted or changed,
        only new rules should be added.
        """
        if new_rule is None:
            return
        if self.furthest_alternative:
            self.furthest_alternative[-1].alternative = new_rule
        else:
            new_rule.parent = self
            new_rule.weight = RDREdge.Alternative if not new_rule.weight else new_rule.weight
            self._alternative = new_rule
        self.furthest_alternative = [new_rule]


@dataclass
class HasRefinementRule:
    _refinement: Optional[HasAlternativeRule] = field(init=False, default=None)
    """
    The refinement rule of the rule, which is evaluated when the rule fires.
    """

    @property
    def refinement(self) -> Optional[Rule]:
        return self._refinement

    @refinement.setter
    def refinement(self, new_rule: Rule):
        """
        Set the refinement rule of the rule. It is important that no rules should be retracted or changed,
        only new rules should be added.
        """
        if new_rule is None:
            return
        if self.refinement:
            self.refinement.alternative = new_rule
        else:
            new_rule.parent = self
            new_rule.weight = RDREdge.Refinement if not isinstance(new_rule,
                                                                   MultiClassFilterRule) else new_rule.weight
            self._refinement = new_rule


@dataclass(eq=False)
class SingleClassRule(Rule, HasAlternativeRule, HasRefinementRule):
    """
    A rule in the SingleClassRDR classifier, it can have a refinement or an alternative rule or both.
    """

    mutually_exclusive: bool = field(init=False, default=True)

    def evaluate_next_rule(self, x: Case) -> SingleClassRule:
        if self.fired:
            returned_rule = self.refinement(x) if self.refinement else self
        else:
            returned_rule = self.alternative(x) if self.alternative else self
        return returned_rule if returned_rule.fired else self

    def fit_rule(self, case_query: CaseQuery):
        corner_case_metadata = CaseFactoryMetaData.from_case_query(case_query)
        new_rule = SingleClassRule(case_query.conditions, case_query.target,
                                   corner_case=case_query.case, _parent=self,
                                   corner_case_metadata=corner_case_metadata,
                                   )
        if self.fired:
            self.refinement = new_rule
        else:
            self.alternative = new_rule

    def _to_json(self) -> Dict[str, Any]:
        self.json_serialization = {**super(SingleClassRule, self)._to_json(),
                                   "refinement": self.refinement.to_json() if self.refinement is not None else None,
                                   "alternative": self.alternative.to_json() if self.alternative is not None else None}
        return self.json_serialization

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> SingleClassRule:
        loaded_rule = super(SingleClassRule, cls)._from_json(data)
        loaded_rule.refinement = SingleClassRule.from_json(data["refinement"])
        loaded_rule.alternative = SingleClassRule.from_json(data["alternative"])
        return loaded_rule

    def _if_statement_source_code_clause(self) -> str:
        return "elif" if self.weight == RDREdge.Alternative else "if"


@dataclass(eq=False)
class MultiClassRefinementRule(Rule, HasAlternativeRule, ABC):
    """
    A rule in the MultiClassRDR classifier, it can have an alternative rule and a top rule.
    """
    top_rule: Optional[MultiClassTopRule] = field(init=False, default=None)
    """
    The top rule of the rule, which is the nearest ancestor that fired and this rule is a refinement of.
    """
    mutually_exclusive: bool = field(init=False, default=False)

    def _to_json(self) -> Dict[str, Any]:
        self.json_serialization = {**Rule._to_json(self),
                                   "alternative": self.alternative.to_json() if self.alternative is not None else None}
        return self.json_serialization

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> MultiClassRefinementRule:
        loaded_rule = super(MultiClassRefinementRule, cls)._from_json(data)
        # The following is done to prevent re-initialization of the top rule,
        # so the top rule that is already initialized is passed in the data instead of its json serialization.
        loaded_rule.top_rule = data['top_rule']
        if data['alternative'] is not None:
            data['alternative']['top_rule'] = data['top_rule']
            loaded_rule.alternative = SubclassJSONSerializer.from_json(data["alternative"])
        return loaded_rule

    def _if_statement_source_code_clause(self) -> str:
        return "elif" if self.weight == RDREdge.Alternative else "if"


@dataclass(eq=False)
class MultiClassStopRule(MultiClassRefinementRule):
    """
    A rule in the MultiClassRDR classifier, it can have an alternative rule and a top rule,
    the conclusion of the rule is a Stop category meant to stop the parent conclusion from being made.
    """
    conclusion: CallableExpression = field(default_factory=lambda: CallableExpression(conclusion_type=(Stop,),
                                                                                      conclusion=Stop.stop))
    """
    The conclusion of the rule, which is a CallableExpression that returns a Stop category.
    """

    def evaluate_next_rule(self, x: Case) -> Optional[Union[MultiClassRefinementRule, MultiClassTopRule]]:
        if self.fired:
            self.top_rule.fired = False
            return self.top_rule.alternative
        elif self.alternative:
            return self.alternative(x)
        else:
            return self.top_rule.alternative

    def get_conclusion_as_source_code(self, conclusion: Any, parent_indent: str = "") -> Tuple[None, str]:
        return None, f"{parent_indent}{' ' * 4}pass\n"


@dataclass(eq=False)
class MultiClassFilterRule(MultiClassRefinementRule, HasRefinementRule):
    """
    A rule in the MultiClassRDR classifier, it can have an alternative rule and a top rule,
    the conclusion of the rule is a Filter category meant to filter the parent conclusion.
    """
    weight: RDREdge = field(init=False, default_factory=lambda: RDREdge.Filter)

    def evaluate_next_rule(self, x: Case) -> Optional[Union[MultiClassRefinementRule, MultiClassTopRule]]:
        if self.fired:
            if self.refinement:
                case_cp = x
                if isinstance(self.refinement, MultiClassFilterRule):
                    case_cp = self.get_an_updated_case_copy(case_cp)
                return self.refinement(case_cp)
            else:
                return self.top_rule.alternative
        elif self.alternative:
            return self.alternative(x)
        else:
            return self.top_rule.alternative

    def get_conclusion_as_source_code(self, conclusion: Any, parent_indent: str = "") -> Tuple[None, str]:
        func, func_call = super().get_conclusion_as_source_code(str(conclusion), parent_indent=parent_indent)
        conclusion_str = func_call.replace("return ", "").strip()
        conclusion_str = conclusion_str.replace("(case)", "(case_cp)")

        parent_to_filter = self.get_parent_to_filter()
        statement = (
            f"\n{parent_indent}    case_cp = get_an_updated_case_copy(case, {parent_to_filter.generated_conclusion_function_name},"
            f" attribute_name, conclusion_type, mutually_exclusive)")
        statement += f"\n{parent_indent}    conclusions.update(make_set({conclusion_str}))\n"
        return func, statement

    def get_parent_to_filter(self, parent: Union[None, MultiClassRefinementRule, MultiClassTopRule] = None) \
            -> Union[MultiClassFilterRule, MultiClassTopRule]:
        parent = self.parent if parent is None else parent
        if isinstance(parent, (MultiClassFilterRule, MultiClassTopRule)) and parent.fired:
            return parent
        else:
            return parent.parent

    def _to_json(self) -> Dict[str, Any]:
        self.json_serialization = super(MultiClassFilterRule, self)._to_json()
        self.json_serialization['refinement'] = self.refinement.to_json() if self.refinement is not None else None
        return self.json_serialization

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> MultiClassFilterRule:
        loaded_rule = super(MultiClassFilterRule, cls)._from_json(data)
        if data['refinement'] is not None:
            data['refinement']['top_rule'] = data['top_rule']
        loaded_rule.refinement = cls.from_json(data["refinement"]) if data["refinement"] is not None else None
        return loaded_rule


@dataclass(eq=False)
class MultiClassTopRule(Rule, HasRefinementRule, HasAlternativeRule):
    """
    A rule in the MultiClassRDR classifier, it can have a refinement and a next rule.
    """
    mutually_exclusive: bool = field(init=False, default=False)
    weight: RDREdge = field(init=False, default_factory=lambda: RDREdge.Next)

    def evaluate_next_rule(self, x: Case) -> Optional[Union[MultiClassStopRule, MultiClassTopRule]]:
        if self.fired and self.refinement:
            case_cp = x
            if isinstance(self.refinement, MultiClassFilterRule):
                case_cp = self.get_an_updated_case_copy(case_cp)
            return self.refinement(case_cp)
        elif self.alternative:  # Here alternative refers to next rule in MultiClassRDR
            return self.alternative
        return None

    def fit_rule(self, case_query: CaseQuery, refinement_type: Optional[Type[MultiClassRefinementRule]] = None):
        if self.fired and case_query.target != self.conclusion:
            if refinement_type in [None, MultiClassStopRule]:
                new_rule = MultiClassStopRule(case_query.conditions, corner_case=case_query.case,
                                              _parent=self)
            elif refinement_type is MultiClassFilterRule:
                new_rule = MultiClassFilterRule.from_case_query(case_query, parent=self)
            else:
                raise ValueError(f"Unknown refinement type {refinement_type}")
            new_rule.top_rule = self
            self.refinement = new_rule
        elif not self.fired:
            self.alternative = MultiClassTopRule.from_case_query(case_query, parent=self)

    def _to_json(self) -> Dict[str, Any]:
        self.json_serialization = {**super()._to_json(),
                                   "refinement": self.refinement.to_json() if self.refinement is not None else None,
                                   "alternative": self.alternative.to_json() if self.alternative is not None else None}
        return self.json_serialization

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> MultiClassTopRule:
        loaded_rule = super(MultiClassTopRule, cls)._from_json(data)
        # The following is done to prevent re-initialization of the top rule,
        # so the top rule that is already initialized is passed in the data instead of its json serialization.
        if data['refinement'] is not None:
            data['refinement']['top_rule'] = loaded_rule
            data_type = get_type_from_string(data["refinement"]["_type"])
            loaded_rule.refinement = data_type.from_json(data["refinement"])
        loaded_rule.alternative = MultiClassTopRule.from_json(data["alternative"])
        return loaded_rule

    def get_conclusion_as_source_code(self, conclusion: Any, parent_indent: str = "") -> Tuple[str, str]:
        func, func_call = super().get_conclusion_as_source_code(str(conclusion), parent_indent=parent_indent)
        conclusion_str = func_call.replace("return ", "").strip()
        indent = parent_indent + " " * 4
        statement = (f"{indent}update_case_and_conclusions_with_rule_output(case, conclusions, {conclusion_str},"
                                     f"attribute_name, conclusion_type, mutually_exclusive)\n")
        # new_conclusions_statement = f"{indent}new_conclusions = rule_conclusion - conclusions\n"
        # update_if_statement = f"{indent}if new_conclusions:\n"
        # conclusions_update_statement = f"{indent}    conclusions.update(new_conclusions)\n"
        # case_update_statement = f"{indent}    update_case_with_conclusion_output(case, new_conclusions, attribute_name, conclusion_type, mutually_exclusive)\n"
        # statement = rule_conclusion_statement + new_conclusions_statement + update_if_statement + conclusions_update_statement + case_update_statement
        return func, statement

    def _if_statement_source_code_clause(self) -> str:
        return "if"

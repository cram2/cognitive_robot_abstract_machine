from dataclasses import dataclass

from typing_extensions import Optional

from krrood.entity_query_language.predicate import Symbol


@dataclass(unsafe_hash=True)
class Department(Symbol):
    name: str


@dataclass
class Employee(Symbol):
    name: str
    department: Department
    salary: float
    starting_salary: Optional[float] = None

    def __post_init__(self):
        if self.starting_salary is None:
            self.starting_salary = self.salary / 2

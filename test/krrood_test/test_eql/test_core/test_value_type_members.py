"""
Tests that a variable's value type is reached through its members: a call receives the
values of the variables passed to it, and a field keeps its meaning when a method of
every variable has the same name.
"""

from __future__ import annotations

import datetime

import pytest

from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    CallWithExpressionArguments,
    CallWithValueArguments,
    ExpressionMemberOrDomainField,
)
from krrood.entity_query_language.factories import an, entity, variable

from ...dataset.rule_tree_domains import Ledger, Payment, Period, TaxYear


# %% calls with variables as arguments


def test_a_method_called_with_an_attribute_of_another_variable_receives_its_value():
    """
    A method called with an attribute of another variable receives that attribute's
    value.
    """
    payments = [Payment(year=2016), Payment(year=2017)]
    payment = variable(Payment, domain=payments)
    tax_year = variable(TaxYear, domain=[TaxYear(number=2017)])

    query = an(entity(payment).where(payment.is_paid_in(year=tax_year.number)))

    assert query.tolist() == [payments[1]]


def test_a_call_with_a_variable_argument_ranges_over_that_variables_values():
    """
    A call with a variable as its argument holds for the values of that variable it is
    true for.
    """
    payment = variable(Payment, domain=[Payment(year=2017)])
    tax_years = [TaxYear(number=2016), TaxYear(number=2017)]
    tax_year = variable(TaxYear, domain=tax_years)

    query = an(entity(tax_year).where(payment.is_paid_in(year=tax_year.number)))

    assert query.tolist() == [tax_years[1]]


@pytest.mark.parametrize(
    "argument_is_a_variable, call_type",
    [(True, CallWithExpressionArguments), (False, CallWithValueArguments)],
)
def test_a_call_is_built_by_whether_an_argument_is_a_variable(
    argument_is_a_variable: bool, call_type: type
):
    """
    A call whose arguments are all plain values reaches one value; one with a variable
    argument may reach several.
    """
    payment = variable(Payment, domain=[])
    tax_year = variable(TaxYear, domain=[])
    argument = tax_year.number if argument_is_a_variable else 2017

    assert type(payment.is_paid_in(year=argument)) is call_type


# %% fields named like methods of every variable


def test_a_field_named_like_a_method_of_every_variable_reads_the_field():
    """
    A field of the value type named like a method of every variable reads the field.
    """
    period = Period(first=datetime.date(2017, 1, 1))
    period_variable = variable(Period, domain=[period])

    query = an(
        entity(period_variable).where(
            period_variable.first >= datetime.date(2016, 1, 1)
        )
    )

    assert query.tolist() == [period]


def test_a_method_of_every_variable_stays_a_method_when_no_field_has_its_name():
    """
    Without a field of its name, a method of every variable stays callable.
    """
    tax_year = TaxYear(number=2017)

    assert variable(TaxYear, domain=[tax_year]).first() is tax_year


def test_every_public_method_of_a_variable_steps_aside_for_a_field_of_its_name():
    """
    Each public method a variable has reads the field of the value type with its name.
    """
    ledger = variable(Ledger, domain=[])

    assert [
        type(member)
        for member in (
            ledger.first,
            ledger.tolist,
            ledger.evaluate,
            ledger.conclusions_of_type,
        )
    ] == [Attribute] * 4


def test_a_property_steps_aside_for_a_field_of_its_name():
    """
    A property of a variable reads the field of the value type with its name.
    """
    first = ExpressionMemberOrDomainField(
        member_name="first", member=property(lambda expression: 2017)
    )
    period = variable(Period, domain=[])

    assert type(first.__get__(period, type(period))) is Attribute


def test_a_property_stays_a_property_when_no_field_has_its_name():
    """
    Without a field of its name, a property of a variable reads the property.
    """
    first = ExpressionMemberOrDomainField(
        member_name="first", member=property(lambda expression: 2017)
    )
    tax_year = variable(TaxYear, domain=[])

    assert first.__get__(tax_year, type(tax_year)) == 2017

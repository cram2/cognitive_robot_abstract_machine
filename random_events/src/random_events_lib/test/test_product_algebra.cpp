#include "gtest/gtest.h"
#include "product_algebra.h"
#include "interval.h"
#include "set.h"
#include "sigma_algebra.h"
#include "variable.h"
#include <memory>

auto all_elements_int = make_shared_all_elements(std::set<long long>{0, 1, 2});

auto s0 = make_shared_set_element(0, all_elements_int);
auto s1 = make_shared_set_element(1, all_elements_int);
auto s2 = make_shared_set_element(2, all_elements_int);

TEST(SimpleEventTest, IntersectionWith) {
    // Define the sets and intervals
    auto sabc = make_shared_simple_set_set();
    sabc->insert(s0);
    sabc->insert(s1);
    sabc->insert(s2);

    auto sab = make_shared_simple_set_set();
    sab->insert(s0);
    sab->insert(s1);

    auto name = std::make_shared<std::string>("a");
    auto a_a = make_shared_symbolic(name, make_shared_set(s0, all_elements_int));
    auto a_b = make_shared_symbolic(name, make_shared_set(s1, all_elements_int));
    auto a_ab = make_shared_symbolic(name, make_shared_set(sab, all_elements_int));
    auto a_abc = make_shared_symbolic(name, make_shared_set(sabc, all_elements_int));
    auto c = make_shared_symbolic(name, make_shared_set(s2, all_elements_int));
    auto x = make_shared_continuous("x");
    auto y = make_shared_continuous("y");

    auto var = make_shared_variable_set();
    var->insert(a_ab);
    var->insert(x);
    var->insert(y);

    auto var2 = make_shared_variable_set();
    var2->insert(a_a);
    var2->insert(x);

    auto var3 = make_shared_variable_set();
    var3->insert(c);

    auto var_exp = make_shared_variable_set();
    var_exp->insert(a_a);
    var_exp->insert(x);
    var_exp->insert(y);

    // Create SimpleEvent instances
    auto event1 = make_shared_simple_event(var);
    auto event2 = make_shared_simple_event(var2);
    auto event_exp = make_shared_simple_event(var_exp);

    // Perform intersection
    auto intersection = event1->intersection_with(event2);

    auto var_set_test = make_shared_variable_set();
    var_set_test->insert(a_a);

    auto var_set_test2 = make_shared_variable_set();
    var_set_test2->insert(a_ab);

    // ASSERT_EQ(*var_set_test, *var_set_test2);

    // Assertions
    ASSERT_TRUE(*intersection == *event_exp);
    ASSERT_TRUE(*intersection != *event1);

    // Test for empty intersection
    auto event3 = make_shared_simple_event(var3);
    auto second_intersection = std::static_pointer_cast<SimpleEvent>(event1->intersection_with(event3));
    ASSERT_TRUE(second_intersection->is_empty());
}

TEST(ProductAlgebra, EmptyUnion) {
    const auto empty_event = make_shared_event();
    const auto empty_event2 = make_shared_event();
    const auto union_event = empty_event->union_with(empty_event2);

    ASSERT_TRUE(union_event->is_empty());

    const auto intersection_event = empty_event->intersection_with(empty_event2);

    ASSERT_TRUE(intersection_event->is_empty());
}

TEST(ProductAlgebra, VariableGetting) {
    // Define the sets and intervals
    auto sabc = make_shared_simple_set_set();
    sabc->insert(s0);
    sabc->insert(s1);
    sabc->insert(s2);

    auto sab = make_shared_simple_set_set();
    sab->insert(s0);
    sab->insert(s1);

    auto a_a = make_shared_symbolic("a", make_shared_set(s0, all_elements_int));
    auto a_b = make_shared_symbolic("a", make_shared_set(s1, all_elements_int));
    auto a_ab = make_shared_symbolic("a", make_shared_set(sab, all_elements_int));
    auto a_abc = make_shared_symbolic("a", make_shared_set(sabc, all_elements_int));
    auto c = make_shared_symbolic("a", make_shared_set(s2, all_elements_int));
    auto b_a = make_shared_symbolic("b", make_shared_set(s0, all_elements_int));
    auto x = make_shared_continuous("x");
    auto y = make_shared_continuous("y");

    auto map1 = std::make_shared<VariableMap>();
    map1->insert({a_a, a_a->domain});
    map1->insert({b_a, b_a->domain});
    map1->insert({a_ab, a_ab->domain});

    auto event1 = SimpleEvent(map1);

    auto map2 = std::make_shared<VariableMap>();
    map2->insert({x, x->domain});
    map2->insert({a_abc, a_abc->domain});

    auto event2 = SimpleEvent(map2);


    // should be two due to a_a and a_ab being the same variable
    ASSERT_TRUE(map1->size() == 2);

    auto event2_variables = event2.get_variables();

    // x, a_abc
    ASSERT_TRUE(event2_variables->size() == 2);

    auto all_variables = event2.merge_variables(event1.get_variables());
    // a_a, b_a, x
    ASSERT_TRUE(all_variables->size() == 3);
}

TEST(ProductAlgebra, Complement){
     // Define the sets and intervals
    auto sab = make_shared_simple_set_set();
    sab->insert(s0);
    sab->insert(s1);

    auto a_a = make_shared_symbolic("a", make_shared_set(s0, all_elements_int));
    auto a_b = make_shared_symbolic("a", make_shared_set(s1, all_elements_int));
    auto a_ab = make_shared_symbolic("a", make_shared_set(sab, all_elements_int));
    auto c = make_shared_symbolic("a", make_shared_set(s2, all_elements_int));
    auto b_a = make_shared_symbolic("b", make_shared_set(s0, all_elements_int));
    auto x = make_shared_continuous("x");
    auto y = make_shared_continuous("y");


    auto interval1 = SimpleInterval::make_shared(0.0, 1.0, BorderType::CLOSED, BorderType::OPEN);
    auto simple_interval = make_shared_simple_set_set();
    simple_interval->insert(interval1);

    auto variables = std::make_shared<VariableMap>();
    variables->insert({a_ab, a_ab->domain});
    variables->insert({x, Interval::make_shared(simple_interval)});
    variables->insert({y, y->domain});

    auto event = make_shared_simple_event(variables);

    auto complement = event->complement();

    // ASSERT_EQ(complement->size(), 2);

    auto expected_1 = std::make_shared<VariableMap>();
    expected_1->insert({c, c->domain});
    expected_1->insert({x, x->domain});
    expected_1->insert({y, y->domain});

    auto expected_2 = std::make_shared<VariableMap>();
    expected_2->insert({a_ab, a_ab->domain});
    expected_2->insert({x, variables->at(x)->complement()});
    expected_2->insert({y, y->domain});

    auto complement_1 = make_shared_simple_event(expected_1);
    auto complement_2 = make_shared_simple_event(expected_2);

    auto simple_set_set = make_shared_simple_set_set();
    simple_set_set->insert(complement_1);
    simple_set_set->insert(complement_2);

    ASSERT_TRUE(compare_sets(complement, simple_set_set));
}


TEST(ProductAlgebra, Simplify){
    // Define the sets and intervals
    auto sabc = make_shared_simple_set_set();
    sabc->insert(s0);
    sabc->insert(s1);
    sabc->insert(s2);

    auto sab = make_shared_simple_set_set();
    sab->insert(s0);
    sab->insert(s1);

    auto a_a = make_shared_symbolic("a", make_shared_set(s0, all_elements_int));
    auto a_b = make_shared_symbolic("a", make_shared_set(s1, all_elements_int));
    auto a_ab = make_shared_symbolic("a", make_shared_set(sab, all_elements_int));
    auto a_abc = make_shared_symbolic("a", make_shared_set(sabc, all_elements_int));
    auto c = make_shared_symbolic("a", make_shared_set(s2, all_elements_int));
    auto x = make_shared_continuous("x");
    auto y = make_shared_continuous("y");

    auto interval1 = SimpleInterval::make_shared(0.0, 1.0, BorderType::CLOSED, BorderType::OPEN);
    auto simple_interval = make_shared_simple_set_set();
    simple_interval->insert(interval1);

    auto variables = std::make_shared<VariableMap>();
    variables->insert({a_ab, a_ab->domain});
    variables->insert({x, Interval::make_shared(simple_interval)});
    variables->insert({y, Interval::make_shared(simple_interval)});

    auto variables2 = std::make_shared<VariableMap>();
    variables2->insert({c, c->domain});
    variables2->insert({x, Interval::make_shared(simple_interval)});
    variables2->insert({y, Interval::make_shared(simple_interval)});

    auto event1 = make_shared_simple_event(variables);
    // (*event1->variable_map)[x] = closed(0, 1);
    // (*event1->variable_map)[y] = closed(0, 1);
    auto event2 = make_shared_simple_event(variables2);


    auto composite_event = make_shared_event(event1);
    composite_event->simple_sets->insert(event2);
    auto result = composite_event->simplify();

    ASSERT_EQ(result->simple_sets->size(), 1);

    auto contained_event = std::static_pointer_cast<SimpleEvent>(*result->simple_sets->begin());
    ASSERT_EQ(contained_event->variable_map->size(), 3);
    ASSERT_EQ(contained_event->variable_map->at(x)->simple_sets->size(), 1);

    auto simplified = composite_event->simplify();
    ASSERT_EQ(*simplified, *result);

}

TEST(ProductAlgebra, UnionDifferentVariables) {
    const auto continuous1 = make_shared_continuous("x");
    const auto continuous2 = make_shared_continuous("y");

    auto var_map1 = std::make_shared<VariableMap>();
    var_map1->insert({continuous1, closed(0, 1)});

    auto var_map2 = std::make_shared<VariableMap>();
    var_map2->insert({continuous2, closed(3, 4)});

    const auto simple_event_x = make_shared_simple_event(var_map1);
    const auto simple_event_y = make_shared_simple_event(var_map2);

    auto e1 = make_shared_event(simple_event_x);
    auto e2 = make_shared_event(simple_event_y);

    auto x = make_shared_variable_set(e2->get_variables_from_simple_events());
    e1->fill_missing_variables(x);
    auto y = make_shared_variable_set(e1->get_variables_from_simple_events());
    e2->fill_missing_variables(y);

    auto union_event = e1->union_with(e2);

    auto expected_var_map1 = std::make_shared<VariableMap>();
    expected_var_map1->insert({continuous1, closed(0, 1)});
    expected_var_map1->insert({continuous2, open(-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity())});

    auto expected_var_map2 = std::make_shared<VariableMap>();
    expected_var_map2->insert({continuous1, open(-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity())});
    expected_var_map2->insert({continuous2, closed(3, 4)});

    const auto expected_simple_event1 = make_shared_simple_event(expected_var_map1);
    const auto expected_simple_event2 = make_shared_simple_event(expected_var_map2);

    const auto simple_set_set = make_shared_simple_set_set();
    simple_set_set->insert(expected_simple_event1);
    simple_set_set->insert(expected_simple_event2);

    const auto expected_result = make_shared_event(simple_set_set)->make_disjoint();

    ASSERT_TRUE(*union_event == *expected_result);
}

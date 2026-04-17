#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "interval.h"
#include "product_algebra.h"
#include "set.h"

namespace py = pybind11;

PYBIND11_MODULE(random_events_lib, handle) {
    handle.doc()= "A module for handling random events";

    py::class_<AbstractSimpleSet, std::shared_ptr<AbstractSimpleSet>>(handle, "AbstractSimpleSet")
        .def("intersection_with", &AbstractSimpleSet::intersection_with)
        .def("complement", [](AbstractSimpleSet &x){return * x.complement();})
        .def("contains", &AbstractSimpleSet::contains)
        .def("is_empty", &AbstractSimpleSet::is_empty)
        .def("difference_with", [](AbstractSimpleSet &x, AbstractSimpleSet &y) {
            auto const p = AbstractSimpleSetPtr_t(&y);
            return *x.difference_with(p);
        })
        .def ("__repr__", &AbstractSimpleSet::to_string)
        .def("__eq__", &AbstractSimpleSet::operator==)
        .def("__lt__", &AbstractSimpleSet::operator<);


    py::class_<AbstractCompositeSet, std::shared_ptr<AbstractCompositeSet>>(handle, "AbstractCompositeSet")
        .def_property("simple_sets",
            [](const AbstractCompositeSet &x){return *x.simple_sets;},
            [](AbstractCompositeSet &x, SimpleSetSet_t const &v){x.simple_sets = make_shared_simple_set_set(v);})
        .def("is_empty", &AbstractCompositeSet::is_empty)
        .def("is_disjoint", &AbstractCompositeSet::is_disjoint)
        .def("simplify", &AbstractCompositeSet::simplify)
        .def("make_disjoint", &AbstractCompositeSet::make_disjoint)
        .def("intersection_with", pybind11::overload_cast<const AbstractCompositeSetPtr_t&>(&AbstractCompositeSet::intersection_with), "Intersect this with another composite set.")
        .def("intersection_with_simple_set", pybind11::overload_cast<const AbstractSimpleSetPtr_t&>(&AbstractCompositeSet::intersection_with), "Intersect this with another simple set.")
        .def("complement", [](const AbstractCompositeSet &x){return x.complement();})
        .def("union_with", pybind11::overload_cast<const AbstractCompositeSetPtr_t&>(&AbstractCompositeSet::union_with), "Union this with another composite set.")
        .def("union_with", pybind11::overload_cast<const AbstractSimpleSetPtr_t&>(&AbstractCompositeSet::union_with), "Union this with a simple set.")
        .def("difference_with", pybind11::overload_cast<const AbstractCompositeSetPtr_t&>(&AbstractCompositeSet::difference_with), "Difference this with another composite set.")
        .def("difference_with", pybind11::overload_cast<const AbstractSimpleSetPtr_t&>(&AbstractCompositeSet::difference_with), "Difference this with a simple set.")
        .def("add_new_simple_set", &AbstractCompositeSet::add_new_simple_set)
        .def("__eq__", &AbstractCompositeSet::operator==)
        .def("__lt__", &AbstractCompositeSet::operator<);


    py::enum_<BorderType>(handle, "BorderType")
        .value("OPEN", BorderType::OPEN)
        .value("CLOSED", BorderType::CLOSED);


    py::class_<SimpleInterval, AbstractSimpleSet, std::shared_ptr<SimpleInterval>>(handle, "SimpleInterval")
        .def(py::init([](float const &lower, float const &upper, int const &left, int const &right) {
            auto x = BorderType::OPEN;
            if(left == 1) {
                x = BorderType::OPEN;
            } else {
                x = BorderType::CLOSED;
            }
            auto y = BorderType::OPEN;
            if (right == 1) {
                y = BorderType::OPEN;
            } else {
                y = BorderType::CLOSED;
            }

            return std::make_shared<SimpleInterval>(lower, upper, x, y);
        }))
        .def_readwrite("lower", &SimpleInterval::lower)
        .def_readwrite("upper", &SimpleInterval::upper)
        .def_readwrite("left", &SimpleInterval::left)
        .def_readwrite("right", &SimpleInterval::right)
        .def("__hash__", [](const SimpleInterval &interval) {
            return std::hash<double>()(interval.lower) ^ std::hash<double>()(interval.upper) ^
                   std::hash<int>()(static_cast<int>(interval.left)) ^ std::hash<int>()(static_cast<int>(interval.right));
        });


    py::class_<Interval, AbstractCompositeSet, std::shared_ptr<Interval>>(handle, "Interval")
        .def(py::init())
        .def(py::init([](SimpleInterval const &x) {
            auto p = std::make_shared<SimpleInterval>(x);
            return std::make_shared<Interval>(p);
        }))
        .def(py::init([](SimpleSetSet_t const &x) {
            auto p = std::make_shared<SimpleSetSet_t>(x);
            return std::make_shared<Interval>(p);
        }));


    handle.def("closed", &closed, "Create a closed interval");
    handle.def("open", &open, "Create an open interval");
    handle.def("closed_open", &closed_open, "Create a closed-open interval");
    handle.def("open_closed", &open_closed, "Create an open-closed interval");
    handle.def("singleton", &singleton, "Create a singleton interval");
    handle.def("empty", &empty, "Create an empty interval");
    handle.def("reals", &reals, "Create the real line interval");


    py::class_<SetElement, AbstractSimpleSet, std::shared_ptr<SetElement>>(handle, "SetElement")
        .def(py::init([](std::set<long long> const &x) {
            auto const p = make_shared_all_elements(x);
            const auto set_element = SetElement(p);
            return make_shared_set_element(set_element);
        }))
        .def(py::init([](int const &x, std::set<long long> const &y) {
            auto const p = make_shared_all_elements(y);
            const auto set_element = SetElement(x, p);
            return make_shared_set_element(set_element);
        }))
        .def_property("element_index", [](SetElement const &x){return x.element_index;},
            [](SetElement &x, int const &v){x.element_index = v;})
        .def_property("all_elements", [](SetElement const &x){return *x.all_elements;},
            [](SetElement &x, std::set<long long> const &v){x.all_elements = make_shared_all_elements(v);})
        .def("__hash__", [](SetElement const &x) {
            return std::hash<int>{}(x.element_index);
        });


    py::class_<Set, AbstractCompositeSet, std::shared_ptr<Set>>(handle, "Set")
        .def(py::init([] (std::set<long long> const &x) {
            auto const p = make_shared_all_elements(x);
            return std::make_shared<Set>(p);
        }))
        .def(py::init([](SimpleSetSet_t const &x, std::set<long long> const &y) {
            auto const p = make_shared_simple_set_set(x);
            auto const q = make_shared_all_elements(y);
            return std::make_shared<Set>(p, q);
        }))
        .def(py::init([](SetElement const &x, std::set<long long> const &y) {
            auto const q = std::make_shared<SetElement>(x);
            auto const p = make_shared_all_elements(y);
            return std::make_shared<Set>(q, p);
        }))
        .def_property("all_elements", [](Set const &x){return *x.all_elements;},
            [](Set &x, std::set<long long> const &v){x.all_elements = make_shared_all_elements(v);});

    py::class_<SimpleEvent, AbstractSimpleSet, std::shared_ptr<SimpleEvent>>(handle, "SimpleEvent")
        .def(py::init())
        .def(py::init([](VariableMap const &x) {
            auto p = std::make_shared<VariableMap>(x);
            return std::make_shared<SimpleEvent>(p);
        }))
        .def(py::init([](VariableSet const &x) {
            auto const p = make_shared_variable_set(x);
            return std::make_shared<SimpleEvent>(p);
        }))
        .def_property("variable_map", [](SimpleEvent const &x){return * x.variable_map;},
            [](SimpleEvent x, VariableMap const &v){x.variable_map = std::make_shared<VariableMap>(v);})
        .def("marginal", [](const SimpleEvent &x, VariableSet const &y) {
            auto const p = make_shared_variable_set(y);
            return x.marginal(p);
        })
        .def("fill_missing_variables", [](const SimpleEvent &e, const VariableSet &v) {
            auto const p = make_shared_variable_set(v);
            e.fill_missing_variables(p);})
        .def("__hash__", [](SimpleEvent const &x) {
            return VariableMapHash{}(*x.variable_map);
        });

    py::class_<Event, AbstractCompositeSet, std::shared_ptr<Event>>(handle, "Event")
        .def(py::init())
        .def(py::init([](SimpleSetSet_t const &x) {
            auto p = std::make_shared<SimpleSetSet_t>(x);
            return make_shared_event(p);
        }))
        .def(py::init([](SimpleEvent const &x) {
            auto p = std::make_shared<SimpleEvent>(x);
            return make_shared_event(p);
        }))
        .def("simplify_once", &Event::simplify_once)
        .def("fill_missing_variables", [](const Event &e, const VariableSet &v) {
            auto const p = make_shared_variable_set(v);
            e.fill_missing_variables(p);
        })
        .def("marginal", [](const Event &x, VariableSet const &y) {
            auto const p = make_shared_variable_set(y);
            return x.marginal(p);
        });


    py::class_<AbstractVariable, std::shared_ptr<AbstractVariable>>(handle, "AbstractVariable")
        // .def("get_domain", &AbstractVariable::get_domain)
        .def("__eq__", &AbstractVariable::operator==)
        .def("__lt__", &AbstractVariable::operator<)
        .def("__hash__", [](AbstractVariable const &x) {
            return std::hash<std::string>{}(*x.name);
        });

    py::class_<Symbolic, AbstractVariable, std::shared_ptr<Symbolic>>(handle, "Symbolic")
        .def(py::init([](std::string const &x, Set const &y) {
            auto const p = std::make_shared<std::string>(x);
            auto const q = std::make_shared<Set>(y);
            return std::make_shared<Symbolic>(p, q);
        }))
        .def(py::init([](char* const x, Set const &y) {
            auto const p = std::make_shared<std::string>(x);
            auto const q = std::make_shared<Set>(y);
            return std::make_shared<Symbolic>(p, q);
        }))
        .def_property("name", [](Symbolic const &x){return *x.name;},
            [](Symbolic &x, std::string const &v){x.name = std::make_shared<std::string>(v);})
        .def_property("domain", [](Symbolic const &x){return *x.domain;},
            [](Symbolic &x, Set const &v){x.domain = std::make_shared<Set>(v);});


    py::class_<Continuous, AbstractVariable, std::shared_ptr<Continuous>>(handle, "Continuous")
        .def(py::init([](std::string const &x) {
            auto const p = std::make_shared<std::string>(x);
            return std::make_shared<Continuous>(p);
        }))
        .def(py::init([](char* const x) {
            auto const p = std::make_shared<std::string>(x);
            return std::make_shared<Continuous>(p);
        }))
        .def_property("name", [](Continuous const &x){return *x.name;},
            [](Continuous &x, std::string const &v){x.name = std::make_shared<std::string>(v);});


    py::class_<Integer, AbstractVariable, std::shared_ptr<Integer>>(handle, "Integer")
        .def(py::init([](std::string const &x) {
            auto const p = std::make_shared<std::string>(x);
            return std::make_shared<Integer>(p);
        }))
        .def(py::init([](char* const x) {
            auto const p = std::make_shared<std::string>(x);
            return std::make_shared<Integer>(p);
        }))
        .def_property("name", [](Integer const &x){return *x.name;},
            [](Integer &x, std::string const &v){x.name = std::make_shared<std::string>(v);});

}

#pragma once

#include "sigma_algebra.h"
#include <set>
#include <utility>

// FORWARD DECLARATIONS
class SetElement;
class Set;


// TYPEDEFS
using AllSetElementsPtr_t = std::shared_ptr<std::set<long long>>;
using SetElementPtr_t = std::shared_ptr<SetElement>;
using SetPtr_t = std::shared_ptr<Set>;

template<typename... Args>
AllSetElementsPtr_t make_shared_all_elements(Args &&... args) {
    return std::make_shared<std::set<long long>>(std::forward<Args>(args)...);
}


template<typename... Args>
SetElementPtr_t make_shared_set_element(Args &&... args) {
    return std::make_shared<SetElement>(std::forward<Args>(args)...);
}


template<typename... Args>
SetPtr_t make_shared_set(Args &&... args) {
    return std::make_shared<Set>(std::forward<Args>(args)...);
}


class SetElement : public AbstractSimpleSet {
public:

    /**
     * The set of all possible strings
     */
    AllSetElementsPtr_t all_elements;

    /**
     * The index of the element_index in the all_elements set
     */
    int element_index;

    explicit SetElement(const AllSetElementsPtr_t &all_elements_);

    SetElement(int element_index, const AllSetElementsPtr_t &all_elements_);

    ~SetElement() override;

    AbstractSimpleSetPtr_t intersection_with(const AbstractSimpleSetPtr_t &other) override;

    SimpleSetSetPtr_t complement() override;

    bool contains(const ElementaryVariant *element) override;

    bool is_empty() override;

    /**
     * Two simple sets are equal if the element_index is equal. The all_elements set is not considered.
     *
     * @param other The other simple set.
     * @return True if they are equal.
     */
    bool operator==(const AbstractSimpleSet &other) override;

    bool operator==(const SetElement &other);

    std::string *non_empty_to_string() override;

    bool operator<(const AbstractSimpleSet &other) override;

    /**
     * Compare two set elements. Set elements are ordered by their element index.
     *
     * Note that all elements set is ignored in ordering.
     *
     * @param other The other interval
     * @return True if this interval is less than the other interval.
     */
    bool operator<(const SetElement &other);


    /**
    * Compare two simple intervals. Simple intervals are ordered by lower bound. If the lower bound is equal, they are
    * ordered by upper bound.
    *
    * Note that border types are ignored in ordering.
    *
    * @param other The other interval
    * @return True if this interval is less or equal to the other interval.
    */
    bool operator<=(const SetElement &other);

};


class Set : public AbstractCompositeSet {
public:

    AllSetElementsPtr_t all_elements;

    explicit Set(const AllSetElementsPtr_t& all_elements_);
    Set(const SetElementPtr_t& element_, const AllSetElementsPtr_t& all_elements_);
    Set(const SimpleSetSetPtr_t& elements, const AllSetElementsPtr_t& all_elements_);

    ~Set() override;

    AbstractCompositeSetPtr_t simplify() override;

    AbstractCompositeSetPtr_t make_new_empty() const override;

    std::string *to_string() override;

};
#pragma once

#include <set>
#include <vector>
#include <tuple>
#include <memory>
#include <string>

// FORWARD DECLARATIONS
class AbstractSimpleSet;
class AbstractCompositeSet;


// TYPE DEFINITIONS
template<typename T>
struct PointerLess {
    bool operator()(T const &lhs, T const &rhs) const {
        return *lhs < *rhs;
    }
};

typedef std::shared_ptr<AbstractSimpleSet> AbstractSimpleSetPtr_t;
typedef std::shared_ptr<AbstractCompositeSet> AbstractCompositeSetPtr_t;

typedef std::set<AbstractSimpleSetPtr_t, PointerLess<AbstractSimpleSetPtr_t>> SimpleSetSet_t;
typedef std::shared_ptr<SimpleSetSet_t> SimpleSetSetPtr_t;

template<typename... Args>
SimpleSetSetPtr_t make_shared_simple_set_set(Args&&... args) {
    return std::make_shared<SimpleSetSet_t>(std::forward<Args>(args)...);
}

static std::string EMPTY_SET_SYMBOL = "∅";

union ElementaryVariant {
    float f;
    int i;
    std::string s;
};

template <typename T>
bool compare_sets(const T &lhs, const T &rhs) {
    if (lhs->size() != rhs->size()) {
        return false;
    }
    auto it_lhs = lhs->begin();
    auto end_lhs = lhs->end();
    auto it_rhs = rhs->begin();

    while (it_lhs != end_lhs) {
        if (**it_lhs != **it_rhs) {
            return false;
        }
        ++it_lhs;
        ++it_rhs;
    }
    return true;
}

class AbstractSimpleSet : public std::enable_shared_from_this<AbstractSimpleSet>{
public:
    virtual ~AbstractSimpleSet() = default;

    /**
    * Intersect this with another simple set.
    *
    * @param other the other simples set.
    * @return The intersection of both as simple set.
    */
    virtual AbstractSimpleSetPtr_t intersection_with(const AbstractSimpleSetPtr_t &other)= 0;

    /**
    * This method depends on the type of simple set and has to be overwritten.
    *
    * @return The complement of this simple set as disjoint composite set.
    */
    virtual SimpleSetSetPtr_t complement()= 0;

    /**
    * Check if an elementary event is contained in this.
    *
    * @param element The element_index to check.
    * @return True if the element_index is contained in this.
    */
    virtual bool contains(const ElementaryVariant *element)= 0;


    /**
    * This method depends on the type of simple set and has to be overwritten.
    *
    * @return True if this is empty.
    */
    virtual bool is_empty()= 0;

    /**
    * Form the difference with another simple set.
    *
    * @param other The other simple set.
    * @return The difference as disjoint composite set.
    */
    SimpleSetSetPtr_t difference_with(const AbstractSimpleSetPtr_t& other);

    virtual std::string *non_empty_to_string()= 0;

    std::string *to_string();

    virtual bool operator==(const AbstractSimpleSet &other)= 0;

    virtual bool operator<(const AbstractSimpleSet &other)= 0;

    bool operator!=(const AbstractSimpleSet &other);

    std::shared_ptr<AbstractSimpleSet> share_more()
    {
        return shared_from_this();
    }

};

/**
 * Unique Combinations of elements within a vector.
 * The unique combinations are pairs of elements which exclude:
 * - symmetric pairs (A, A)
 * - (A,B) if (B, A) is already visited.
 *
 * @param elements The vector.
 * @return The unique combinations of elements of the vector.
 */
template<typename T>
std::vector<std::tuple<T, T>> unique_combinations(const std::vector<T> &elements) {
    size_t n = elements.size();
    size_t total = n > 1 ? n*(n-1)/2 : 0;
    std::vector<std::tuple<T, T>> result;
    result.reserve(total);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            result.emplace_back(elements[i], elements[j]);
    return result;
}


/**
* Abstract class for composite elements.
* Composite elements contain a **disjoint** union of (abstract) simple sets.
*/
class AbstractCompositeSet : public std::enable_shared_from_this<AbstractCompositeSet>{
public:

    SimpleSetSetPtr_t simple_sets;

    AbstractCompositeSet() = default;

    virtual ~AbstractCompositeSet() {
        simple_sets->clear();
    }

    /**
    * @return True if this is empty.
    */
    bool is_empty();

    /**
     * @return True if the composite set is disjoint union of simple sets.
     */
    bool is_disjoint();

    /**
    * Simplify the composite set into a shorter but equal representation.
    * The size (shortness9 refers to the number of simple sets contained.
    *
    * @return The simplified composite set into a shorter but equal representation.
    */
    virtual AbstractCompositeSetPtr_t simplify()= 0;

    /**

     * @return A **new** empty composite set
     */
    virtual AbstractCompositeSetPtr_t make_new_empty() const = 0;

    virtual /**
     * @return A string representation of this.
     */
    std::string *to_string();

    bool operator==(const AbstractCompositeSet &other) const;
    bool operator!=(const AbstractCompositeSet &other) const;
    bool operator<(const AbstractCompositeSet &other) const;

    /**
    * Split this composite set into disjoint and non-disjoint parts.
    *
    * This method is required for making the composite set disjoint.
    * The partitioning is done by removing every other simple set from every simple set.
    * The purified simple sets are then disjoint by definition and the pairwise intersections are (potentially)
    * not disjoint yet.
    *
    * This method requires:
    *  - the intersection of two simple sets as a simple set
    *  - the difference of a simple set (A) and another simple set (B) that is completely contained in A (B ⊆ A).
    *      The result of that difference has to be a composite set with only one simple set in it.
    *
    * @return A tuple of disjoint and non-disjoint composite sets.
    */
    std::tuple<AbstractCompositeSetPtr_t, AbstractCompositeSetPtr_t> split_into_disjoint_and_non_disjoint() const;

    /**
    * Create an equal composite set that contains a disjoint union of simple sets.
    *
    * @return The disjoint composite set.
    */
    AbstractCompositeSetPtr_t make_disjoint() const;

    /**
     * Form the intersection with an simple set.
     * The intersection is only disjoint if this is disjoint.
     * @param simple_set The simple event to intersect with.
     * @return The intersection.
     */
    AbstractCompositeSetPtr_t intersection_with(const AbstractSimpleSetPtr_t &simple_set);

    AbstractCompositeSetPtr_t intersection_with(const SimpleSetSetPtr_t &other);

    /**
    * Form the intersection with another composite set.
    *
    * The intersection is only disjoint if both composite sets are disjoint.
    *
    * @param other The other composite set.
    * @return The intersection as composite set.
    */
    AbstractCompositeSetPtr_t intersection_with(const AbstractCompositeSetPtr_t &other);

    /**
     * @return the complement of a composite set as disjoint composite set.
     */
    AbstractCompositeSetPtr_t complement() const;

    /**
    * Form the union with a simple set.
    *
    * @param other The other simple set.
    * @return The union as disjoint composite set.
    */
    AbstractCompositeSetPtr_t union_with(const AbstractSimpleSetPtr_t &other);

    /**
    * Form the union with another composite set.
    *
    * @param other The other composite set.
    * @return The union as disjoint composite set.
    */
    AbstractCompositeSetPtr_t union_with(const AbstractCompositeSetPtr_t &other);

    /**
     * Form the difference with a simple set.
     *
     * @param other the simple set
     * @return The difference as disjoint composite set.
     */
    AbstractCompositeSetPtr_t difference_with(const AbstractSimpleSetPtr_t &other);

    /**
     * Form the difference with another composite set.
     *
     * @param other The other composite set.
     * @return The difference as disjoint composite set.
     */
    AbstractCompositeSetPtr_t difference_with(const AbstractCompositeSetPtr_t &other);

    bool contains(const AbstractCompositeSetPtr_t &other);

    void add_new_simple_set(const AbstractSimpleSetPtr_t& simple_set) const;

};

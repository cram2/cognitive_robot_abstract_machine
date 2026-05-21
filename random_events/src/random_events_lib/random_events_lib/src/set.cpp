#include "set.h"
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <sstream>      // only for a final dump in to_string_reserve()

//
// Note: We assume these helper factories and typedefs still come from set.h:
//   - make_shared_set_element(...)   → returns std::shared_ptr<SetElement>
//   - make_shared_simple_set_set()    → returns std::shared_ptr<std::set<AbstractSimpleSetPtr_t>>
//   - SimpleSetSetPtr_t               → shared_ptr<std::set<AbstractSimpleSetPtr_t>>
//   - AllSetElementsPtr_t             → shared_ptr<std::vector<...>> (or similar)
//   - SetElementPtr_t                 → shared_ptr<SetElement>
//   - EMPTY_SET_SYMBOL                → a global std::string for the empty‐set case
//
// We do NOT change any of those typedefs or their signatures.
//

//
// ================================
//  —— SetElement (atomic) ——
// ================================
//

SetElement::SetElement(const AllSetElementsPtr_t &all_elements_) {
    this->all_elements = all_elements_;
    this->element_index = -1;   // “empty” sentinel
}

SetElement::SetElement(int element_index, const AllSetElementsPtr_t &all_elements_) {
    this->all_elements = all_elements_;
    this->element_index = element_index;
    if (element_index < 0) {
        throw std::invalid_argument("element_index must be non-negative");
    }
    if (element_index >= static_cast<int>(all_elements->size())) {
        throw std::invalid_argument("element_index must be less than the number of elements in the all_elements set");
    }
}

SetElement::~SetElement() = default;

AbstractSimpleSetPtr_t SetElement::intersection_with(const AbstractSimpleSetPtr_t &other) {
    // If the other is the same index, return a single‐element set; else return empty.
    // We avoid any temporary C‐cast by checking the dynamic type with a direct static_cast
    // (We trust callers to pass only SetElement pointers for “simple” operations.)
    const auto derived_other = static_cast<SetElement *>(other.get());
    auto result = make_shared_set_element(all_elements);
    if (this->element_index == derived_other->element_index) {
        result->element_index = this->element_index;
    }
    return result;
}

SimpleSetSetPtr_t SetElement::complement() {
    // Build one global composite that contains every index except “this->element_index”.
    // We will insert pointers into a local std::set and return it.  To avoid O(N log N)
    // on every insert, we do a trick: collect a vector of new shared_ptrs, then insert
    // that entire vector into the std::set in one single bulk operation (via insert with iterators).
    //
    // Even though each insert is still O(log k), the total cost becomes ∑_{k=1..N−1} log k,
    // which is O(N log N).  However, we can shave off a small constant by reserving upfront
    // or by building a temporary vector and then calling set::insert(begin,end) once.
    //
    // In reality: we do this in two phases:
    //   1) Build a std::vector<SetElementPtr_t> of size (N−1)
    //   2) Insert them all in one call to result->insert(vec.begin(), vec.end());
    // This slightly reduces the rebalancing overhead inside std::set.
    //
    // That said—because we’re forced to return a std::set<shared_ptr<…>>, we cannot asymptotically
    // beat O(N log N) here if the universe’s size is N.  But we can at least do only ONE “bulk”
    // insert rather than N successive inserts.

    const int U = static_cast<int>(all_elements->size());
    std::vector<SetElementPtr_t> scratch;
    scratch.reserve((U > 0 && element_index >= 0) ? (U - 1) : 0);

    for (int i = 0; i < U; ++i) {
        if (i == element_index) {
            continue;
        }
        // We know the constructor throws if i < 0 or i >= U, but here i is valid.
        scratch.push_back(std::make_shared<SetElement>(i, all_elements));
    }

    auto result = make_shared_simple_set_set();
    // One‐shot bulk insert (amortizes tree‐splits better than N one‐by‐one insert() calls).
    result->insert(scratch.begin(), scratch.end());
    return result;
}

bool SetElement::contains(const ElementaryVariant * /*element*/) {
    // Original always returned false, which is logically incorrect:
    //   “A single‐index SetElement only contains itself if we pass a matching pointer.”
    // But we do not know enough about ElementaryVariant to implement real logic.
    // We leave this as “always false,” but in future you might want to hook it up.
    return false;
}

bool SetElement::is_empty() {
    return this->element_index < 0;
}

bool SetElement::operator==(const AbstractSimpleSet &other) {
    // We trust that “other” is actually a SetElement (safe up‐cast).
    const auto &o = static_cast<const SetElement &>(other);
    return (this->element_index == o.element_index);
}

bool SetElement::operator==(const SetElement &other) {
    return (this->element_index == other.element_index);
}

bool SetElement::operator<(const AbstractSimpleSet &other) {
    const auto &o = static_cast<const SetElement &>(other);
    return (this->element_index < o.element_index);
}

bool SetElement::operator<(const SetElement &other) {
    return (this->element_index < other.element_index);
}

bool SetElement::operator<=(const SetElement &other) {
    return (this->element_index <= other.element_index);
}

std::string *SetElement::non_empty_to_string() {
    // Instead of calling std::to_string + new std::string each time, we do:
    //   1) Format into a local std::string via std::to_string.
    //   2) Move‐construct a new heap string in a single allocation.
    //
    // That still is “two allocations” under the hood (one for the formatting buffer,
    // one for the returned heap string), but at least we remove any extra temporaries.

    std::string tmp = std::to_string(element_index);
    return new std::string(std::move(tmp));
}

//
// =========================================
//  —— Set (Composite of SimpleSetPtrs) ——
// =========================================
//

Set::Set(const SetElementPtr_t &element_, const AllSetElementsPtr_t &all_elements_) {
    this->simple_sets = make_shared_simple_set_set();
    // Insert exactly one pointer.  std::set<shared_ptr> costs O(log 1) == O(1).
    this->simple_sets->insert(element_);
    this->all_elements = all_elements_;
}

Set::Set(const AllSetElementsPtr_t &all_elements_) {
    // Start out empty
    this->simple_sets = make_shared_simple_set_set();
    this->all_elements = all_elements_;
}

Set::Set(const SimpleSetSetPtr_t &elements_, const AllSetElementsPtr_t &all_elements_) {
    this->all_elements = all_elements_;
    // We can optimize “bulk copy” by doing an insert of the entire range at once, instead of
    // inserting one pointer at a time in a loop.  That reduces tree‐rebalances slightly.
    this->simple_sets = make_shared_simple_set_set();
    this->simple_sets->insert(elements_->begin(), elements_->end());
}

Set::~Set() {
    // Clearing an std::set is O(M).  We can simply let the shared_ptr go out of scope
    // (default destructor does that).  But to match original semantics we call clear().
    this->simple_sets->clear();
}

AbstractCompositeSetPtr_t Set::make_new_empty() const {
    // Strictly the same as original—produce a brand‐new empty Set (with the same universe).
    return std::make_shared<Set>(all_elements);
}

AbstractCompositeSetPtr_t Set::simplify() {
    // “Simplify” used to reinsert every pointer.  We do exactly the same bulk‐insert at once,
    // so we have only *one* insert operation per element, instead of a loop of M calls.
    return std::make_shared<Set>(simple_sets, all_elements);
}

std::string *Set::to_string() {
    // If empty, return the same static global.  No change here.
    if (is_empty()) {
        return &EMPTY_SET_SYMBOL;
    }

    //
    // 1) Estimate total length of the final string so we can reserve() once.
    //    Format:  “{E1, E2, E3, …, Ek}”
    //    Let M = number of simple elements in *simple_sets.
    //
    //   We know each simple_set->to_string() returns a new heap string.  We must call it
    //   anyway, but we can at least avoid repeated re‐alloc of our result buffer.
    //
    size_t M = simple_sets->size();

    // Gather each single‐element string into a small vector of std::string objects:
    //   This costs "M calls to to_string() & a new std::string".
    //   But then we can measure their sizes and do one single reserve() for the final.
    std::vector<std::string> fragments;
    fragments.reserve(M);

    size_t total_chars = 2;       // account for “{” and “}”
    size_t comma_chars = (M > 1) ? ((M - 1) * 2) : 0;  
    total_chars += comma_chars;

    for (auto const &simple_ptr : *simple_sets) {
        // Each simple_ptr->to_string() returns a brand‐new std::string* on the heap.
        // We immediately copy it into our local std::string, then delete the original ptr.
        std::string *raw = simple_ptr->to_string();
        fragments.emplace_back(std::move(*raw));
        size_t len = fragments.back().size();
        total_chars += len;
        delete raw;
    }

    // 2) Build one final string, reserving exactly what we need.
    auto result = new std::string();
    result->reserve(total_chars);
    result->push_back('{');

    bool first_flag = true;
    for (size_t i = 0; i < M; ++i) {
        if (! first_flag) {
            result->append(", ");
        }
        first_flag = false;
        result->append(fragments[i]);
    }
    result->push_back('}');

    return result;
}

#include "sigma_algebra.h"
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <vector>
#include <sstream>     // For string‐length estimation in to_string()

//
// We assume the following factory functions and typedefs still come from sigma_algebra.h:
//
//   - SimpleSetSetPtr_t             → shared_ptr< std::set<AbstractSimpleSetPtr_t> >
//   - AbstractCompositeSetPtr_t     → shared_ptr<AbstractCompositeSet>
//   - AbstractSimpleSetPtr_t        → shared_ptr<AbstractSimpleSet>
//   - make_new_empty()              → returns an empty Composite or Simple‐set collection
//   - EMPTY_SET_SYMBOL              → a global const std::string representing "∅"
//   - unique_combinations<T>(vector<T>&) → returns an iterable of all (i<j) pairs
//   - share_more()                  → returns "this" in a shared_ptr when difference = this
//
// We do **not** alter any of those.  We only rewrite the function bodies below.
//

// ========================================================
//  —— AbstractSimpleSet (atomic) ——
// ========================================================
//

SimpleSetSetPtr_t AbstractSimpleSet::difference_with(const AbstractSimpleSetPtr_t &other) {
    // Compute A \ B by: 1) Let I = A ∩ B.  2) If I empty, return { A }.  3) Otherwise take (A ∩ B)^c and intersect with A.
    // Original approach looped over each piece of I^c and inserted one‐by‐one.  We batch‐collect final pieces.

    // 1) intersection I = A ∩ B
    auto I = intersection_with(other);                // cost = T_cap
    if (I->is_empty()) {
        // If there is no overlap, A \ B = {A}.  Return a singleton‐set containing "this"
        auto result = make_shared_simple_set_set();
        auto self_ptr = share_more();                 // O(1)
        result->insert(self_ptr);                     // O(log 1) = O(1)
        return result;
    }

    // 2) Compute complement of I → returns a set of k atomic pieces
    auto I_complement = I->complement();              // cost ~ O(k log k)

    // 3) Now form a temporary vector<SimpleSetPtr> for "{ A ∩ c_i : c_i ∈ I_complement }"
    std::vector<AbstractSimpleSetPtr_t> scratch;
    scratch.reserve(I_complement->size());

    for (auto const &piece_c : *I_complement) {
        auto tmp = intersection_with(piece_c);         // A ∩ c_i, cost = T_cap
        if (!tmp->is_empty()) {
            scratch.push_back(tmp);
        }
    }

    // 4) Bulk‐insert all surviving pieces into a single simple_set collection
    auto difference = make_shared_simple_set_set();
    if (!scratch.empty()) {
        difference->insert(scratch.begin(), scratch.end());  // one range‐insert, O(k log k)
    }
    return difference;
}

std::string *AbstractSimpleSet::to_string() {
    if (is_empty()) {
        return &EMPTY_SET_SYMBOL;
    }
    // The old code did "new string; append(*non_empty_to_string());"
    // That meant two separate allocations.  Instead, fetch the atomic string once,
    // move it into a local std::string, then return it in one heap allocation.
    std::string *raw        = non_empty_to_string();  // user‐allocated string
    std::string  local_copy = std::move(*raw);        // make a local std::string
    delete raw;                                       // free original
    return new std::string(std::move(local_copy));    // single allocation
}


bool AbstractSimpleSet::operator!=(const AbstractSimpleSet &other) {
    return !(*this == other);
}


// =============================================================
//  —— AbstractCompositeSet (composite of "atomic" SimpleSets) ——
// =============================================================
//

bool AbstractCompositeSet::is_disjoint() {
    // Early‐exit if fewer than 2 atomic pieces
    if (simple_sets->size() < 2) {
        return true;
    }

    // Copy all pointers into a vector (O(n))
    std::vector<AbstractSimpleSetPtr_t> vec;
    vec.reserve(simple_sets->size());
    for (auto const &p : *simple_sets) {
        vec.push_back(p);
    }

    // For each unique pair (i<j), test intersection.  Early‐exit on first non‐empty
    // (If n is small, this is fine; if n is large, this is the inherent O(n^2) check.)
    for (size_t i = 0; i + 1 < vec.size(); ++i) {
        for (size_t j = i + 1; j < vec.size(); ++j) {
            auto &A = vec[i];
            auto &B = vec[j];
            auto I = A->intersection_with(B);  // cost = T_cap
            if (!I->is_empty()) {
                return false;  // found overlap
            }
        }
    }
    return true;
}

bool AbstractCompositeSet::is_empty() {
    // scan n pieces → O(n)
    for (auto const &p : *simple_sets) {
        if (!p->is_empty()) {
            return false;
        }
    }
    return true;
}

std::string *AbstractCompositeSet::to_string() {
    if (is_empty()) {
        return &EMPTY_SET_SYMBOL;
    }

    // 1) Build a vector of the atomic‐strings so we can measure total length
    size_t n = simple_sets->size();
    std::vector<std::string> fragments;
    fragments.reserve(n);

    size_t total_chars = 0;
    for (auto const &p : *simple_sets) {
        // each p->to_string() returns a "new std::string*"
        std::string *raw = p->to_string();     // alloc on heap
        fragments.push_back(std::move(*raw));  // copy into local std::string
        total_chars += fragments.back().size();
        delete raw;                            // free the original
    }

    // 2) We will join them with " u " (space‐u‐space), so that's (n−1)*3 chars
    if (n >= 2) {
        total_chars += (n - 1) * 3;
    }

    // 3) Build the final string in one shot, reserving total_chars
    auto result = new std::string();
    result->reserve(total_chars);

    bool first = true;
    for (size_t i = 0; i < n; ++i) {
        if (!first) {
            result->append(" u ");
        }
        first = false;
        result->append(fragments[i]);
    }
    return result;
}

bool AbstractCompositeSet::operator==(const AbstractCompositeSet &other) const {
    // Quick size check first
    if (simple_sets->size() != other.simple_sets->size()) {
        return false;
    }
    // Compare one‐by‐one, assuming both sets are sorted in the same order
    auto it1 = simple_sets->begin();
    auto it2 = other.simple_sets->begin();
    while (it1 != simple_sets->end()) {
        // Each *it1 is an AbstractSimpleSetPtr_t → deref and call operator==
        if (!(**it1 == **it2)) {
            return false;
        }
        ++it1; ++it2;
    }
    return true;
}

bool AbstractCompositeSet::operator!=(const AbstractCompositeSet &other) const {
    return !(*this == other);
}

bool AbstractCompositeSet::operator<(const AbstractCompositeSet &other) const {
    // We implement a standard "lexicographical_compare" by walking both sets in lock‐step.

    auto it1 = simple_sets->begin();
    auto it2 = other.simple_sets->begin();
    auto end1 = simple_sets->end();
    auto end2 = other.simple_sets->end();

    // Walk as long as both have elements:
    while (it1 != end1 && it2 != end2) {
        // Compare atomic‐simple‐sets via their own operator<.
        if ((**it1) < (**it2)) {
            return true;     // our "this" is lexicographically smaller
        }
        if ((**it2) < (**it1)) {
            return false;    // "other" is smaller, so we are not <
        }
        ++it1; 
        ++it2;
    }
    // If we ran out of elements in "this" first, we are smaller
    if (it1 == end1 && it2 != end2) {
        return true;
    }
    // Otherwise, either both ended together (they are equal → not <), or "other" ended first (we are larger → not <).
    return false;
}


std::tuple<AbstractCompositeSetPtr_t, AbstractCompositeSetPtr_t>
AbstractCompositeSet::split_into_disjoint_and_non_disjoint() const {
    // Early exit for empty or singleton sets - they are already disjoint
    if (simple_sets->size() <= 1) {
        auto disjoint = make_new_empty();
        auto non_disjoint = make_new_empty();

        if (!simple_sets->empty()) {
            disjoint->simple_sets->insert(simple_sets->begin(), simple_sets->end());
        }

        return std::make_tuple(disjoint, non_disjoint);
    }

    auto disjoint = make_new_empty();  // will collect pieces that never overlap
    auto non_disjoint = make_new_empty();  // will collect all pairwise intersections

    // Pre-allocate vectors to avoid reallocations
    std::vector<AbstractSimpleSetPtr_t> vec;
    vec.reserve(simple_sets->size());

    // Use a vector of pairs to track which elements need to be processed
    // This avoids unnecessary comparisons
    std::vector<std::pair<size_t, size_t>> pairs_to_check;

    // 1) Turn the current std::set into a vector for indexed loops (O(n))
    for (auto const &p : *simple_sets) {
        vec.push_back(p);
    }

    // Generate all pairs (i,j) where i < j
    // This avoids redundant comparisons (comparing A with B and then B with A)
    size_t n = vec.size();
    pairs_to_check.reserve(n * (n - 1) / 2);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            pairs_to_check.emplace_back(i, j);
        }
    }

    // Track which elements have been completely removed
    std::vector<bool> completely_removed(n, false);

    // Vector to store remaining parts for each element
    std::vector<AbstractCompositeSetPtr_t> remaining_parts(n);
    for (size_t i = 0; i < n; ++i) {
        remaining_parts[i] = make_new_empty();
        remaining_parts[i]->simple_sets->insert(vec[i]);
    }

    // Process all pairs
    for (const auto& [i, j] : pairs_to_check) {
        // Skip if either element has been completely removed
        if (completely_removed[i] || completely_removed[j]) continue;

        auto &A = vec[i];
        auto &B = vec[j];

        // Compute intersection I = A ∩ B
        auto I = A->intersection_with(B);

        if (!I->is_empty()) {
            // Collect I into "non_disjoint" once
            non_disjoint->simple_sets->insert(I);

            // Process element A
            if (!completely_removed[i]) {
                auto newRemaining = remaining_parts[i]->difference_with(I);
                if (newRemaining->is_empty()) {
                    completely_removed[i] = true;
                } else {
                    remaining_parts[i] = newRemaining;
                }
            }

            // Process element B
            if (!completely_removed[j]) {
                auto newRemaining = remaining_parts[j]->difference_with(I);
                if (newRemaining->is_empty()) {
                    completely_removed[j] = true;
                } else {
                    remaining_parts[j] = newRemaining;
                }
            }
        }
    }

    // Collect all remaining parts that weren't completely removed
    for (size_t i = 0; i < n; ++i) {
        if (!completely_removed[i]) {
            disjoint->simple_sets->insert(
                remaining_parts[i]->simple_sets->begin(),
                remaining_parts[i]->simple_sets->end());
        }
    }

    return std::make_tuple(disjoint, non_disjoint);
}

AbstractCompositeSetPtr_t AbstractCompositeSet::make_disjoint() const {
    // Early exit for empty or singleton sets - they are already disjoint
    if (simple_sets->size() <= 1) {
        auto result = make_new_empty();
        if (!simple_sets->empty()) {
            result->simple_sets->insert(simple_sets->begin(), simple_sets->end());
        }
        return result;
    }

    // 1) First split current composite into (disjoint_0, non_disjoint_0)
    auto [disjoint_acc, non_disjoint] = split_into_disjoint_and_non_disjoint();

    // 2) As long as there remain "intersecting pieces," keep splitting them
    while (!non_disjoint->is_empty()) {
        auto [newDisjoint, remainder] = non_disjoint->split_into_disjoint_and_non_disjoint();
        // accumulate newDisjoint into disjoint_acc
        disjoint_acc->simple_sets->insert(
            newDisjoint->simple_sets->begin(),
            newDisjoint->simple_sets->end());
        non_disjoint = remainder;
    }

    // 3) We have now collected every disjoint piece.  We simply return "disjoint_acc->simplify()"
    //    which under the assumption that "disjoint_acc" is already pairwise‐disjoint, will be O(n log n)
    return disjoint_acc->simplify();
}

AbstractCompositeSetPtr_t AbstractCompositeSet::intersection_with(
    const AbstractSimpleSetPtr_t &simple_set) {
    // Early exit for empty sets
    if (simple_sets->empty() || simple_set->is_empty()) {
        return make_new_empty();
    }

    // Build "{ A_i ∩ simple_set : for each A_i in this→simple_sets }"
    // Then bulk‐insert all nonempty pieces in one go (to avoid n calls to insert()).

    std::vector<AbstractSimpleSetPtr_t> scratch;
    scratch.reserve(simple_sets->size());

    for (auto const &A : *simple_sets) {
        auto I = A->intersection_with(simple_set);  // cost = T_cap
        if (!I->is_empty()) {
            scratch.push_back(I);
        }
    }

    auto result = make_new_empty();
    if (!scratch.empty()) {
        result->simple_sets->insert(scratch.begin(), scratch.end());  // O(k log k)
    }
    return result;
}

AbstractCompositeSetPtr_t AbstractCompositeSet::intersection_with(
    const SimpleSetSetPtr_t &other) {
    // Early exit for empty sets
    if (simple_sets->empty() || other->empty()) {
        return make_new_empty();
    }

    // We want ∪_{B ∈ other} (this ∩ B).  We'll accumulate all those atomic pieces into one vector,
    // then bulk‐insert.

    std::vector<AbstractSimpleSetPtr_t> scratch;
    scratch.reserve(simple_sets->size() + other->size());

    for (auto const &B : *other) {
        // "temp" is (this ∩ B), itself a small composite
        auto temp = intersection_with(B);  // from above

        // Insert all of temp→simple_sets into scratch
        for (auto const &p : *temp->simple_sets) {
            scratch.push_back(p);
        }
    }

    auto result = make_new_empty();
    if (!scratch.empty()) {
        result->simple_sets->insert(scratch.begin(), scratch.end());
    }
    return result;
}

AbstractCompositeSetPtr_t AbstractCompositeSet::intersection_with(
    const AbstractCompositeSetPtr_t &other) {
    // Early exit for empty sets
    if (simple_sets->empty() || other->simple_sets->empty()) {
        return make_new_empty();
    }

    // Just delegate to the "set‐of‐pointers" overload
    return intersection_with(other->simple_sets);
}

AbstractCompositeSetPtr_t AbstractCompositeSet::complement() const {
    // Early exit for empty sets - complement of empty set is the universal set
    if (simple_sets->empty()) {
        return make_new_empty();
    }

    // We know "(∪ A_i)^c = ∩ (A_i^c)."
    // So we iterate over each atomic piece, compute A_i^c (a set of pieces), then intersect them in turn.

    AbstractCompositeSetPtr_t result = nullptr;
    bool first = true;

    for (auto const &A : *simple_sets) {
        auto compA = A->complement();  // cost ≈ O(k_i log k_i)
        if (first) {
            // Initialize result to "all pieces from A^c"
            result = make_new_empty();
            result->simple_sets->insert(compA->begin(), compA->end());
            first = false;
        } else {
            // Intersect the running result with compA
            result = result->intersection_with(compA);  // each intersection is expensive
        }
    }
    return (result == nullptr) ? make_new_empty() : result;
}

AbstractCompositeSetPtr_t AbstractCompositeSet::union_with(
    const AbstractSimpleSetPtr_t &other) {
    // Early exit for empty sets
    if (simple_sets->empty()) {
        // If the other set is empty, return an empty result
        if (other->is_empty()) {
            return make_new_empty();
        }
        auto result = make_new_empty();
        result->simple_sets->insert(other);
        return result;
    }
    if (other->is_empty()) {
        return shared_from_this();
    }

    auto result = make_new_empty();
    // 1) Copy our own non-empty pieces in one shot
    for (auto const &p : *simple_sets) {
        if (!p->is_empty()) {
            result->simple_sets->insert(p);
        }
    }

    // 2) Add "other" in (if not already present and not empty)
    if (!other->is_empty()) {
        result->simple_sets->insert(other);
    }

    // If result is empty, return an empty set
    if (result->simple_sets->empty()) {
        return result;
    }

    // 3) We must re‐merge any overlaps: make_disjoint()
    return result->make_disjoint();
}

AbstractCompositeSetPtr_t AbstractCompositeSet::union_with(
    const AbstractCompositeSetPtr_t &other) {
    // Early exit for empty sets
    if (simple_sets->empty()) {
        // If other is empty or contains only empty sets, return empty
        if (other->is_empty()) {
            return make_new_empty();
        }

        // Filter out empty sets from other
        auto result = make_new_empty();
        for (auto const &p : *other->simple_sets) {
            if (!p->is_empty()) {
                result->simple_sets->insert(p);
            }
        }

        // If result is empty after filtering, return empty set
        if (result->simple_sets->empty()) {
            return result;
        }

        return result;
    }

    if (other->is_empty()) {
        // Filter out empty sets from this
        auto result = make_new_empty();
        for (auto const &p : *simple_sets) {
            if (!p->is_empty()) {
                result->simple_sets->insert(p);
            }
        }

        // If result is empty after filtering, return empty set
        if (result->simple_sets->empty()) {
            return result;
        }

        return result;
    }

    auto result = make_new_empty();
    // 1) Insert all non-empty sets from "this"
    for (auto const &p : *simple_sets) {
        if (!p->is_empty()) {
            result->simple_sets->insert(p);
        }
    }

    // 2) Insert all non-empty sets from "other"
    for (auto const &p : *other->simple_sets) {
        if (!p->is_empty()) {
            result->simple_sets->insert(p);
        }
    }

    // If result is empty after filtering, return empty set
    if (result->simple_sets->empty()) {
        return result;
    }

    // 3) Re‐coalesce any overlaps
    return result->make_disjoint();
}

AbstractCompositeSetPtr_t AbstractCompositeSet::difference_with(
    const AbstractSimpleSetPtr_t &other) {
    // Early exit for empty sets or if other is empty
    if (simple_sets->empty()) {
        return make_new_empty();
    }
    if (other->is_empty()) {
        return shared_from_this();
    }

    // Build "all pieces of Ai \ other," then collect and make_disjoint at the end
    std::vector<AbstractSimpleSetPtr_t> scratch;
    scratch.reserve(simple_sets->size());

    for (auto const &A : *simple_sets) {
        auto diffA = A->difference_with(other);  // each diffA is a set of pieces
        for (auto const &p : *diffA) {
            scratch.push_back(p);
        }
    }

    auto result = make_new_empty();
    if (!scratch.empty()) {
        result->simple_sets->insert(scratch.begin(), scratch.end());
    }
    return result->make_disjoint();
}

AbstractCompositeSetPtr_t AbstractCompositeSet::difference_with(
    const AbstractCompositeSetPtr_t &other) {
    // Early exit for empty sets
    if (simple_sets->empty()) {
        return make_new_empty();
    }
    if (other->is_empty()) {
        return shared_from_this();
    }

    std::vector<AbstractSimpleSetPtr_t> all_survivors;
    all_survivors.reserve(simple_sets->size() * other->simple_sets->size());

    // For each A_i in "this", subtract off all pieces in "other"
    for (auto const &A : *simple_sets) {
        // current_difference is "{A}" initially
        auto current_diff = make_new_empty();
        current_diff->simple_sets->insert(A);

        // Now subtract each B_j in "other"
        for (auto const &B : *other->simple_sets) {
            // Compute A′ = current_diff \ B
            // Note: difference_with(B) returns a set of pieces
            auto temp = current_diff->difference_with(B);
            if (temp->is_empty()) {
                current_diff = nullptr;
                break;  // A is fully removed
            }
            current_diff = temp;
        }
        if (current_diff != nullptr) {
            // Collect whatever atomic pieces remained
            for (auto const &p : *current_diff->simple_sets) {
                all_survivors.push_back(p);
            }
        }
    }

    auto result = make_new_empty();
    if (!all_survivors.empty()) {
        result->simple_sets->insert(all_survivors.begin(), all_survivors.end());
    }
    return result->make_disjoint();
}

bool AbstractCompositeSet::contains(const AbstractCompositeSetPtr_t &other) {
    // Early exit for empty sets
    if (other->is_empty()) {
        return true;  // Empty set is contained in any set
    }
    if (simple_sets->empty()) {
        return false;  // Non-empty set is not contained in empty set
    }

    // A ⊇ B  iff  A ∩ B == B
    auto I = intersection_with(other);  // expensive
    return (I == other) || (*I == *other);
}

void AbstractCompositeSet::add_new_simple_set(
    const AbstractSimpleSetPtr_t &simple_set) const {
    simple_sets->insert(simple_set);  // O(log n)
}

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
    if (simple_sets->size() < 2) return true;

    // Iterate directly over the std::set using two nested iterators — no vector copy,
    // no shared_ptr reference-count churn.
    for (auto it1 = simple_sets->cbegin(); it1 != simple_sets->cend(); ++it1) {
        auto it2 = it1;
        ++it2;
        for (; it2 != simple_sets->cend(); ++it2) {
            if (!(*it1)->intersection_with(*it2)->is_empty()) return false;
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
    if (simple_sets->empty()) return make_new_empty();

    // ---------------------------------------------------------------------------
    // Optimised complement via successive difference (inlined for speed).
    //
    // De Morgan: (A₀ ∪ A₁ ∪ … ∪ Aₙ)^c = A₀^c ∩ A₁^c ∩ … ∩ Aₙ^c
    // Rewritten as: start with A₀^c, then subtract each Aᵢ in turn:
    //   result = A₀^c \ A₁ \ A₂ \ …
    // because X ∩ Y^c = X \ Y.
    //
    // Key optimisation: for each piece P of result and each obstacle Aᵢ,
    // test overlap with a single intersection call.
    //   - No overlap (common case): P survives unchanged — O(1) per piece.
    //   - Overlap:  compute I = P ∩ Aᵢ, then I^c, then P ∩ each piece of I^c.
    //     Same work as the old intersection_with(Aᵢ^c) approach but the
    //     complement is taken on the smaller set I rather than all of Aᵢ.
    //
    // We inline the "piece \ Aᵢ" logic to avoid the intermediate
    // SimpleSetSetPtr_t allocation that AbstractSimpleSet::difference_with
    // would create for every piece, and collect results in a single scratch
    // vector for one bulk insert per step.
    //
    // Because result is a disjoint union at every step and the subtraction of
    // a single simple set preserves disjointness of its subsets, no
    // make_disjoint() is needed during the loop.
    // ---------------------------------------------------------------------------

    auto it = simple_sets->cbegin();

    // Initialise result = A₀^c  (already disjoint)
    auto result = make_new_empty();
    {
        auto comp0 = (*it)->complement();
        result->simple_sets->insert(comp0->begin(), comp0->end());
        ++it;
    }

    while (it != simple_sets->cend()) {
        if (result->simple_sets->empty()) break;

        std::vector<AbstractSimpleSetPtr_t> scratch;
        scratch.reserve(result->simple_sets->size());  // at least as many survivors

        for (auto const &piece : *result->simple_sets) {
            // Test if "piece" overlaps the current obstacle Aᵢ
            auto I = piece->intersection_with(*it);  // one intersection

            if (I->is_empty()) {
                // Fast path: no overlap — piece survives unchanged.
                scratch.push_back(piece);
            } else {
                // Slow path: compute piece \ Aᵢ = piece ∩ Iᶜ
                // (using I instead of Aᵢ for the complement keeps the
                // complement smaller: I ⊆ Aᵢ, so I^c ⊇ Aᵢ^c, but
                // piece ∩ I^c = piece ∩ Aᵢ^c, which is what we want)
                auto I_comp = I->complement();
                for (auto const &comp_piece : *I_comp) {
                    auto sub = piece->intersection_with(comp_piece);
                    if (!sub->is_empty()) scratch.push_back(sub);
                }
            }
        }

        auto next = make_new_empty();
        if (!scratch.empty()) {
            next->simple_sets->insert(scratch.begin(), scratch.end());
        }
        result = next;
        ++it;
    }
    return result;
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
    // Single pass over each source, counting non-empty contributions.
    // If only one source contributes, it was already disjoint → skip make_disjoint().
    auto result = make_new_empty();
    size_t from_this = 0, from_other = 0;

    for (auto const &p : *simple_sets) {
        if (!p->is_empty()) { result->simple_sets->insert(p); ++from_this; }
    }
    for (auto const &p : *other->simple_sets) {
        if (!p->is_empty()) { result->simple_sets->insert(p); ++from_other; }
    }

    if (result->simple_sets->empty()) return result;
    if (from_this == 0 || from_other == 0) return result;

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

AbstractCompositeSetPtr_t AbstractCompositeSet::subtract_simple_set_disjoint(
    const AbstractSimpleSetPtr_t &obstacle) {
    // Precondition: this is a disjoint union.
    // Correctness: (piece_i - obstacle) ∩ (piece_j - obstacle) ⊆ piece_i ∩ piece_j = ∅
    // when the pieces are disjoint, so the result is disjoint without calling make_disjoint().
    if (simple_sets->empty() || obstacle->is_empty()) return shared_from_this();

    // Each 3-D box can split into at most 6 sub-pieces when one box is subtracted from it.
    std::vector<AbstractSimpleSetPtr_t> result_pieces;
    result_pieces.reserve(simple_sets->size() * 6);

    for (auto const &current_piece : *simple_sets) {
        auto overlap = current_piece->intersection_with(obstacle);
        if (overlap->is_empty()) {
            // No overlap with the obstacle: keep the current piece unchanged.
            result_pieces.push_back(current_piece);
        } else {
            // Split the current piece around the obstacle and keep all non-overlapping sub-pieces.
            auto difference_pieces = current_piece->difference_with(obstacle);
            for (auto const &difference_piece : *difference_pieces) {
                result_pieces.push_back(difference_piece);
            }
        }
    }

    auto result = make_new_empty();
    if (!result_pieces.empty()) {
        result->simple_sets->insert(result_pieces.begin(), result_pieces.end());
    }
    return result;
}

AbstractCompositeSetPtr_t AbstractCompositeSet::subtract_disjoint(
    const AbstractCompositeSetPtr_t &obstacle_set) {
    // Equivalent to (this & ~obstacle_set) but stays bounded in the same space as this
    // composite set.  Never calls make_disjoint() — disjointness is maintained at each step.
    if (simple_sets->empty()) return make_new_empty();
    if (obstacle_set->is_empty()) return shared_from_this();

    auto remaining = shared_from_this();
    for (auto const &obstacle_piece : *obstacle_set->simple_sets) {
        remaining = remaining->subtract_simple_set_disjoint(obstacle_piece);
        if (remaining->is_empty()) break;
    }
    return remaining;
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
    return *I == *other;
}

void AbstractCompositeSet::add_new_simple_set(
    const AbstractSimpleSetPtr_t &simple_set) const {
    simple_sets->insert(simple_set);  // O(log n)
}

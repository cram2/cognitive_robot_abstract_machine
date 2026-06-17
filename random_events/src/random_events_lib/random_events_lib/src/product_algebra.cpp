#include <stdexcept>
#include <iterator>
#include <vector>
#include <sstream>
#include "product_algebra.h"

//
// ===============================
//  —— SimpleEvent (atomic) ——
// ===============================
//

// Helper: Collect all keys of a map<AbstractVariablePtr_t, ...> into a sorted vector.
//   We use this for merging or iterating without rebuilding a std::set each time.
static std::vector<AbstractVariablePtr_t>
map_keys_to_vector(const VariableMapPtr_t &varmap) {
    std::vector<AbstractVariablePtr_t> keys;
    keys.reserve(varmap->size());
    for (auto const &kv : *varmap) {
        keys.push_back(kv.first);
    }
    // 'varmap' is already in sorted order (PointerLess), so 'keys' is sorted.
    return keys;
}


AbstractSimpleSetPtr_t SimpleEvent::intersection_with(const AbstractSimpleSetPtr_t &other) {
    // We want to build: ∀ v in (vars_self ∪ vars_other), the appropriate assignment intersection.
    //
    // 1) Extract maps and keys
    const auto self_map  = variable_map;  
    const auto other_ptr = static_cast<SimpleEvent *>(other.get());
    const auto other_map = other_ptr->variable_map;

    // Fetch keys in sorted order (no need to build a full std::set for union if we merge two sorted vectors)
    auto self_keys  = map_keys_to_vector(self_map);
    auto other_keys = map_keys_to_vector(other_map);

    // 2) Merge two sorted lists into one union‐vector (two‐pointer sweep)
    std::vector<AbstractVariablePtr_t> all_keys;
    all_keys.reserve(self_keys.size() + other_keys.size());
    size_t i = 0, j = 0;
    while (i < self_keys.size() || j < other_keys.size()) {
        if (i == self_keys.size()) {
            all_keys.push_back(other_keys[j++]);
        } else if (j == other_keys.size()) {
            all_keys.push_back(self_keys[i++]);
        } else if ((*self_keys[i]) < (*other_keys[j])) {
            // note: comparing AbstractVariablePtr_t by PointerLess requires deref; we can trust keys are sorted
            all_keys.push_back(self_keys[i++]);
        } else if ((*other_keys[j]) < (*self_keys[i])) {
            all_keys.push_back(other_keys[j++]);
        } else {
            // equal variable pointer
            all_keys.push_back(self_keys[i]);
            ++i; ++j;
        }
    }

    // 3) Build the resulting SimpleEvent in one shot
    auto result = make_shared_simple_event();
    auto &res_map = result->variable_map;  // alias for convenience

    // 4) For each variable in the merged key‐list, decide which assignment to insert
    for (auto const &var : all_keys) {
        // both present?
        auto it_self  = self_map->find(var);
        auto it_other = other_map->find(var);

        if (it_self != self_map->end() && it_other != other_map->end()) {
            // Present in both: intersect the two composite assignments
            auto inter_assign = it_self->second->intersection_with(it_other->second);
            res_map->insert({var, inter_assign});
        }
        else if (it_self != self_map->end()) {
            // Only in self
            res_map->insert({var, it_self->second});
        }
        else {
            // Only in other
            res_map->insert({var, it_other->second});
        }
    }

    return result;
}

void SimpleEvent::fill_missing_variables(const VariableSetPtr_t &variables) const {
    // For each var in 'variables', if not in variable_map, insert domain
    // We expect 'variables' is a sorted std::set, so iterating is O(|variables|)
    for (auto const &var : *variables) {
        if (variable_map->find(var) == variable_map->end()) {
            variable_map->insert({var, var->get_domain()});
        }
    }
}

VariableSetPtr_t SimpleEvent::get_variables() const {
    // Instead of building a new std::set by inserting keys one by one (O(v log v)),
    // we can construct a set from the map_keys vector via the range‐constructor.
    auto keys = map_keys_to_vector(variable_map);
    return std::make_shared<VariableSet>(keys.begin(), keys.end());
}

VariableSetPtr_t SimpleEvent::merge_variables(const VariableSetPtr_t &other) const {
    // We want vars_self ∪ other in one std::set, but avoid O(v_self log v + v_other log v) re‐inserts.
    // Since 'keys_self' and '*other' are both sorted, we can do a two‐pointer merge directly into a vector,
    // then call the std::set range‐ctor on that vector.

    // 1) Get sorted keys of self
    auto keys_self = map_keys_to_vector(variable_map);

    // 2) 'other' is already a sorted std::set, so we can copy it into a temporary vector
    std::vector<AbstractVariablePtr_t> keys_other;
    keys_other.reserve(other->size());
    for (auto const &v : *other) {
        keys_other.push_back(v);
    }

    // 3) Merge them
    std::vector<AbstractVariablePtr_t> merged;
    merged.reserve(keys_self.size() + keys_other.size());
    size_t i = 0, j = 0;
    while (i < keys_self.size() || j < keys_other.size()) {
        if (i == keys_self.size()) {
            merged.push_back(keys_other[j++]);
        } else if (j == keys_other.size()) {
            merged.push_back(keys_self[i++]);
        } else if ((*keys_self[i]) < (*keys_other[j])) {
            merged.push_back(keys_self[i++]);
        } else if ((*keys_other[j]) < (*keys_self[i])) {
            merged.push_back(keys_other[j++]);
        } else {
            // equal
            merged.push_back(keys_self[i]);
            ++i; ++j;
        }
    }

    // 4) Construct a std::set from that merged vector (range‐constructor)
    return std::make_shared<VariableSet>(merged.begin(), merged.end());
}

SimpleEvent::SimpleEvent(VariableMapPtr_t &variable_map_ptr) {
    variable_map = variable_map_ptr;
}

SimpleSetSetPtr_t SimpleEvent::complement() {
    // We want to generate, for each variable key v_i, a new SimpleEvent in which:
    //   - v_i is assigned 'assignment->complement()'
    //   - every variable processed earlier is assigned value from this->variable_map
    //   - every variable not yet processed is assigned its full domain
    //
    // The original did repeated get_variables() and repeated map lookups.  We will:
    //   1) Collect all keys of variable_map once, in sorted order
    //   2) Iterate that vector with an index 'idx'—so we know which variables are "before" vs "after"
    //   3) Build each current_complement with exactly v variable‐map‐inserts, in O(v log v) per iteration
    //   4) Append to result only if non‐empty
    //
    // This eliminates repeated calls to get_variables() inside the loop.

    auto result = make_shared_simple_set_set();

    // 1) Get all keys and values in variable_map once, sorted (O(v) total)
    auto all_keys = map_keys_to_vector(variable_map);
    size_t vcount = all_keys.size();

    // Pre-extract values so inner loops use O(1) indexed access instead of O(log v) map lookups
    std::vector<AbstractCompositeSetPtr_t> all_values;
    all_values.reserve(vcount);
    for (auto const &k : all_keys) all_values.push_back(variable_map->at(k));

    // 2) For each index i = 0..vcount−1, build a complement event
    for (size_t idx = 0; idx < vcount; ++idx) {
        auto const &var_i    = all_keys[idx];
        auto const &assign_i = all_values[idx];

        // 2a) Build current_complement event
        auto current_complement = make_shared_simple_event();
        auto &cur_map = current_complement->variable_map;

        // 2b) v_i gets assignment->complement()
        auto compl_i = assign_i->complement();
        cur_map->insert({var_i, compl_i});

        // 2c) For every variable before var_i (0..idx−1), assign original (O(1) per lookup)
        for (size_t k = 0; k < idx; ++k) {
            cur_map->insert({ all_keys[k], all_values[k] });
        }
        // 2d) For every variable after var_i (idx+1..vcount−1), assign full domain
        for (size_t k = idx + 1; k < vcount; ++k) {
            cur_map->insert({ all_keys[k], all_keys[k]->get_domain() });
        }

        // 2e) If this new SimpleEvent is not empty, add it to result
        if (!current_complement->is_empty()) {
            result->insert(current_complement);
        }
    }

    return result;
}

bool SimpleEvent::contains(const ElementaryVariant * /*element*/) {
    // Original always returned false. We keep that behavior.
    return false;
}

bool SimpleEvent::is_empty() {
    // If there are no variables, it’s empty
    if (variable_map->empty()) {
        return true;
    }
    // If any assignment in variable_map is empty, return true
    for (auto const &kv : *variable_map) {
        if (kv.second->is_empty()) {
            return true;
        }
    }
    return false;
}

std::string *SimpleEvent::non_empty_to_string() {
    // We want a single heap allocation at the end, so:
    // 1) Gather all “var: assignment” substrings into a vector of std::string
    // 2) Compute total length (including commas, braces)
    // 3) Reserve once, then append everything
    if (is_empty()) {
        // If empty, return “{ }” or however you want to display it; original never handled empty specially,
        // but to preserve semantics, we just produce "{}".
        return new std::string("{}");
    }

    // 1) Gather “var: assignment” for each kv
    std::vector<std::string> fragments;
    fragments.reserve(variable_map->size());

    size_t total_chars = 2;  // for '{' and '}'
    size_t comma_space = (variable_map->size() > 1) ? ((variable_map->size() - 1) * 2) : 0; // ", " between entries
    total_chars += comma_space;

    for (auto const &kv : *variable_map) {
        // format "<var_name>: <assignment_string>"
        //   - var_name is a std::string*, so *kv.first->name is variable name
        std::string varpart = *kv.first->name;
        std::string *raw_assign = kv.second->to_string();
        std::string assignstr = std::move(*raw_assign);
        delete raw_assign;

        // Build "var: assign" local
        std::string combined;
        combined.reserve(varpart.size() + 2 + assignstr.size());
        combined.append(varpart);
        combined.append(": ");
        combined.append(assignstr);

        total_chars += combined.size();
        fragments.push_back(std::move(combined));
    }

    // 2) One final heap allocation
    auto result = new std::string();
    result->reserve(total_chars);
    result->push_back('{');

    bool first = true;
    for (auto const &frag : fragments) {
        if (!first) {
            result->append(", ");
        }
        first = false;
        result->append(frag);
    }
    result->push_back('}');

    return result;
}

bool SimpleEvent::operator==(const AbstractSimpleSet &other) {
    // Compare two SimpleEvents for equality of variable_map
    const auto &rhs = static_cast<const SimpleEvent &>(other);

    // 1) Quick size check
    if (variable_map->size() != rhs.variable_map->size()) {
        return false;
    }

    // 2) Since both maps are ordered by PointerLess, we can walk them in lock‐step
    auto it1 = variable_map->begin();
    auto it2 = rhs.variable_map->begin();
    while (it1 != variable_map->end()) {
        if (*(it1->first) != *(it2->first)) {
            return false; // different variable pointer or name
        }
        // Compare the composite assignments
        if (!(*(it1->second) == *(it2->second))) {
            return false;
        }
        ++it1; ++it2;
    }
    return true;
}

bool SimpleEvent::operator<(const AbstractSimpleSet &other) {
    // Lexicographical compare on (var → assignment) maps
    const auto &rhs = static_cast<const SimpleEvent &>(other);

    auto it1 = variable_map->begin();
    auto it2 = rhs.variable_map->begin();
    auto end1 = variable_map->end();
    auto end2 = rhs.variable_map->end();

    // Walk as long as both have entries
    while (it1 != end1 && it2 != end2) {
        // Compare the variable pointers themselves (PointerLess)
        if (*(it1->first) < *(it2->first)) {
            return true;
        }
        if (*(it2->first) < *(it1->first)) {
            return false;
        }
        // Same variable key, compare the assignments
        if (*(it1->second) < *(it2->second)) {
            return true;
        }
        if (*(it2->second) < *(it1->second)) {
            return false;
        }
        ++it1; ++it2;
    }
    // If we ran out of keys in ‘this’ first, we are smaller
    if (it1 == end1 && it2 != end2) {
        return true;
    }
    // Otherwise, either both ended together (equal → not <) or rhs ended first (we are larger → false)
    return false;
}

SimpleEvent::SimpleEvent(const VariableSetPtr_t &variables) {
    variable_map = std::make_shared<VariableMap>();
    // Build the entire map in one go
    for (auto const &var : *variables) {
        variable_map->insert({var, var->get_domain()});
    }
}

SimpleEvent::SimpleEvent() {
    variable_map = std::make_shared<VariableMap>();
}

AbstractSimpleSetPtr_t SimpleEvent::marginal(const VariableSetPtr_t &variables) const {
    // We return an event restricted to just those variables.  Any variable not in this->variable_map is ignored.
    auto result = make_shared_simple_event();
    auto &res_map = result->variable_map;

    // If ‘variables’ is small compared to variable_map, iterate over ‘variables’:
    for (auto const &var : *variables) {
        auto it = variable_map->find(var);
        if (it != variable_map->end()) {
            res_map->insert({var, it->second});
        }
    }

    return result;
}

//
// ===============================
//  —— Event (composite of SimpleEvent) ——
// ===============================
//

Event::Event() {
    simple_sets = make_shared_simple_set_set();
}

Event::Event(const SimpleSetSetPtr_t &simple_events) {
    simple_sets = simple_events;
    fill_missing_variables();
}

Event::Event(const SimpleEventPtr_t &simple_event) {
    simple_sets = make_shared_simple_set_set();
    simple_sets->insert(simple_event);
    fill_missing_variables();
}

void Event::fill_missing_variables(const VariableSetPtr_t &variable_set) const {
    // For each SimpleEvent in this composite, call its fill_missing_variables
    for (auto const &simple_event : *simple_sets) {
        auto casted = static_cast<SimpleEvent *>(simple_event.get());
        casted->fill_missing_variables(variable_set);
    }
}

void Event::fill_missing_variables() const {
    // 1) Gather all variables from each SimpleEvent in simple_sets, but avoid rebuilding a separate map for each Event.
    // We will collect them into a single VariableSet.
    VariableSet all_vars;
    all_vars.clear();

    for (auto const &simple_event_ptr : *simple_sets) {
        auto casted = static_cast<SimpleEvent *>(simple_event_ptr.get());
        // Instead of calling casted->get_variables() which builds a new std::set, we iterate casted->variable_map directly
        for (auto const &kv : *(casted->variable_map)) {
            all_vars.insert(kv.first);
        }
    }

    // 2) Now call the overload for every SimpleEvent
    auto shared_vars = std::make_shared<VariableSet>(std::move(all_vars));
    for (auto const &simple_event_ptr : *simple_sets) {
        auto casted = static_cast<SimpleEvent *>(simple_event_ptr.get());
        casted->fill_missing_variables(shared_vars);
    }
}

VariableSet Event::get_variables_from_simple_events() const {
    // Instead of calling get_variables() on each SimpleEvent (which builds new sets),
    // we iterate each SimpleEvent's variable_map and insert keys into a local std::set.
    VariableSet result;
    result.clear();

    for (auto const &simple_event_ptr : *simple_sets) {
        auto casted = static_cast<SimpleEvent *>(simple_event_ptr.get());
        for (auto const &kv : *(casted->variable_map)) {
            result.insert(kv.first);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// try_merge_pair: O(v) check whether two SimpleEvents can be merged.
//
// Two SimpleEvents are “merge-compatible” if they differ in exactly one
// variable (their value for that variable may differ; all other variable
// assignments must be equal).  If so, returns the merged SimpleEvent whose
// differing-variable assignment is the union of both values.
// Otherwise returns nullptr.
//
// Assumes fill_missing_variables() has been called so both maps have
// identical key sets — the lockstep walk exploits this for speed.
// ---------------------------------------------------------------------------
static SimpleEventPtr_t try_merge_pair(const SimpleEventPtr_t &A,
                                       const SimpleEventPtr_t &B)
{
    auto &mapA = A->variable_map;
    auto &mapB = B->variable_map;

    auto itA = mapA->cbegin(), endA = mapA->cend();
    auto itB = mapB->cbegin(), endB = mapB->cend();

    size_t mismatch_count = 0;
    AbstractVariablePtr_t mismatch_var = nullptr;
    AbstractCompositeSetPtr_t mismatch_val_A = nullptr;  // A’s value at mismatch
    AbstractCompositeSetPtr_t mismatch_val_B = nullptr;  // B’s value at mismatch

    while (itA != endA && itB != endB) {
        if (*(itA->first) < *(itB->first)) {
            if (++mismatch_count > 1) return nullptr;
            mismatch_var   = itA->first;
            mismatch_val_A = itA->second;
            ++itA;
        } else if (*(itB->first) < *(itA->first)) {
            if (++mismatch_count > 1) return nullptr;
            mismatch_var   = itB->first;
            mismatch_val_B = itB->second;
            ++itB;
        } else {
            if (!(*(itA->second) == *(itB->second))) {
                if (++mismatch_count > 1) return nullptr;
                mismatch_var   = itA->first;
                mismatch_val_A = itA->second;
                mismatch_val_B = itB->second;  // saved here — avoids B->at() later
            }
            ++itA; ++itB;
        }
    }
    while (itA != endA) {
        if (++mismatch_count > 1) return nullptr;
        mismatch_var = itA->first; mismatch_val_A = itA->second;
        ++itA;
    }
    while (itB != endB) {
        if (++mismatch_count > 1) return nullptr;
        mismatch_var = itB->first; mismatch_val_B = itB->second;
        ++itB;
    }

    if (mismatch_count != 1) return nullptr;

    // Compute the union value for the differing variable.
    // If one side is missing (asymmetric key sets), union with the variable’s domain.
    AbstractCompositeSetPtr_t union_val;
    if (mismatch_val_A && mismatch_val_B) {
        union_val = mismatch_val_A->union_with(mismatch_val_B);
    } else if (mismatch_val_A) {
        union_val = mismatch_val_A->union_with(mismatch_var->get_domain());
    } else {
        union_val = mismatch_val_B->union_with(mismatch_var->get_domain());
    }

    // Build merged SimpleEvent: copy mapA entries, replace the mismatch variable.
    auto merged = make_shared_simple_event();
    auto &mmap  = merged->variable_map;
    for (auto const &kv : *mapA) {
        if (*(kv.first) == *mismatch_var) {
            mmap->insert({kv.first, union_val});
        } else {
            mmap->insert({kv.first, kv.second});
        }
    }
    // If the mismatch variable was only in B, add it.
    if (!mismatch_val_A) {
        mmap->insert({mismatch_var, union_val});
    }

    return merged;
}

std::tuple<EventPtr_t, bool> Event::simplify_once() {
    std::vector<SimpleEventPtr_t> vec;
    vec.reserve(simple_sets->size());
    for (auto const &ptr : *simple_sets) {
        vec.push_back(std::static_pointer_cast<SimpleEvent>(ptr));
    }
    size_t n = vec.size();

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            auto merged = try_merge_pair(vec[i], vec[j]);
            if (merged) {
                auto result = make_shared_event();
                result->simple_sets->insert(merged);
                for (size_t k = 0; k < n; ++k) {
                    if (k != i && k != j) result->simple_sets->insert(vec[k]);
                }
                return std::make_tuple(result, true);
            }
        }
    }
    auto self_copy = make_shared_event();
    self_copy->simple_sets = make_shared_simple_set_set(*simple_sets);
    return std::make_tuple(self_copy, false);
}

AbstractCompositeSetPtr_t Event::simplify() {
    // ---------------------------------------------------------------------------
    // O(n²) greedy in-place simplification.
    //
    // The old approach called simplify_once() in a loop: each call scanned all
    // n(n-1)/2 pairs, returned on the FIRST merge, then restarted from (0,0).
    // For k merges this costs O(k × n²) ≈ O(n³).
    //
    // The new approach stays at position i after a successful merge: the newly
    // merged element at i is immediately rechecked against remaining j’s before
    // advancing i.  Each pair is examined at most once per outer while-pass,
    // giving O(n²) total — a factor of O(n) improvement for large events.
    // ---------------------------------------------------------------------------
    std::vector<SimpleEventPtr_t> vec;
    vec.reserve(simple_sets->size());
    for (auto const &ptr : *simple_sets) {
        vec.push_back(std::static_pointer_cast<SimpleEvent>(ptr));
    }

    bool any_merge = true;
    while (any_merge) {
        any_merge = false;
        for (size_t i = 0; i < vec.size(); ++i) {
            for (size_t j = i + 1; j < vec.size(); ) {
                auto merged = try_merge_pair(vec[i], vec[j]);
                if (merged) {
                    vec[i] = merged;           // replace i with merged element
                    vec.erase(vec.begin() + static_cast<std::ptrdiff_t>(j));
                    any_merge = true;
                    // do NOT advance j — recheck new vec[i] against the element
                    // that just slid into position j (and all subsequent ones)
                } else {
                    ++j;
                }
            }
        }
    }

    auto result = make_shared_event();
    for (auto const &se : vec) {
        result->simple_sets->insert(se);
    }
    return result;
}

AbstractCompositeSetPtr_t Event::make_new_empty() const {
    return make_shared_event();
}

AbstractCompositeSetPtr_t Event::marginal(const VariableSetPtr_t &variables) const {
    // Build { E_i.marginal(variables) : for each E_i in simple_sets }, then make_disjoint()
    // Instead of inserting one‐by‐one, we gather them first and do a single bulk‐insert.

    std::vector<AbstractSimpleSetPtr_t> scratch;
    scratch.reserve(simple_sets->size());

    for (auto const &ptr : *simple_sets) {
        auto casted = static_cast<SimpleEvent *>(ptr.get());
        auto marg = casted->marginal(variables);
        scratch.push_back(marg);
    }

    auto result = make_shared_event();
    if (!scratch.empty()) {
        result->simple_sets->insert(scratch.begin(), scratch.end());
    }
    return result->make_disjoint();
}

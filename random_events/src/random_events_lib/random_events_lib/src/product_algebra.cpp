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

    // 1) Get all keys in variable_map once, sorted
    auto all_keys = map_keys_to_vector(variable_map);
    size_t vcount = all_keys.size();

    // 2) For each index i = 0..vcount−1, build a complement event
    for (size_t idx = 0; idx < vcount; ++idx) {
        auto const &var_i = all_keys[idx];
        auto const &assign_i = variable_map->at(var_i);

        // 2a) Build current_complement event
        auto current_complement = make_shared_simple_event();
        auto &cur_map = current_complement->variable_map;

        // 2b) v_i gets assignment->complement()
        auto compl_i = assign_i->complement();
        cur_map->insert({var_i, compl_i});

        // 2c) For every variable before var_i (0..idx−1), assign original
        for (size_t k = 0; k < idx; ++k) {
            cur_map->insert({ all_keys[k], variable_map->at(all_keys[k]) });
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
        std::string varpart = *kv.first->name;        // copy var name
        std::string assignstr = *kv.second->to_string(); // ask composite for its string (heap‐alloc)
        delete kv.second->to_string(); // avoid leaking the returned pointer

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
    auto shared_vars = std::make_shared<VariableSet>(all_vars.begin(), all_vars.end());
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

std::tuple<EventPtr_t, bool> Event::simplify_once() {
    // We want to find any two SimpleEvents that differ in exactly one variable,
    // merge their assignments on that variable, and rebuild the composite.
    //
    // The original made a vector of size n, then used unique_combinations, then on each pair:
    //   - Called get_variables() twice (O(v log v) each)
    //   - Looped over variables, did map lookups, etc.
    //   - If mergeable, built a brand‐new composite in O(n + v) time and returned.
    //   - Otherwise, continued all n(n−1)/2 checks (cost ~O(n^2 (v log v + n))).
    //
    // We optimize by:
    //   1) Building one single vector<AbstractSimpleSetPtr_t> of size n (no change)
    //   2) For each i<j, directly inspect the two ordered maps (they have the same key‐sets because every Event should have fill_missing_variables first)
    //   3) Compare the two maps in one sweep (walk the two variable_map containers in lockstep), counting mismatches.  That costs O(v) instead of O(v log v).  
    //   4) If exactly one mismatch, we build the new merged SimpleEvent by copying variable_map once (O(v log v)), and then unioning the two composite‐sets at that variable (O(T_union)).  
    //   5) Assemble the new Event by inserting n−2 other SimpleEvents in one shot (range‐insert).
    // This reduces each pair’s comparison to O(v) rather than O(v log v), and each merge to O(v log v + (n−2) log n).

    // 1) Copy pointers into a vector
    std::vector<SimpleEventPtr_t> vec;
    vec.reserve(simple_sets->size());
    for (auto const &ptr : *simple_sets) {
        vec.push_back(std::static_pointer_cast<SimpleEvent>(ptr));
    }
    size_t n = vec.size();

    // 2) For i<j pairs:
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            auto &A = vec[i]->variable_map;  // map<AbstractVariablePtr_t, AbstractCompositeSetPtr_t>
            auto &B = vec[j]->variable_map;
            // Both A and B should have identical keys (because fill_missing_variables() was run),
            // so we can walk them in lockstep.  But to be robust, we do a two‐iterator sweep on both maps:
            auto itA = A->begin(), endA = A->end();
            auto itB = B->begin(), endB = B->end();

            size_t mismatch_count = 0;
            AbstractVariablePtr_t mismatch_var = nullptr;

            while (itA != endA && itB != endB) {
                // Compare variable pointers
                if (*(itA->first) < *(itB->first)) {
                    // A has a key not in B → mismatch
                    mismatch_count++;
                    mismatch_var = itA->first;
                    ++itA;
                }
                else if (*(itB->first) < *(itA->first)) {
                    // B has a key not in A → mismatch
                    mismatch_count++;
                    mismatch_var = itB->first;
                    ++itB;
                }
                else {
                    // Same key
                    if (!(*(itA->second) == *(itB->second))) {
                        mismatch_count++;
                        mismatch_var = itA->first;
                    }
                    ++itA; 
                    ++itB;
                }
                if (mismatch_count > 1) {
                    break;
                }
            }
            // If one map has leftover keys, each is a mismatch:
            if (mismatch_count <= 1) {
                while (itA != endA) {
                    mismatch_count++;
                    mismatch_var = itA->first;
                    ++itA;
                    if (mismatch_count > 1) break;
                }
                while (itB != endB && mismatch_count <= 1) {
                    mismatch_count++;
                    mismatch_var = itB->first;
                    ++itB;
                }
            }

            // 3) If exactly one mismatch, we can merge
            if (mismatch_count == 1) {
                // Build a brand‐new SimpleEvent “merged”
                auto merged_event = make_shared_simple_event();
                auto &mmap = merged_event->variable_map;

                // Copy all keys from A (or B) since they now have approximately the same key set.
                // We can just walk A’s map in order.
                for (auto const &kv : *A) {
                    auto var = kv.first;
                    if (var == mismatch_var) {
                        // union A[var] ∪ B[var]
                        auto unioned = kv.second->union_with(B->at(var));
                        mmap->insert({var, unioned});
                    } else {
                        mmap->insert({var, kv.second});
                    }
                }

                // Build the new Event composite:
                auto result = make_shared_event();
                // Insert the merged event first
                result->simple_sets->insert(merged_event);

                // Now insert all others except i,j.  Instead of inserting one‐by‐one (O((n−2) log n)),
                // we gather them in a vector and do a single range‐insert at the end.

                // Prepare a local vector of the other pointers
                std::vector<AbstractSimpleSetPtr_t> to_insert;
                to_insert.reserve(n - 2);
                for (size_t k = 0; k < n; ++k) {
                    if (k == i || k == j) continue;
                    to_insert.push_back(vec[k]);
                }
                result->simple_sets->insert(to_insert.begin(), to_insert.end());

                return std::make_tuple(result, true);
            }
        }
    }

    // No simplification found—return a copy of this Event
    {
        auto self_copy = make_shared_event();
        self_copy->simple_sets = make_shared_simple_set_set(*simple_sets);
        return std::make_tuple(self_copy, false);
    }
}

AbstractCompositeSetPtr_t Event::simplify() {
    auto [current, changed] = simplify_once();
    while (changed) {
        auto [next, next_changed] = current->simplify_once();
        current = next;
        changed = next_changed;
    }
    return current;
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

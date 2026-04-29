#pragma once

#include <iostream>

#include "sigma_algebra.h"
#include <memory>
#include <utility>
#include <limits>


//FORWARD DECLARE
class SimpleInterval;

class Interval;

// TYPEDEFS
using SimpleIntervalPtr_t = std::shared_ptr<SimpleInterval>;

using IntervalPtr_t = std::shared_ptr<Interval>;


/**
 * Enum for border types of simple_sets.
 */
enum class BorderType {
    /**
     * Open indicates that a value is included in the interval.
     */
    CLOSED,

    /**
     * Close indicates that a value is excluded in the interval.
     */
    OPEN
};

/**
 * Logically intersect borders.
 * @param border_1 One of the borders to intersect.
 * @param border_2 The other border to intersect.
 * @return The intersection of the borders.
 */
inline BorderType intersect_borders(const BorderType border_1, const BorderType border_2) {
    return (border_1 == BorderType::OPEN || border_2 == BorderType::OPEN) ? BorderType::OPEN : BorderType::CLOSED;
}

/**
 * Logically t_complement a border.
 * @param border The borders to t_complement.
 * @return The t_complement a border.
 */
inline BorderType invert_border(const BorderType border) {
    return border == BorderType::OPEN ? BorderType::CLOSED : BorderType::OPEN;
}


/**
 * Class that represents a simple interval.
 */
class SimpleInterval : public AbstractSimpleSet {
public:
    /**
     * The lower value.
     */
    double lower = 0;

    /**
     * The upper value.
     */
    double upper = 0;

    /**
     * THe left border type.
     */
    BorderType left = BorderType::OPEN;

    /**
     * The right border type.
     */
    BorderType right = BorderType::OPEN;

    /**
     * Construct an atomic interval.
     */
    explicit SimpleInterval(double lower = 0, double upper = 0, BorderType left = BorderType::OPEN,
                            BorderType right = BorderType::OPEN) {
        this->lower = lower;
        this->upper = upper;
        this->left = left;
        this->right = right;
    }


    AbstractSimpleSetPtr_t intersection_with(const AbstractSimpleSetPtr_t &other) override {
        const auto derived_other = (SimpleInterval *) other.get();

        // get the new lower and upper bounds
        const double new_lower = std::max(lower, derived_other->lower);
        const double new_upper = std::min(upper, derived_other->upper);

        // return the empty interval if the new lower bound is greater than the new upper bound
        if (new_lower > new_upper) {
            return make_shared();
        }

        // initialize the new borders
        BorderType new_left;
        BorderType new_right;

        // if the lower bounds are equal, intersect the borders
        if (lower == derived_other->lower) {
            new_left = intersect_borders(left, derived_other->left);
        } else {
            // else take the border of the interval with the lower bound
            new_left = lower == new_lower ? left : derived_other->left;
        }

        // if the upper bounds are equal, intersect the borders
        if (upper == derived_other->upper) {
            new_right = intersect_borders(right, derived_other->right);
        } else {
            // else take the border of the interval with the upper bound
            new_right = upper == new_upper ? right : derived_other->right;
        }

        return make_shared(new_lower, new_upper, new_left, new_right);
    };

    SimpleSetSetPtr_t complement() override {
        auto resulting_intervals = make_shared_simple_set_set();

        // if the interval is the real line, return an empty set
        if (lower == -std::numeric_limits<double>::infinity() and
            upper == std::numeric_limits<double>::infinity()) {
            return resulting_intervals;
        }

        // if the interval is empty, return the real line
        if (is_empty()) {
            resulting_intervals->insert(
                    SimpleInterval::make_shared(-std::numeric_limits<double>::infinity(),
                                                             std::numeric_limits<double>::infinity(),
                                                             BorderType::OPEN, BorderType::OPEN));
            return resulting_intervals;
        }

        // if the interval has nothing left
        if (upper < std::numeric_limits<double>::infinity()) {
            resulting_intervals->insert(
                    SimpleInterval::make_shared(upper, std::numeric_limits<double>::infinity(),
                                                             invert_border(right), BorderType::OPEN));
        }

        if (lower > -std::numeric_limits<double>::infinity()) {
            resulting_intervals->insert(
                    SimpleInterval::make_shared(-std::numeric_limits<double>::infinity(), lower,
                                                             BorderType::OPEN, invert_border(left)));
        }

        return resulting_intervals;
    };

    bool contains(const ElementaryVariant *element) override {
        return false;
    };

    bool contains(double element) const {
        if (left == BorderType::OPEN and element <= lower) {
            return false;
        }

        if (right == BorderType::OPEN and element >= upper) {
            return false;
        }

        if (left == BorderType::CLOSED and element < lower) {
            return false;
        }

        if (right == BorderType::CLOSED and element > upper) {
            return false;
        }

        return true;
    };

    bool is_empty() override {
        return lower > upper or (lower == upper and (left == BorderType::OPEN or right == BorderType::OPEN));
    };

    /**
     * This method depends on the type of simple set and has to be overloaded.
     *
     * @param other The other simple set.
     * @return True if they are equal.
     */
    bool operator==(const AbstractSimpleSet &other) override {
        auto derived_other = (SimpleInterval *) &other;
        return *this == *derived_other;
    };

    bool operator==(const SimpleInterval &other) {
        return lower == other.lower and upper == other.upper and left == other.left and right == other.right;
    };

    std::string *non_empty_to_string() override {
        const char left_representation = left == BorderType::OPEN ? '(' : '[';
        const char right_representation = right == BorderType::OPEN ? ')' : ']';
        return new std::string(
                left_representation + std::to_string(lower) + ", " + std::to_string(upper) + right_representation);
    };

    bool operator<(const AbstractSimpleSet &other) override {
        const auto derived_other = (SimpleInterval *) &other;
        return *this < *derived_other;
    };

    /**
     * Compare two simple intervals. Simple intervals are ordered by lower bound. If the lower bound is equal, they are
     * ordered by upper bound.
     *
     * Note that border types are ignored in ordering.
     *
     * @param other The other interval
     * @return True if this interval is less than the other interval.
     */
    bool operator<(const SimpleInterval &other) {
        if (lower == other.lower) {
            return upper < other.upper;
        }
        return lower < other.lower;
    };

    template<typename... Args>
    static SimpleIntervalPtr_t make_shared(Args &&... args) {
        return std::make_shared<SimpleInterval>(std::forward<Args>(args)...);
    };

};

/**
 * Hash function for simple intervals.
 */
namespace std {
    template <>
    struct hash<SimpleInterval> {
        size_t operator()(const SimpleInterval &interval) const {
            return std::hash<double>()(interval.lower) ^ std::hash<double>()(interval.upper) ^
                   std::hash<int>()(static_cast<int>(interval.left)) ^ std::hash<int>()(static_cast<int>(interval.right));
        }
    };
}

/**
 * Class that represents a composite interval.
 * An interval is an (automatically simplified) union of simple simple_sets.
 */
class Interval : public AbstractCompositeSet {
public:

    Interval() {
        this->simple_sets = make_shared_simple_set_set();
    };

    explicit Interval(const SimpleSetSetPtr_t &simple_sets_) {
        this->simple_sets = simple_sets_;
    }


    explicit Interval(const SimpleIntervalPtr_t &simple_interval) {
        this->simple_sets = make_shared_simple_set_set();
        this->simple_sets->insert(simple_interval);
    }

    ~Interval() override {
        simple_sets->clear();
    };

    bool operator <(const AbstractCompositeSet &other) {
        const auto derived_other = (Interval *) &other;
        return *this < *derived_other;
    };

    AbstractCompositeSetPtr_t simplify() override {
        auto result = make_shared_simple_set_set();
        bool first_iteration = true;

        for (const auto &current_simple_set: *simple_sets) {
            auto current_simple_interval = std::static_pointer_cast<SimpleInterval>(current_simple_set);

            // if this is the first iteration, just copy the interval
            if (first_iteration) {
                result->insert(current_simple_interval);
                first_iteration = false;
                continue;
            }

            auto last_simple_interval = std::dynamic_pointer_cast<SimpleInterval>(*result->rbegin());

            if (last_simple_interval->upper > current_simple_interval->lower or (
                last_simple_interval->upper == current_simple_interval->lower and not (
                  last_simple_interval->right == BorderType::OPEN and
                  current_simple_interval->left == BorderType::OPEN))) {
                last_simple_interval->upper = current_simple_interval->upper;
                last_simple_interval->right = current_simple_interval->right;
                  } else {
                      result->insert(current_simple_interval);
                  }
        }
        return Interval::make_shared(result);
    };

    AbstractCompositeSetPtr_t make_new_empty() const override {
        return Interval::make_shared();
    };

    double lower() const {
        return std::dynamic_pointer_cast<SimpleInterval>(*simple_sets->begin())->lower;
    };

    double upper() const {
        return std::dynamic_pointer_cast<SimpleInterval>(*simple_sets->rbegin())->upper;
    };

    bool contains(double element) const {
        for (const auto &simple_set: *simple_sets) {
            auto simple_interval = std::static_pointer_cast<SimpleInterval>(simple_set);
            if (simple_interval->contains(element)) {
                return true;
            }
        }
        return false;
    };

    template<typename... Args>
    static std::shared_ptr<Interval> make_shared(Args &&... args) {
        return std::make_shared<Interval>(std::forward<Args>(args)...);
    }

};

inline IntervalPtr_t closed(const double lower, const double upper) {
    auto interval = SimpleInterval::make_shared(lower, upper, BorderType::CLOSED, BorderType::CLOSED);
    auto intervals = make_shared_simple_set_set();
    intervals->insert(interval);
    return Interval::make_shared(intervals);
}

inline IntervalPtr_t open(const double lower, const double upper) {
    auto interval = SimpleInterval::make_shared(lower, upper, BorderType::OPEN, BorderType::OPEN);
    auto intervals = make_shared_simple_set_set();
    intervals->insert(interval);
    return Interval::make_shared(intervals);
}

inline IntervalPtr_t open_closed(const double lower, const double upper) {
    auto interval = SimpleInterval::make_shared(lower, upper, BorderType::OPEN, BorderType::CLOSED);
    auto intervals = make_shared_simple_set_set();
    intervals->insert(interval);
    return Interval::make_shared(intervals);
}

inline IntervalPtr_t closed_open(const double lower, const double upper) {
    auto interval = SimpleInterval::make_shared(lower, upper, BorderType::CLOSED, BorderType::OPEN);
    auto intervals = make_shared_simple_set_set();
    intervals->insert(interval);
    return Interval::make_shared(intervals);
}

inline IntervalPtr_t singleton(const double value) {
    auto interval = SimpleInterval::make_shared(value, value, BorderType::CLOSED, BorderType::CLOSED);
    auto intervals = make_shared_simple_set_set();
    intervals->insert(interval);
    return Interval::make_shared(intervals);
}

inline IntervalPtr_t empty() {
    auto intervals = make_shared_simple_set_set();
    return Interval::make_shared(intervals);
}

inline IntervalPtr_t reals() {
    auto interval = SimpleInterval::make_shared(-std::numeric_limits<double>::infinity(),
                                                             std::numeric_limits<double>::infinity(),
                                                             BorderType::OPEN, BorderType::OPEN);
    auto intervals = make_shared_simple_set_set();
    intervals->insert(interval);
    return Interval::make_shared(intervals);
}

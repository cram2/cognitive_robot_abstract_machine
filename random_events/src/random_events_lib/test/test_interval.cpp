#include "gtest/gtest.h"
#include "interval.h"
#include "sigma_algebra.h"
#include <set>
#include <memory>
#include <iostream>


TEST(AtomicIntervalCreationTestSuite, SimpleInterval) {
    auto interval = SimpleInterval();
    interval.lower = 0.0;
    interval.upper = 1.0;
    interval.left = BorderType::OPEN;
    interval.right = BorderType::CLOSED;
    EXPECT_EQ(interval.lower, 0.0);
    EXPECT_EQ(interval.upper, 1.0);
    EXPECT_EQ(interval.left, BorderType::OPEN);
    EXPECT_EQ(interval.right, BorderType::CLOSED);
}

TEST(AtomicIntervalIntersectionTestSuite, SimpleInterval) {
    auto simple_interval_1 = SimpleInterval::make_shared(0.0, 1.0, BorderType::OPEN, BorderType::CLOSED);
    auto simple_interval_2 =  SimpleInterval::make_shared(0.5, 1.5, BorderType::CLOSED, BorderType::OPEN);
    const auto intersection = simple_interval_1->intersection_with(simple_interval_2);


    const auto intersection_by_hand =  SimpleInterval::make_shared(0.5, 1, BorderType::CLOSED, BorderType::CLOSED);
    EXPECT_TRUE(*intersection_by_hand == *intersection);
}

TEST(AtomicIntervalEmptyIntersectionTestSuite, SimpleInterval) {
    auto interval1 =  SimpleInterval::make_shared(0.0, 1.0, BorderType::OPEN, BorderType::CLOSED);
    auto interval2 =  SimpleInterval::make_shared(2., 3., BorderType::CLOSED, BorderType::OPEN);
    auto intersection = interval1->intersection_with(interval2);
    auto intersection_by_hand =  SimpleInterval::make_shared(0., 0, BorderType::OPEN, BorderType::OPEN);
    EXPECT_TRUE(*intersection == *intersection_by_hand);
    EXPECT_TRUE(intersection->is_empty());
}

// TEST(AtomicIntervalContainsTestSuite, SimpleInterval){
//     auto interval = SimpleInterval{0.0, 1.0, BorderType::CLOSED, BorderType::CLOSED};
//     EXPECT_TRUE(interval.contains(0.5));
//     EXPECT_TRUE(interval.contains(0.0));
//     EXPECT_TRUE(interval.contains(1.0));
//     EXPECT_FALSE(interval.contains(1.5));
//
//     auto interval2 = SimpleInterval{0.0, 1.0, BorderType::OPEN, BorderType::CLOSED};
//     EXPECT_TRUE(interval2.contains(0.5));
//     EXPECT_FALSE(interval2.contains(0.0));
//     EXPECT_TRUE(interval2.contains(1.0));
//     EXPECT_FALSE(interval2.contains(1.5));
// }

TEST(AtomicIntervalInvertTestSuite, SimpleInterval) {
    auto interval =  SimpleInterval::make_shared(0, 1.0, BorderType::OPEN, BorderType::CLOSED);
    auto inverted = interval->complement();
    auto element_1 =  SimpleInterval::make_shared(-std::numeric_limits<double>::infinity(), 0.0, BorderType::OPEN,
                                                 BorderType::CLOSED);
    auto element_2 =  SimpleInterval::make_shared(1.0, std::numeric_limits<double>::infinity(), BorderType::OPEN,
                                                 BorderType::OPEN);
    auto element_3 =  SimpleInterval::make_shared(1.0, std::numeric_limits<double>::infinity(), BorderType::OPEN,
                                                 BorderType::OPEN);
    auto inverted_by_hand = make_shared_simple_set_set();
    inverted_by_hand->insert(element_1);
    inverted_by_hand->insert(element_2);
    inverted_by_hand->insert(element_3);
    EXPECT_TRUE(compare_sets(inverted, inverted_by_hand));
}

TEST(AtomicIntervalIsEmptyTestSuite, SimpleInterval) {
    auto interval = SimpleInterval{0.0, 1.0, BorderType::CLOSED, BorderType::CLOSED};
    EXPECT_FALSE(interval.is_empty());

    auto interval2 = SimpleInterval{0, 0, BorderType::CLOSED, BorderType::CLOSED};
    EXPECT_FALSE(interval2.is_empty());

    auto interval3 = SimpleInterval{1, 1, BorderType::OPEN, BorderType::OPEN};
    EXPECT_TRUE(interval3.is_empty());
}

TEST(AtomicIntervalDifferenceTest, SimpleInterval) {
    auto interval1 = SimpleInterval::make_shared(0, 1.0, BorderType::OPEN, BorderType::CLOSED);
    auto interval2 = SimpleInterval::make_shared(0., 1., BorderType::CLOSED, BorderType::CLOSED);
    auto interval3 = SimpleInterval::make_shared(1., 2., BorderType::OPEN, BorderType::CLOSED);
    auto interval4 = SimpleInterval::make_shared(0., 3., BorderType::OPEN, BorderType::CLOSED);

    auto empty_difference = interval1->difference_with(interval2);
    EXPECT_TRUE(empty_difference->empty());

    auto identity_difference = interval1->difference_with(interval3);
    EXPECT_EQ(identity_difference->size(), 1);
    EXPECT_EQ(*identity_difference->begin(), interval1);

    auto difference_from_middle_element = interval4->difference_with(interval3);
    auto difference_from_middle_element_by_hand = make_shared_simple_set_set();
    difference_from_middle_element_by_hand->insert(
            SimpleInterval::make_shared(0., 1., BorderType::OPEN, BorderType::CLOSED));
    difference_from_middle_element_by_hand->insert(
            SimpleInterval::make_shared(2., 3., BorderType::OPEN, BorderType::CLOSED));
    EXPECT_TRUE(compare_sets(difference_from_middle_element, difference_from_middle_element_by_hand));
}

// TEST(SimplifyIntervalTestSuite, Interval) {
//     auto interval1 = SimpleInterval<>::make_shared(0.0, 1.0, BorderType::OPEN, BorderType::OPEN);
//     auto interval2 = SimpleInterval<>::make_shared(0.5, 1.5, BorderType::OPEN, BorderType::OPEN);
//     auto interval3 = SimpleInterval<>::make_shared(1.5, 2.0, BorderType::OPEN, BorderType::CLOSED);
//     auto interval4 = SimpleInterval<>::make_shared(3.0, 4.0, BorderType::OPEN, BorderType::OPEN);
//
//     auto interval = empty();
//     interval->simple_sets->insert(interval1);
//     interval->simple_sets->insert(interval2);
//     interval->simple_sets->insert(interval3);
//     interval->simple_sets->insert(interval4);
//
//     auto simplified = interval->simplify();
//
//     auto sbh1 = SimpleInterval<>::make_shared(0.0, 2, BorderType::OPEN, BorderType::OPEN);
//     auto sbh2 = SimpleInterval<>::make_shared(3, 4, BorderType::OPEN, BorderType::OPEN);
//
//     auto result_by_hand = Interval<>::make_shared(make_shared_simple_set_set());
//     result_by_hand->simple_sets->insert(sbh1);
//     result_by_hand->simple_sets->insert(sbh2);
//
//
//     EXPECT_TRUE(compare_sets(simplified->simple_sets, result_by_hand->simple_sets));
// }

TEST(SplitIntervalTestSuit, Interval) {
    auto interval1 = SimpleInterval::make_shared(0.0, 1.0, BorderType::CLOSED, BorderType::OPEN);
    auto interval2 = SimpleInterval::make_shared(1.0, 1.5, BorderType::CLOSED, BorderType::OPEN);
    auto interval3 = SimpleInterval::make_shared(1.5, 2.0, BorderType::OPEN, BorderType::CLOSED);
    auto interval4 = SimpleInterval::make_shared(3.0, 5.0, BorderType::CLOSED, BorderType::CLOSED);

    auto inner = make_shared_simple_set_set();
    inner->insert(interval1);
    inner->insert(interval2);
    inner->insert(interval3);
    inner->insert(interval4);

    auto interval = Interval::make_shared(inner);
    auto [disjoint, non_disjoint] = interval->split_into_disjoint_and_non_disjoint();

    EXPECT_TRUE(disjoint->is_disjoint());
}

TEST(IntervalMakeDisjointTestSuite, Interval) {
    auto interval1 = SimpleInterval::make_shared(0.0, 1.0, BorderType::CLOSED, BorderType::CLOSED);
    auto interval2 = SimpleInterval::make_shared(0.5, 1.5, BorderType::CLOSED, BorderType::CLOSED);
    auto interval3 = SimpleInterval::make_shared(1.5, 2.0, BorderType::CLOSED, BorderType::CLOSED);
    auto interval4 = SimpleInterval::make_shared(2.0, 3.0, BorderType::CLOSED, BorderType::CLOSED);
    auto intervals = make_shared_simple_set_set();
    intervals->insert(interval1);
    intervals->insert(interval2);
    intervals->insert(interval3);
    intervals->insert(interval4);

    auto composed_interval = Interval::make_shared(intervals);
    auto disjoint_interval = composed_interval->make_disjoint();
    EXPECT_EQ(disjoint_interval->simple_sets->size(), 1);
    EXPECT_TRUE(disjoint_interval->is_disjoint());
}

TEST(IntervalIntersectionSimple, Interval) {
    auto interval1 = SimpleInterval::make_shared(0.0, 1.0, BorderType::CLOSED, BorderType::CLOSED);
    auto interval2 = SimpleInterval::make_shared(2.0, 3.0, BorderType::CLOSED, BorderType::CLOSED);
    auto interval3 = SimpleInterval::make_shared(0.5, 2.5, BorderType::CLOSED, BorderType::CLOSED);
    auto intervals = make_shared_simple_set_set();
    intervals->insert(interval1);
    intervals->insert(interval2);

    auto interval = Interval::make_shared(intervals);
    auto intersection = interval->intersection_with(interval3);
    EXPECT_TRUE(intersection->is_disjoint());
    EXPECT_EQ(intersection->simple_sets->size(), 2);
}

TEST(IntervalIntersectionInterval, Interval) {
    auto interval1 = SimpleInterval::make_shared(0.0, 1.0, BorderType::CLOSED, BorderType::CLOSED);
    auto interval2 = SimpleInterval::make_shared(2.0, 3.0, BorderType::CLOSED, BorderType::CLOSED);
    auto interval3 = SimpleInterval::make_shared(4, 5, BorderType::CLOSED, BorderType::CLOSED);
    auto intervals1 = make_shared_simple_set_set();
    intervals1->insert(interval1);
    intervals1->insert(interval2);

    auto intervals2 = make_shared_simple_set_set();
    intervals2->insert(interval2);
    intervals2->insert(interval3);
    auto composite_interval1 = Interval::make_shared(intervals1);
    auto composite_interval2 = Interval::make_shared(intervals2);
    auto intersection = composite_interval1->intersection_with(composite_interval2);
    EXPECT_TRUE(intersection->is_disjoint());
    EXPECT_EQ(intersection->simple_sets->size(), 1);
}

TEST(IntervalComplement, Interval) {
    auto interval1 = SimpleInterval::make_shared(0.0, 1.0, BorderType::CLOSED, BorderType::CLOSED);
    auto interval2 = SimpleInterval::make_shared(2.0, 3.0, BorderType::CLOSED, BorderType::CLOSED);
    auto intervals = make_shared_simple_set_set();
    intervals->insert(interval1);
    intervals->insert(interval2);
    auto interval = Interval::make_shared(intervals);
    auto complement = interval->complement();
    EXPECT_EQ(complement->simple_sets->size(), 3);
    EXPECT_TRUE(complement->is_disjoint());
    EXPECT_TRUE(complement->intersection_with(interval)->is_empty());
}

TEST(IntervalUnion, Interval) {
    auto interval1 = SimpleInterval::make_shared(0.0, 1.0, BorderType::CLOSED, BorderType::CLOSED);
    auto interval2 = SimpleInterval::make_shared(0.5, 1.5, BorderType::CLOSED, BorderType::CLOSED);
    auto interval3 = SimpleInterval::make_shared(2.0, 3.0, BorderType::CLOSED, BorderType::CLOSED);
    auto intervals = make_shared_simple_set_set();
    intervals->insert(interval1);
    intervals->insert(interval2);
    auto interval = Interval::make_shared(intervals);
    auto other_intervals = make_shared_simple_set_set();
    other_intervals->insert(interval2);
    other_intervals->insert(interval3);
    auto other_interval = Interval::make_shared(other_intervals);
    auto union_ = interval->union_with(interval2);
    EXPECT_EQ(union_->simple_sets->size(), 1);
    EXPECT_TRUE(union_->is_disjoint());

    auto union_2 = union_->union_with(other_interval);
    EXPECT_EQ(union_2->simple_sets->size(), 2);
    EXPECT_TRUE(union_2->is_disjoint());
}

TEST(IntervalDifference, Interval) {
    auto interval1 = SimpleInterval::make_shared(0.0, 1.0, BorderType::CLOSED, BorderType::CLOSED);
    auto interval2 = SimpleInterval::make_shared(0.5, 1.5, BorderType::CLOSED, BorderType::CLOSED);
    auto interval3 = SimpleInterval::make_shared(2.0, 3.0, BorderType::CLOSED, BorderType::CLOSED);
    auto intervals = make_shared_simple_set_set();
    intervals->insert(interval1);
    intervals->insert(interval2);
    auto interval = Interval::make_shared(intervals);

    auto other_intervals = make_shared_simple_set_set();
    other_intervals->insert(interval2);
    other_intervals->insert(interval3);


    auto other_interval = Interval::make_shared(other_intervals);
    auto difference = other_interval->difference_with(interval1);
    EXPECT_EQ(difference->simple_sets->size(), 2);
    EXPECT_TRUE(difference->is_disjoint());

    auto difference_2 = other_interval->difference_with(difference);
    EXPECT_EQ(difference_2->simple_sets->size(), 1);
    EXPECT_TRUE(difference_2->is_disjoint());
}

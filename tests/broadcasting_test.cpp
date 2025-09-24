#include <gtest/gtest.h>
#include <minidl/detail/broadcasting.h>

using namespace minidl;

TEST(BroadcastShape, WithSameShape) {
    std::vector<std::size_t> a({2, 3});

    auto out = detail::compute_broadcast_shape(a, a);

    EXPECT_EQ(out, a);
}

TEST(BroadcastShape, WithDifferentShape) {
    std::vector<std::size_t> a({1, 3});
    std::vector<std::size_t> b({2, 1});

    std::vector<std::size_t> expected({2, 3});

    auto out = detail::compute_broadcast_shape(a, b);

    EXPECT_EQ(out, expected);
}

TEST(BroadcastShape, WithScalar) {
    std::vector<std::size_t> a({2, 3});
    std::vector<std::size_t> b({});

    std::vector<std::size_t> expected({2, 3});

    auto out = detail::compute_broadcast_shape(a, b);

    EXPECT_EQ(out, expected);
}

TEST(BroadcastShape, LeftPadShapes) {
    std::vector<std::size_t> a({2, 3, 3});
    std::vector<std::size_t> b({3, 3});

    std::vector<std::size_t> expected({2, 3, 3});

    auto out = detail::compute_broadcast_shape(a, b);

    EXPECT_EQ(out, expected);
}

TEST(BroadcastShape, IncompatibleDimsThrows) {
    std::vector<std::size_t> a{2, 3};
    std::vector<std::size_t> b{2};
    EXPECT_THROW(detail::compute_broadcast_shape(a, b), std::runtime_error);
}

TEST(ExpandStrides, SameInOutShape) {
    std::vector<std::size_t> in_shape({2, 3});
    std::vector<std::size_t> out_shape({2, 3});
    std::vector<std::size_t> in_strides({3, 1});

    std::vector<std::size_t> expected({3, 1});
    auto out = detail::expand_strides_for_broadcast(in_shape, in_strides, out_shape);

    EXPECT_EQ(out, expected);
}

TEST(ExpandStrides, BroadcastOnLeadingDim) {
    std::vector<std::size_t> in_shape({1, 3});
    std::vector<std::size_t> out_shape({2, 3});
    std::vector<std::size_t> in_strides({3, 1});

    std::vector<std::size_t> expected({0, 1});
    auto out = detail::expand_strides_for_broadcast(in_shape, in_strides, out_shape);

    EXPECT_EQ(out, expected);
}

TEST(ExpandStrides, BroadcastScalar) {
    std::vector<std::size_t> in_shape;
    std::vector<std::size_t> out_shape({2, 3});
    std::vector<std::size_t> in_strides({});

    std::vector<std::size_t> expected({0, 0});
    auto out = detail::expand_strides_for_broadcast(in_shape, in_strides, out_shape);

    EXPECT_EQ(out, expected);
}

TEST(ExpandStrides, NonContiguosStrides) {
    std::vector<std::size_t> in_shape({2, 3});
    std::vector<std::size_t> out_shape({2, 3});
    std::vector<std::size_t> in_strides({1, 3});

    std::vector<std::size_t> expected({1, 3});
    auto out = detail::expand_strides_for_broadcast(in_shape, in_strides, out_shape);

    EXPECT_EQ(out, expected);
}
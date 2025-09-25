#include <gtest/gtest.h>
#include <minidl/detail/layout.h>

using namespace minidl;

TEST(DefaultStrides, EmptyShape) {
    std::vector<std::size_t> shape;

    auto out = detail::default_strides(shape);
    std::vector<std::size_t> expected({});
    EXPECT_EQ(out, expected);
}

TEST(DefaultStrides, NormalShapes) {
    // one dim
    std::vector<std::size_t> shape1({3});
    auto out1 = detail::default_strides(shape1);
    std::vector<std::size_t> expected1({1});
    EXPECT_EQ(out1, expected1);

    // two dims
    std::vector<std::size_t> shape2({2, 3});
    auto out2 = detail::default_strides(shape2);
    std::vector<std::size_t> expected2({3, 1});
    EXPECT_EQ(out2, expected2);

    // three dims
    std::vector<std::size_t> shape3({2, 3, 4});
    auto out3 = detail::default_strides(shape3);
    std::vector<std::size_t> expected3({12, 4, 1});
    EXPECT_EQ(out3, expected3);
}

TEST(IsContiguous, TrueCases) {
    std::vector<std::size_t> shape1({});
    std::vector<std::size_t> strides1({});
    EXPECT_TRUE(detail::is_contiguous(shape1, strides1));

    std::vector<std::size_t> shape2({6});
    std::vector<std::size_t> strides2({1});
    EXPECT_TRUE(detail::is_contiguous(shape2, strides2));

    std::vector<std::size_t> shape3({2, 3});
    std::vector<std::size_t> strides3({3, 1});
    EXPECT_TRUE(detail::is_contiguous(shape3, strides3));

    std::vector<std::size_t> shape4({2, 1, 4});
    std::vector<std::size_t> strides4({4, 4, 1});
    EXPECT_TRUE(detail::is_contiguous(shape4, strides4));
}

TEST(IsContiguous, FalseCases) {
    std::vector<std::size_t> shape1({2, 3});
    std::vector<std::size_t> strides1({1, 3});
    EXPECT_FALSE(detail::is_contiguous(shape1, strides1));

    std::vector<std::size_t> shape2({2, 1, 4});
    std::vector<std::size_t> strides2({5, 2, 1});
    EXPECT_FALSE(detail::is_contiguous(shape2, strides2));
}
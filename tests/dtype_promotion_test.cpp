#include <gtest/gtest.h>
#include <minidl/detail/dtype_promotion.h>
#include <minidl/dtype.h>

using namespace minidl;

using Param = std::tuple<DType, DType, DType>;

class DTypePromotionTest : public ::testing::TestWithParam<Param> {};

TEST_P(DTypePromotionTest, Works) {
    DType a, b, expected;
    std::tie(a, b, expected) = GetParam();

    auto out = detail::promote_dtype(a, b);

    EXPECT_EQ(out, expected);
}

INSTANTIATE_TEST_SUITE_P(DTypePromotionAll, DTypePromotionTest,
                         ::testing::Values(Param(DType::i32, DType::f32, DType::f32),
                                           Param(DType::f32, DType::i32, DType::f32),
                                           Param(DType::f32, DType::f32, DType::f32),
                                           Param(DType::i32, DType::i32, DType::i32)));
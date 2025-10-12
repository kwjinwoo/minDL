#include <gtest/gtest.h>
#include <minidl/tensor.h>

using namespace minidl;

TEST(ToTest, TestF32toI32) {
    auto a = Tensor::zeros(Shape({2, 3}), DType::f32);
    auto cated_a = a.to(DType::i32);

    EXPECT_EQ(cated_a.dtype(), DType::i32);
}

TEST(ToTest, TestI32toF32) {
    auto a = Tensor::zeros(Shape({2, 3}), DType::i32);
    auto cated_a = a.to(DType::f32);

    EXPECT_EQ(cated_a.dtype(), DType::f32);
}
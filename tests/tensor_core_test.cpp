#include <gtest/gtest.h>
#include <minidl/tensor.h>

using namespace minidl;

TEST(ToTest, TestF32toI32) {
    auto a = Tensor::zeros(Shape({2, 3}), DType::f32);
    a.to(DType::i32);

    EXPECT_EQ(a.dtype(), DType::i32);
}

TEST(ToTest, TestI32toF32) {
    auto a = Tensor::zeros(Shape({2, 3}), DType::i32);
    a.to(DType::f32);

    EXPECT_EQ(a.dtype(), DType::f32);
}
#include <gtest/gtest.h>
#include <minidl/tensor.h>

using namespace minidl;

TEST(ToTest, TestF32toI32) {
    auto t = Tensor::zeros(Shape({4}), DType::f32);
    auto* x = static_cast<float*>(t.data());
    x[0] = 0.0f;
    x[1] = 1.9f;
    x[2] = -1.9f;
    x[3] = 42.0f;

    auto z = t.to(DType::i32);
    EXPECT_EQ(z.dtype(), DType::i32);
    const auto* p = static_cast<const int32_t*>(z.data());

    EXPECT_EQ(p[0], 0);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], -1);
    EXPECT_EQ(p[3], 42);
}

TEST(ToTest, TestI32toF32) {
    auto t = Tensor::zeros(Shape({4}), DType::i32);
    auto* x = static_cast<int32_t*>(t.data());
    x[0] = 0;
    x[1] = 2;
    x[2] = -2;
    x[3] = 42;

    auto z = t.to(DType::f32);
    EXPECT_EQ(z.dtype(), DType::f32);
    const auto* p = static_cast<const float*>(z.data());

    EXPECT_FLOAT_EQ(p[0], 0.0f);
    EXPECT_FLOAT_EQ(p[1], 2.0f);
    EXPECT_FLOAT_EQ(p[2], -2.0f);
    EXPECT_FLOAT_EQ(p[3], 42.0f);
}

TEST(ToTest, TestScalar) {
    auto t = Tensor::zeros(Shape(), DType::f32);
    auto* x = static_cast<float*>(t.data());
    x[0] = -2.9;

    auto z = t.to(DType::i32);
    EXPECT_EQ(z.dtype(), DType::i32);
    const auto* p = static_cast<const int32_t*>(z.data());

    EXPECT_EQ(p[0], -2);
}

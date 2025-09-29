#include <gtest/gtest.h>
#include <minidl/ops.h>
#include <minidl/shape.h>
#include <minidl/tensor.h>

using namespace minidl;

TEST(Add, AddContiguous) {
    auto a = Tensor::ones(Shape({2, 3}), DType::f32);
    auto b = Tensor::ones(Shape({2, 3}), DType::f32);

    auto c = ops::add(a, b);

    const auto* data = static_cast<const float*>(c.data());
    for (std::size_t i = 0; i < c.numel(); i++) {
        EXPECT_FLOAT_EQ(data[i], 2.0f);
    }
}

TEST(Add, AddNonContiguous) {
    auto x = Tensor::ones(Shape({2, 3}), DType::f32);
    auto a = x.transpose({1, 0});
    auto b = Tensor::ones(Shape({3, 2}), DType::f32);

    ASSERT_FALSE(a.is_contiguous());
    auto as = a.strides();

    EXPECT_NE(as[0], as[1]);

    auto c = ops::add(a, b);
    EXPECT_EQ(c.shape().dims(), (std::vector<std::size_t>{3, 2}));

    EXPECT_TRUE(c.is_contiguous());

    const float* data = static_cast<const float*>(c.data());
    for (size_t i = 0; i < c.numel(); ++i) EXPECT_FLOAT_EQ(data[i], 2.0f);
}

TEST(Add, BroadcastScalar) {
    auto a = Tensor::ones(Shape({}), DType::f32);      // scalar 1.0
    auto b = Tensor::ones(Shape({2, 3}), DType::f32);  // 1.0
    auto c = ops::add(a, b);
    EXPECT_EQ(c.shape().dims(), (std::vector<std::size_t>{2, 3}));
    const float* p = static_cast<const float*>(c.data());
    for (size_t i = 0; i < c.numel(); ++i) EXPECT_FLOAT_EQ(p[i], 2.0f);
}

TEST(Add, BroadcastMiddleDim) {
    auto a = Tensor::ones(Shape({2, 1, 3}), DType::f32);
    auto b = Tensor::ones(Shape({1, 4, 1}), DType::f32);
    auto c = ops::add(a, b);
    EXPECT_EQ(c.shape().dims(), (std::vector<std::size_t>{2, 4, 3}));
    const float* p = static_cast<const float*>(c.data());
    for (size_t i = 0; i < c.numel(); ++i) EXPECT_FLOAT_EQ(p[i], 2.0f);
}

TEST(Add, BroadCastAdd) {
    auto a = Tensor::ones(Shape({1, 3}), DType::f32);
    auto b = Tensor::ones(Shape({2, 1}), DType::f32);

    auto c = ops::add(a, b);
    const auto* data = static_cast<const float*>(c.data());
    EXPECT_EQ(c.shape().dims(), std::vector<std::size_t>({2, 3}));
    for (std::size_t i = 0; i < c.numel(); i++) {
        EXPECT_FLOAT_EQ(data[i], 2.0f);
    }
}

TEST(Add, BroadcastIncompatible) {
    auto a = Tensor::ones(Shape({2, 3}), DType::f32);
    auto b = Tensor::ones(Shape({4, 1}), DType::f32);
    EXPECT_THROW({ auto _ = ops::add(a, b); }, std::runtime_error);
}

TEST(Add, ZeroSizeTensor) {
    auto a = Tensor::ones(Shape({0, 3}), DType::f32);
    auto b = Tensor::ones(Shape({0, 3}), DType::f32);
    auto c = ops::add(a, b);
    EXPECT_EQ(c.numel(), 0u);
    EXPECT_EQ(c.nbytes(), 0u);
    EXPECT_TRUE(c.is_contiguous());
}

TEST(Add, InputsAreNotModified) {
    auto a = Tensor::ones(Shape({2, 3}), DType::f32);
    auto before = a.contiguous();
    auto c = ops::add(a, a);

    const float* pa = static_cast<const float*>(a.data());
    const float* pb = static_cast<const float*>(before.data());
    for (size_t i = 0; i < a.numel(); ++i) {
        EXPECT_FLOAT_EQ(pa[i], 1.0f);
        EXPECT_FLOAT_EQ(pb[i], pa[i]);
    }
    const float* pc = static_cast<const float*>(c.data());
    for (size_t i = 0; i < c.numel(); ++i) EXPECT_FLOAT_EQ(pc[i], 2.0f);
}
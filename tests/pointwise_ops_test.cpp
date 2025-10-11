#include <gtest/gtest.h>
#include <minidl/ops.h>
#include <minidl/tensor.h>

using namespace minidl;

TEST(ReluTest, ReluF32) {
    auto a = Tensor::ones(Shape({2, 3}), DType::f32);

    auto* x = static_cast<float*>(a.data());
    std::size_t n = a.numel();
    std::size_t half = n / 2;

    // preparer data
    for (std::size_t i = 0; i < half; i++) {
        x[i] = -1.0f;
    }

    auto out = ops::relu(a);
    auto* z = static_cast<float*>(out.data());

    for (std::size_t i = 0; i < half; i++) {
        EXPECT_FLOAT_EQ(z[i], 0.0f);
    }

    for (std::size_t i = half; i < n; i++) {
        EXPECT_FLOAT_EQ(z[i], x[i]);
    }
}

TEST(ReluTest, ReluI32) {
    auto a = Tensor::ones(Shape({2, 3}), DType::i32);

    auto* x = static_cast<int32_t*>(a.data());
    std::size_t n = a.numel();
    std::size_t half = n / 2;

    // preparer data
    for (std::size_t i = 0; i < half; i++) {
        x[i] = -1.0f;
    }

    auto out = ops::relu(a);
    auto* z = static_cast<int32_t*>(out.data());

    for (std::size_t i = 0; i < half; i++) {
        EXPECT_EQ(z[i], 0);
    }

    for (std::size_t i = half; i < n; i++) {
        EXPECT_EQ(z[i], x[i]);
    }
}

TEST(ReluTest, ReluScalarPositive) {
    auto a = Tensor::zeros(Shape(), DType::f32);
    auto* x = static_cast<float*>(a.data());
    x[0] = 5.0f;

    auto out = ops::relu(a);
    auto* z = static_cast<float*>(out.data());

    EXPECT_FLOAT_EQ(z[0], 5.0f);
}

TEST(ReluTest, ReluScalarNegative) {
    auto a = Tensor::zeros(Shape(), DType::f32);
    auto* x = static_cast<float*>(a.data());
    x[0] = -1.0f;

    auto out = ops::relu(a);
    auto* z = static_cast<float*>(out.data());

    EXPECT_FLOAT_EQ(z[0], 0.0f);
}

TEST(ReluTest, ReluNonContig) {
    auto a = Tensor::zeros(Shape({3, 2}), DType::f32).transpose({1, 0});
    auto b = Tensor::ones(Shape({2, 3}), DType::f32);

    auto c = ops::sub(a, b);

    auto out = ops::relu(c);

    auto* z = static_cast<float*>(out.data());
    for (std::size_t i = 0; i < out.numel(); i++) {
        EXPECT_FLOAT_EQ(z[i], 0.0f);
    }
}

TEST(SigmoidTest, SigmoidF32) {
    auto a = Tensor::ones(Shape({2, 3}), DType::f32);
    const float ev = 1 / (1 + std::exp(-1));
    auto out = ops::sigmoid(a);
    auto* x = static_cast<float*>(out.data());

    for (std::size_t i = 0; i < out.numel(); i++) {
        EXPECT_FLOAT_EQ(x[i], ev);
    }
}

TEST(SigmoidTest, SigmoidScalar) {
    auto a = Tensor::ones(Shape({}), DType::f32);
    const float ev = 1 / (1 + std::exp(-1));
    auto out = ops::sigmoid(a);
    auto* x = static_cast<float*>(out.data());

    for (std::size_t i = 0; i < out.numel(); i++) {
        EXPECT_FLOAT_EQ(x[i], ev);
    }
}

TEST(SigmoidTest, SigmoidNonContig) {
    auto a = Tensor::ones(Shape({2, 3}), DType::f32);
    auto b = a.transpose({1, 0});

    const float ev = 1 / (1 + std::exp(-1));
    auto out = ops::sigmoid(b);
    auto* x = static_cast<float*>(out.data());

    for (std::size_t i = 0; i < out.numel(); i++) {
        EXPECT_FLOAT_EQ(x[i], ev);
    }
}
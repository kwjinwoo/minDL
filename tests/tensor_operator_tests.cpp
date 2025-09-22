#include <gtest/gtest.h>
#include <minidl/tensor.h>

using namespace minidl;

TEST(TensorAddTest, BasicF32Add) {
    Tensor a = Tensor::zeros({2, 3}, DType::f32);
    Tensor b = Tensor::ones({2, 3}, DType::f32);

    Tensor c = a + b;

    EXPECT_EQ(c.shape().dims(), a.shape().dims());
    EXPECT_EQ(c.dtype(), a.dtype());

    auto* data = static_cast<const float*>(c.data());
    for (std::size_t i = 0; i < c.numel(); i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST(TensorAddTest, BasicI32Add) {
    Tensor a = Tensor::ones({3, 2}, DType::i32);
    Tensor b = Tensor::ones({3, 2}, DType::i32);

    Tensor c = a + b;

    EXPECT_EQ(c.shape().dims(), a.shape().dims());
    EXPECT_EQ(c.dtype(), a.dtype());

    auto* data = static_cast<const std::int32_t*>(c.data());
    for (std::size_t i = 0; i < c.numel(); i++) {
        EXPECT_EQ(data[i], 2);
    }
}

TEST(TensoAddTest, BroadCastTest) {
    Tensor a = Tensor::ones({1, 2}, DType::f32);
    Tensor b = Tensor::ones({3, 1}, DType::f32);

    Tensor c = a + b;

    EXPECT_EQ(c.shape().dims()[0], 3);
    EXPECT_EQ(c.shape().dims()[1], 2);

    auto* data = static_cast<const float*>(c.data());
    for (std::size_t i = 0; i < c.numel(); i++) {
        EXPECT_FLOAT_EQ(data[i], 2.0f);
    }
}
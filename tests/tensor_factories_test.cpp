#include <gtest/gtest.h>
#include <minidl/tensor.h>

using namespace minidl;

template <typename T>
void expect_all_equal(const T* ptr, size_t n, T val) {
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(ptr[i], val) << "mismatch at i=" << i;
    }
}

TEST(Zeros, CreateFP32Zeros) {
    auto t = Tensor::zeros({2, 3}, DType::f32);

    EXPECT_EQ(t.dtype(), DType::f32);
    EXPECT_EQ(t.shape().rank(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.strides()[0], 3);
    EXPECT_EQ(t.strides()[1], 1);

    auto* p = static_cast<const float*>(t.data());
    ASSERT_NE(p, nullptr);
    expect_all_equal<float>(p, 6, 0.0f);
}

TEST(Zeros, CreateI32Zeros) {
    auto t = Tensor::zeros({2, 2}, DType::i32);
    EXPECT_EQ(t.dtype(), DType::i32);
    auto* p = static_cast<const int32_t*>(t.data());
    ASSERT_NE(p, nullptr);
    expect_all_equal<int32_t>(p, 4, 0);
}

TEST(Zeros, CreateScalarZeros) {
    auto t = Tensor::zeros(Shape(), DType::f32);

    EXPECT_EQ(t.numel(), 1);
    EXPECT_EQ(t.shape().rank(), 0);
    EXPECT_TRUE(t.is_contiguous());

    auto* data = static_cast<const float*>(t.data());
    EXPECT_FLOAT_EQ(data[0], 0.0f);
}

TEST(Ones, CreateFP32Ones) {
    auto t = Tensor::ones({2, 3}, DType::f32);

    EXPECT_EQ(t.dtype(), DType::f32);
    EXPECT_EQ(t.shape().rank(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.strides()[0], 3);
    EXPECT_EQ(t.strides()[1], 1);

    auto* p = static_cast<const float*>(t.data());
    ASSERT_NE(p, nullptr);
    expect_all_equal<float>(p, 6, 1.0f);
}

TEST(Ones, CreateScalarOnes) {
    auto t = Tensor::ones(Shape(), DType::f32);
    EXPECT_EQ(t.numel(), 1);
    EXPECT_EQ(t.shape().rank(), 0);
    auto* p = static_cast<const float*>(t.data());
    ASSERT_NE(p, nullptr);
    EXPECT_FLOAT_EQ(p[0], 1.0f);
}

TEST(Arange, CreateArange) {
    auto t = Tensor::arange(4, DType::f32);

    EXPECT_EQ(t.dtype(), DType::f32);
    EXPECT_EQ(t.shape().rank(), 1);
    EXPECT_EQ(t.shape()[0], 4);
    EXPECT_EQ(t.numel(), 4);
    EXPECT_EQ(t.strides()[0], 1);

    auto* p = static_cast<const float*>(t.data());
    ASSERT_NE(p, nullptr);

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(p[i], static_cast<float>(i));
    }
}

TEST(Arange, ZeroSize) {
    auto t = Tensor::arange(0, DType::f32);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.shape().rank(), 1);
    EXPECT_EQ(t.shape()[0], 0);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.data(), nullptr);
}

TEST(Arange, I32Arange) {
    auto t = Tensor::arange(5, DType::i32);
    EXPECT_EQ(t.dtype(), DType::i32);
    auto* p = static_cast<const int32_t*>(t.data());
    ASSERT_NE(p, nullptr);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(p[i], i);
}

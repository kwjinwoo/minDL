#include <gtest/gtest.h>
#include <minidl/dtype.h>
#include <minidl/shape.h>
#include <minidl/tensor.h>

using namespace minidl;

template <typename T>
void expect_all_equal(const T* ptr, size_t n, T val) {
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(ptr[i], val) << "mismatch at i=" << i;
    }
}

// ---------- zeros ----------
TEST(TensorFactorys, ZerosCreatesAllZeros) {
    auto t = Tensor::zeros({2, 3}, DType::f32);

    EXPECT_EQ(t.dtype(), DType::f32);
    EXPECT_EQ(t.shape().rank(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    // EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.strides()[0], 3);
    EXPECT_EQ(t.strides()[1], 1);

    auto* p = static_cast<const float*>(t.data());
    ASSERT_NE(p, nullptr);
    expect_all_equal<float>(p, 6, 0.0f);
}

// ---------- ones ----------
TEST(TensorFactorys, OnesCreatesAllOnes) {
    auto t = Tensor::ones({2, 3}, DType::f32);

    EXPECT_EQ(t.dtype(), DType::f32);
    EXPECT_EQ(t.shape().rank(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    // EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.strides()[0], 3);
    EXPECT_EQ(t.strides()[1], 1);

    auto* p = static_cast<const float*>(t.data());
    ASSERT_NE(p, nullptr);
    expect_all_equal<float>(p, 6, 1.0f);
}
#include <gtest/gtest.h>
#include <minidl/dtype.h>
#include <minidl/shape.h>
#include <minidl/tensor.h>

using namespace minidl;

// ---------- view ----------
TEST(ShapeManipulation, ViewCreate) {
    auto t = Tensor::zeros({2, 3}, DType::f32);
    auto v = t.view({3, 2});

    EXPECT_EQ(v.shape()[0], 3);
    EXPECT_EQ(v.shape()[1], 2);
    EXPECT_EQ(v.strides()[0], 2);
    EXPECT_EQ(v.strides()[1], 1);
}

// ---------- transpose ----------
TEST(ShapeManipulation, Permute2D) {
    Tensor a = Tensor::arange(6, DType::i32).view({2, 3});
    auto b = a.transpose({1, 0});
    EXPECT_EQ(b.shape().dims(), (std::vector<std::size_t>{3, 2}));
    EXPECT_EQ(b.strides(), (std::vector<int64_t>{1, 3}));
    EXPECT_FALSE(b.is_contiguous());
}

TEST(TensorTranspose, Permute3D) {
    Tensor x = Tensor::zeros({2, 3, 4}, DType::f32);
    auto y = x.transpose({1, 2, 0});
    EXPECT_EQ(y.shape().dims(), (std::vector<std::size_t>{3, 4, 2}));
    EXPECT_EQ(y.strides().size(), 3);
    EXPECT_FALSE(y.is_contiguous());
}
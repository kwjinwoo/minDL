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
#include <gtest/gtest.h>
#include <minidl/tensor.h>

using namespace minidl;

TEST(View, ViewCreate) {
    auto t = Tensor::zeros({2, 3}, DType::f32);
    auto v = t.view({3, 2});

    EXPECT_EQ(v.shape()[0], 3);
    EXPECT_EQ(v.shape()[1], 2);
    EXPECT_EQ(v.strides()[0], 2);
    EXPECT_EQ(v.strides()[1], 1);
    EXPECT_TRUE(v.is_contiguous());
    EXPECT_EQ(v.data(), t.data());
    EXPECT_EQ(v.numel(), t.numel());
}

TEST(View, ViewScalar) {
    auto t = Tensor::zeros(Shape(), DType::f32);
    auto v = t.view({1});

    EXPECT_EQ(v.shape()[0], 1);
    EXPECT_EQ(v.strides()[0], 1);
    EXPECT_TRUE(v.is_contiguous());
    EXPECT_EQ(v.data(), t.data());
}

TEST(Transpose, Permute2D) {
    Tensor a = Tensor::arange(6, DType::i32).view({2, 3});
    auto b = a.transpose({1, 0});
    EXPECT_EQ(b.shape().dims(), (std::vector<std::size_t>{3, 2}));
    EXPECT_EQ(b.strides(), (std::vector<std::size_t>{1, 3}));
    EXPECT_EQ(b.data(), a.data());
    EXPECT_FALSE(b.is_contiguous());
}

TEST(Transpose, Permute3D) {
    Tensor x = Tensor::zeros({2, 3, 4}, DType::f32);  // strides {12,4,1}
    auto y = x.transpose({1, 2, 0});
    EXPECT_EQ(y.shape().dims(), (std::vector<std::size_t>{3, 4, 2}));
    EXPECT_EQ(y.strides(), (std::vector<std::size_t>{4, 1, 12}));
    EXPECT_EQ(y.data(), x.data());
    EXPECT_FALSE(y.is_contiguous());
}

TEST(Transpose, ThrowOnInvalidPermutation) {
    Tensor a = Tensor::zeros({2, 3, 4}, DType::f32);
    EXPECT_THROW(a.transpose({0, 0, 1}), std::runtime_error);  // duplicate
    EXPECT_THROW(a.transpose({0, 1}), std::runtime_error);     // wrong rank
}

TEST(Reshape, ReshapeContiguous) {
    Tensor a = Tensor::zeros({2, 3}, DType::f32);

    auto b = a.reshape({3, 2});
    EXPECT_EQ(b.shape().dims(), (std::vector<std::size_t>{3, 2}));
    EXPECT_EQ(b.strides(), (std::vector<std::size_t>{2, 1}));
    EXPECT_TRUE(b.is_contiguous());
    EXPECT_EQ(b.data(), a.data());
}

TEST(Reshape, NonContigReshapeCopiesAndPreservesData) {
    Tensor a = Tensor::arange(6, DType::i32).view({3, 2});  // strides {2,1}
    Tensor b = a.transpose({1, 0});                         // shape {2,3}, strides {1,2}, non-contig
    auto c = b.reshape({3, 2});                             // must copy → contig {2,1}
    EXPECT_EQ(c.shape().dims(), (std::vector<std::size_t>{3, 2}));
    EXPECT_EQ(c.strides(), (std::vector<std::size_t>{2, 1}));
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_NE(c.data(), b.data());
    // data check
    std::vector<std::int32_t> expected({0, 2, 4, 1, 3, 5});
    auto* cd = static_cast<const int*>(c.data());
    for (size_t i = 0; i < 6; ++i) EXPECT_EQ(cd[i], expected[i]);
}

TEST(Reshape, ThrowOnNumelMismatch) {
    Tensor a = Tensor::zeros({2, 3}, DType::f32);
    EXPECT_THROW(a.reshape({3, 3}), std::runtime_error);
}

TEST(Contiguous, NoOpOnAlreadyContiguous) {
    Tensor a = Tensor::zeros({4, 5}, DType::f32);
    auto z = a.contiguous();
    EXPECT_TRUE(z.is_contiguous());
    EXPECT_EQ(z.data(), a.data());  // no copy
}
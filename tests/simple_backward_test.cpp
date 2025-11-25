#include <gtest/gtest.h>
#include <minidl/autograd/grad_fn.h>
#include <minidl/ops.h>
#include <minidl/tensor.h>

#include <memory>

using namespace minidl;

TEST(AddBackward, SimpleAddTest) {
    auto a = Tensor::from_scalar(2.0f, nullptr, true);
    auto b = Tensor::from_scalar(3.0f, nullptr, true);

    Tensor c = ops::add(a, b);

    EXPECT_TRUE(c.requires_grad());
    EXPECT_TRUE(c.grad_fn() != nullptr);
    c.backward();

    EXPECT_TRUE(a.grad() != nullptr);
    auto a_grad = a.grad();
    auto a_grad_data = static_cast<const float*>(a_grad->data());
    EXPECT_FLOAT_EQ(a_grad_data[0], 1.0f);

    EXPECT_TRUE(b.grad() != nullptr);
    auto b_grad = b.grad();
    auto b_grad_data = static_cast<const float*>(b_grad->data());
    EXPECT_FLOAT_EQ(b_grad_data[0], 1.0f);
}

TEST(MulBackward, SimpleMulTest) {
    auto a = Tensor::from_scalar(2.0f, nullptr, true);
    auto b = Tensor::from_scalar(3.0f, nullptr, true);

    Tensor c = ops::mul(a, b);

    EXPECT_TRUE(c.requires_grad());
    EXPECT_TRUE(c.grad_fn() != nullptr);
    c.backward();

    EXPECT_TRUE(a.grad() != nullptr);
    auto a_grad = a.grad();
    auto a_grad_data = static_cast<const float*>(a_grad->data());
    EXPECT_FLOAT_EQ(a_grad_data[0], 3.0f);

    EXPECT_TRUE(b.grad() != nullptr);
    auto b_grad = b.grad();
    auto b_grad_data = static_cast<const float*>(b_grad->data());
    EXPECT_FLOAT_EQ(b_grad_data[0], 2.0f);
}

TEST(ChainRule, SimpleChainRuleTest) {
    auto a = Tensor::from_scalar(2.0f, nullptr, true);
    auto b = Tensor::from_scalar(3.0f, nullptr, true);

    Tensor c1 = ops::mul(a, b);
    Tensor c2 = ops::mul(b, b);
    Tensor c = ops::add(c1, c2);
    c.backward();

    EXPECT_TRUE(a.grad() != nullptr);
    auto a_grad = a.grad();
    auto a_grad_data = static_cast<const float*>(a_grad->data());
    EXPECT_FLOAT_EQ(a_grad_data[0], 3.0f);

    EXPECT_TRUE(b.grad() != nullptr);
    auto b_grad = b.grad();
    auto b_grad_data = static_cast<const float*>(b_grad->data());
    EXPECT_FLOAT_EQ(b_grad_data[0], 8.0f);
}

TEST(MatMulBackward, Simple2DMatMulTest) {
    auto a = Tensor::zeros({2, 3}, DType::f32, nullptr, true);
    float* a_data = static_cast<float*>(a.data());
    a_data[0] = 0.17f;
    a_data[1] = 0.46f;
    a_data[2] = 1.45f;
    a_data[3] = -0.54f;
    a_data[4] = -0.02f;
    a_data[5] = -0.59f;

    auto b = Tensor::zeros({3, 2}, DType::f32, nullptr, true);
    float* b_data = static_cast<float*>(b.data());
    b_data[0] = -0.04f;
    b_data[1] = -0.26f;
    b_data[2] = -0.39f;
    b_data[3] = 1.57f;
    b_data[4] = -0.12f;
    b_data[5] = 1.75f;

    Tensor c = ops::matmul(a, b);
    c.backward();

    EXPECT_TRUE(a.grad() != nullptr);
    auto b_t = b.transpose({1, 0});
    auto expected_ga = ops::matmul(Tensor::ones_like(c), b_t);
    auto expected_ga_data = static_cast<float*>(expected_ga.data());
    auto a_grad = a.grad();
    auto a_grad_data = static_cast<float*>(a_grad->data());
    const std::size_t na = a_grad->numel();
    for (std::size_t i = 0; i < na; i++) {
        EXPECT_FLOAT_EQ(a_grad_data[i], expected_ga_data[i]);
    }

    EXPECT_TRUE(b.grad() != nullptr);
    auto a_t = a.transpose({1, 0});
    auto expected_gb = ops::matmul(a_t, Tensor::ones_like(c));
    auto expected_gb_data = static_cast<float*>(expected_gb.data());
    auto b_grad = b.grad();
    auto b_grad_data = static_cast<float*>(b_grad->data());
    const std::size_t nb = b_grad->numel();
    for (std::size_t i = 0; i < nb; i++) {
        EXPECT_FLOAT_EQ(b_grad_data[i], expected_gb_data[i]);
    }
}

TEST(ReluBackward, SimpleReluTest) {
    auto x = Tensor::zeros({2, 3}, DType::f32, nullptr, true);
    auto x_data = static_cast<float*>(x.data());
    x_data[0] = 0.26f;
    x_data[1] = 0.1f;
    x_data[2] = 0.52f;
    x_data[3] = -0.75f;
    x_data[4] = 0.1f;
    x_data[5] = -0.14f;

    Tensor y = ops::relu(x);
    EXPECT_TRUE(y.requires_grad());
    y.backward();

    EXPECT_TRUE(x.grad() != nullptr);
    auto x_grad_data = static_cast<float*>(x.grad()->data());
    EXPECT_FLOAT_EQ(x_grad_data[0], 1.0f);
    EXPECT_FLOAT_EQ(x_grad_data[1], 1.0f);
    EXPECT_FLOAT_EQ(x_grad_data[2], 1.0f);
    EXPECT_FLOAT_EQ(x_grad_data[3], 0.0f);
    EXPECT_FLOAT_EQ(x_grad_data[4], 1.0f);
    EXPECT_FLOAT_EQ(x_grad_data[5], 0.0f);
}
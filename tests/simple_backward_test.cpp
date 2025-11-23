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
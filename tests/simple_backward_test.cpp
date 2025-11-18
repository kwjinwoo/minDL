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

    auto gf = std::make_shared<AddBackward>(&a, &b);
    c.grad_fn() = gf;
    c.grad_fn()->backward(Tensor::ones_like(c));

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

    auto gf = std::make_shared<MulBackward>(&a, &b);

    c.grad_fn() = gf;
    c.grad_fn()->backward(Tensor::ones_like(c));

    EXPECT_TRUE(a.grad() != nullptr);
    auto a_grad = a.grad();
    auto a_grad_data = static_cast<const float*>(a_grad->data());
    EXPECT_FLOAT_EQ(a_grad_data[0], 3.0f);

    EXPECT_TRUE(b.grad() != nullptr);
    auto b_grad = b.grad();
    auto b_grad_data = static_cast<const float*>(b_grad->data());
    EXPECT_FLOAT_EQ(b_grad_data[0], 2.0f);
}
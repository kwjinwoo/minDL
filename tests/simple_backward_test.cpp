#include <gtest/gtest.h>
#include <minidl/autograd/grad_fn.h>
#include <minidl/ops.h>
#include <minidl/tensor.h>

#include <memory>

using namespace minidl;

TEST(AddBackward, SimpleAddTest) {
    auto pa = std::make_shared<Tensor>(Tensor::ones({2, 3}, DType::f32, nullptr, true));
    auto pb = std::make_shared<Tensor>(Tensor::ones({2, 3}, DType::f32, nullptr, true));

    Tensor c = ops::add(*pa, *pb);

    auto gf = std::make_shared<AddBackward>();
    gf->inputs_.push_back(pa);
    gf->inputs_.push_back(pb);

    c.grad_fn() = gf;
    c.grad_fn()->backward(Tensor::ones({2, 3}));

    EXPECT_TRUE(pa->grad() != nullptr);
}

TEST(MulBackward, SimpleMulTest) {
    auto pa = std::make_shared<Tensor>(Tensor::from_scalar(2.0f, nullptr, true));
    auto pb = std::make_shared<Tensor>(Tensor::from_scalar(3.0f, nullptr, true));

    Tensor c = ops::mul(*pa, *pb);

    auto gf = std::make_shared<MulBackward>();
    gf->inputs_.push_back(pa);
    gf->inputs_.push_back(pb);

    c.grad_fn() = gf;
    c.grad_fn()->backward(Tensor::ones_like(*pa));

    EXPECT_TRUE(pa->grad() != nullptr);
    auto a_grad = *pa->grad();
    auto a_grad_data = static_cast<const float*>(a_grad.data());
    EXPECT_FLOAT_EQ(a_grad_data[0], 3.0f);

    EXPECT_TRUE(pb->grad() != nullptr);
    auto b_grad = *pb->grad();
    auto b_grad_data = static_cast<const float*>(b_grad.data());
    EXPECT_FLOAT_EQ(b_grad_data[0], 2.0f);
}
#include <gtest/gtest.h>
#include <minidl/autograd/grad_fn.h>
#include <minidl/ops.h>
#include <minidl/tensor.h>

#include <memory>

using namespace minidl;

TEST(AddBackward, SimpleTest) {
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
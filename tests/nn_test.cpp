#include "minidl/nn.h"

#include "gtest/gtest.h"
#include "minidl/ops.h"
#include "minidl/tensor.h"

using namespace minidl;
using namespace minidl::nn;

TEST(LinearTest, ForwardShapeNoBias) {
    std::size_t batch = 4;
    std::size_t in_features = 3;
    std::size_t out_features = 5;

    Linear linear(in_features, out_features, /*use_bias=*/false);

    Tensor x = Tensor::zeros({batch, in_features}, DType::f32, nullptr, /*requires_grad=*/false);

    Tensor y = linear.forward(x);

    auto y_dims = y.shape().dims();
    ASSERT_EQ(y.rank(), 2u);
    EXPECT_EQ(y_dims[0], batch);
    EXPECT_EQ(y_dims[1], out_features);
}

TEST(LinearTest, ForwardShapeWithBias) {
    std::size_t batch = 2;
    std::size_t in_features = 4;
    std::size_t out_features = 3;

    Linear linear(in_features, out_features, /*use_bias=*/true);
    Tensor x = Tensor::zeros({batch, in_features}, DType::f32, nullptr, false);

    Tensor y = linear.forward(x);

    auto y_dims = y.shape().dims();
    ASSERT_EQ(y.rank(), 2u);
    EXPECT_EQ(y_dims[0], batch);
    EXPECT_EQ(y_dims[1], out_features);
}

TEST(LinearTest, ForwardNumericNoBias) {
    // y = x @ W^T, no bias
    // x: [1, 2], W: [3, 2] (out, in)
    Linear linear(/*in_features=*/2, /*out_features=*/3, /*use_bias=*/false);

    // x = [[1.0, 2.0]]
    Tensor x = Tensor::zeros({1, 2}, DType::f32, nullptr, false);
    float* x_data = static_cast<float*>(x.data());
    x_data[0] = 1.0f;
    x_data[1] = 2.0f;

    // weight_: [3, 2]
    // w0 = [0.1, 0.2]
    // w1 = [0.3, 0.4]
    // w2 = [0.5, 0.6]
    Tensor& w = linear.weight();
    float* w_data = static_cast<float*>(w.data());
    w_data[0] = 0.1f;
    w_data[1] = 0.2f;
    w_data[2] = 0.3f;
    w_data[3] = 0.4f;
    w_data[4] = 0.5f;
    w_data[5] = 0.6f;

    Tensor y = linear.forward(x);
    ASSERT_EQ(y.rank(), 2u);
    auto y_dims = y.shape().dims();
    ASSERT_EQ(y_dims[0], 1u);
    ASSERT_EQ(y_dims[1], 3u);

    const float* y_data = static_cast<const float*>(y.data());
    // y = x @ W^T = [[1,2]] @ [[0.1,0.3,0.5],
    //                           [0.2,0.4,0.6]]
    //   = [1*0.1 + 2*0.2, 1*0.3 + 2*0.4, 1*0.5 + 2*0.6]
    //   = [0.5, 1.1, 1.7]
    EXPECT_NEAR(y_data[0], 0.5f, 1e-6f);
    EXPECT_NEAR(y_data[1], 1.1f, 1e-6f);
    EXPECT_NEAR(y_data[2], 1.7f, 1e-6f);
}

TEST(LinearTest, ForwardNumericWithBias) {
    Linear linear(/*in_features=*/2, /*out_features=*/2, /*use_bias=*/true);

    // x = [[1.0, -1.0]]
    Tensor x = Tensor::zeros({1, 2}, DType::f32, nullptr, false);
    float* x_data = static_cast<float*>(x.data());
    x_data[0] = 1.0f;
    x_data[1] = -1.0f;

    // weight_: [2, 2]
    // w0 = [1.0, 0.0]
    // w1 = [0.0, 1.0]
    Tensor& w = linear.weight();
    float* w_data = static_cast<float*>(w.data());
    w_data[0] = 1.0f;
    w_data[1] = 0.0f;
    w_data[2] = 0.0f;
    w_data[3] = 1.0f;

    // bias_ = [0.5, -0.5]
    Tensor& b = linear.bias();
    float* b_data = static_cast<float*>(b.data());
    b_data[0] = 0.5f;
    b_data[1] = -0.5f;

    Tensor y = linear.forward(x);
    const float* y_data = static_cast<const float*>(y.data());

    // y = x @ W^T + b
    // W^T = [[1,0],
    //        [0,1]]
    // x @ W^T = [1.0, -1.0]
    // y = [1.0 + 0.5, -1.0 - 0.5] = [1.5, -1.5]
    EXPECT_NEAR(y_data[0], 1.5f, 1e-6f);
    EXPECT_NEAR(y_data[1], -1.5f, 1e-6f);
}

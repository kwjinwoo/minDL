#include "minidl/nn.h"

#include "gtest/gtest.h"
#include "minidl/detail/iter.h"
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

TEST(LinearTest, BackwardComputesCorrectGradients) {
    Linear linear(2, 2, true);

    // weight, bias를 고정된 값으로 설정해 deterministic 테스트
    {
        float* w = static_cast<float*>(linear.weight().data());
        // W shape: (2, 2)  (out_features, in_features)
        // W = [[0.2, -0.3],
        //      [0.5,  0.1]]
        w[0] = 0.2f;
        w[1] = -0.3f;
        w[2] = 0.5f;
        w[3] = 0.1f;

        float* b = static_cast<float*>(linear.bias().data());
        b[0] = 0.1f;
        b[1] = -0.2f;
    }

    // 입력도 deterministic
    Tensor x = Tensor::zeros({3, 2}, DType::f32, nullptr, /*requires_grad=*/true);
    {
        float* xd = static_cast<float*>(x.data());
        // x = [[1, 2],
        //      [-1, 3],
        //      [0, -2]]
        xd[0] = 1.f;
        xd[1] = 2.f;
        xd[2] = -1.f;
        xd[3] = 3.f;
        xd[4] = 0.f;
        xd[5] = -2.f;
    }

    Tensor y = linear.forward(x);
    Tensor loss = ops::sum(y);
    loss.backward();

    // ===== Expected gradients =====

    // 1) dL/db = [3, 3]
    float expected_db[2] = {3.f, 3.f};

    // 2) dL/dW[row_j] = sum(x)
    // sum(x) = [ (1 + -1 + 0), (2 + 3 + -2) ] = [0, 3]
    float expected_dW[4] = {
        0.f, 3.f,  // row 0
        0.f, 3.f   // row 1
    };

    // 3) dL/dx = ones(3×2) @ W
    // row-sum(W) = [0.2+(-0.3), 0.5+0.1] = [-0.1, 0.6]
    // x[i] grad = [-0.1, 0.6]
    float expected_dx[6] = {0.7f, -0.2f, 0.7f, -0.2f, 0.7f, -0.2f};

    // ===== Check grads =====

    // bias grad
    ASSERT_TRUE(linear.bias().grad() != nullptr);
    float* b_grad = static_cast<float*>(linear.bias().grad()->data());
    EXPECT_FLOAT_EQ(b_grad[0], expected_db[0]);
    EXPECT_FLOAT_EQ(b_grad[1], expected_db[1]);

    // weight grad
    ASSERT_TRUE(linear.weight().grad() != nullptr);
    float* w_grad = static_cast<float*>(linear.weight().grad()->data());
    detail::NdCounter counter(linear.weight().grad()->shape().dims());
    for (int i = 0; i < 4; i++) {
        auto w_off = detail::offset_elems(counter.idx, linear.weight().grad()->strides());
        EXPECT_FLOAT_EQ(w_grad[w_off], expected_dW[i]);
        counter.next();
    }

    // input grad
    ASSERT_TRUE(x.grad() != nullptr);
    float* x_grad = static_cast<float*>(x.grad()->data());
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(x_grad[i], expected_dx[i]);
    }
}

TEST(LinearInit, WeightIsRandUniform) {
    constexpr std::size_t IN = 16;
    constexpr std::size_t OUT = 32;

    Linear layer(IN, OUT, /*use_bias=*/false);

    const Tensor& w = layer.weight();
    ASSERT_EQ(w.shape().dims(), Shape({OUT, IN}).dims());
    ASSERT_TRUE(w.is_contiguous());
    ASSERT_TRUE(w.requires_grad());

    const float* wd = static_cast<const float*>(w.data());
    const std::size_t n = w.numel();

    // Kaiming uniform bound
    const float bound = std::sqrt(6.0f / IN);

    bool all_zero = true;
    bool all_same = true;
    float first = wd[0];

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_GE(wd[i], -bound);
        EXPECT_LE(wd[i], bound);

        if (wd[i] != 0.0f) {
            all_zero = false;
        }

        if (wd[i] != first) {
            all_same = false;
        }
    }

    EXPECT_FALSE(all_zero);
    EXPECT_FALSE(all_same);
}

struct AddConst : public Module {
    Tensor c_;

    explicit AddConst(float v) : Module("AddConst") {
        c_ = Tensor::ones({1});
        auto* data = static_cast<float*>(c_.data());
        data[0] = v;
    }

    Tensor forward(const Tensor& x) const override { return ops::add(x, c_); }
};

TEST(SequentialTest, ForwardOrder) {
    Sequential seq;

    seq.add<AddConst>("add1", 1.0f);
    seq.add<AddConst>("add2", 2.0f);

    Tensor x = Tensor::zeros({2, 2}, DType::f32);
    Tensor y = seq.forward(x);

    auto* y_data = static_cast<float*>(y.data());

    // y = x + 1 + 2 = 3
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(y_data[i], 3.0f);
    }
}

TEST(SequentialTest, CallOperator) {
    Sequential seq;
    seq.add<AddConst>("add", 5.0f);

    Tensor x = Tensor::zeros({1}, DType::f32);
    Tensor y = seq(x);  // operator()

    auto* y_data = static_cast<float*>(y.data());
    EXPECT_FLOAT_EQ(y_data[0], 5.0f);
}

TEST(SequentialTest, ParametersRecursive) {
    Sequential seq;

    seq.add<Linear>("fc1", 4, 3);
    seq.add<Linear>("fc2", 3, 2);

    auto params = seq.parameters(/*recurse=*/true);

    // Linear 1개당 weight + bias = 2
    EXPECT_EQ(params.size(), 4);

    for (auto* p : params) {
        EXPECT_TRUE(p->requires_grad());
        EXPECT_NE(p->data(), nullptr);
    }
}

TEST(ZeroGradTest, ResetExistingGrad) {
    Sequential seq;
    auto& fc = seq.add<Linear>("fc", 3, 2);

    Tensor x = Tensor::ones({1, 3}, DType::f32);
    Tensor y = fc(x);
    Tensor out = ops::sum(y);
    out.backward();

    auto params = seq.parameters();
    ASSERT_FALSE(params.empty());

    for (auto* p : params) {
        ASSERT_TRUE(p->grad() != nullptr);
    }

    seq.zero_grad();

    for (auto* p : params) {
        auto g = p->grad();
        ASSERT_TRUE(g != nullptr);

        auto* g_data = static_cast<float*>(g->data());
        std::size_t n = g->numel();

        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_FLOAT_EQ(g_data[i], 0.0f);
        }
    }
}

TEST(ZeroGradTest, NoGradDoesNotCrash) {
    Sequential seq;
    seq.add<Linear>("fc", 4, 3);

    auto params = seq.parameters();
    ASSERT_FALSE(params.empty());

    for (auto* p : params) {
        EXPECT_EQ(p->grad(), nullptr);
    }

    EXPECT_NO_THROW({ seq.zero_grad(); });

    for (auto* p : params) {
        EXPECT_EQ(p->grad(), nullptr);
    }
}
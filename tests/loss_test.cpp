
#include <gtest/gtest.h>
#include <minidl/ops.h>
#include <minidl/tensor.h>

#include <cmath>
#include <cstdint>

using namespace minidl;

static void fill_f32(Tensor& t, const std::vector<float>& vals) {
    ASSERT_EQ(t.dtype(), DType::f32);
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < vals.size(); ++i) p[i] = vals[i];
}

static void fill_i32(Tensor& t, const std::vector<int32_t>& vals) {
    ASSERT_EQ(t.dtype(), DType::i32);
    int32_t* p = static_cast<int32_t*>(t.data());
    for (size_t i = 0; i < vals.size(); ++i) p[i] = vals[i];
}

// 1) All-zero logits => uniform softmax => CE = log(C)
TEST(CrossEntropyForward, UniformLogitsGivesLogC) {
    const size_t N = 4;
    const size_t C = 10;

    Tensor logits = Tensor::zeros(Shape({N, C}), DType::f32, nullptr, /*requires_grad=*/false);
    Tensor target = Tensor::zeros(Shape({N, 1}), DType::i32, nullptr, /*requires_grad=*/false);

    fill_i32(target, {0, 1, 2, 3});

    Tensor loss = ops::cross_entropy(logits, target);

    ASSERT_EQ(loss.rank(), 0) << "loss should be scalar (0-d)";
    const float got = *static_cast<const float*>(loss.data());
    const float expected = std::log(static_cast<float>(C));

    EXPECT_NEAR(got, expected, 1e-5f);
}

TEST(CrossEntropyForward, LargeCorrectLogitGivesSmallLoss) {
    const size_t N = 2;
    const size_t C = 3;

    Tensor logits = Tensor::zeros(Shape({N, C}), DType::f32, nullptr, false);
    Tensor target = Tensor::zeros(Shape({N, 1}), DType::i32, nullptr, false);

    // sample0 target=0, logits=[20,0,0]
    // sample1 target=2, logits=[0,0,20]
    fill_f32(logits, {20.0f, 0.0f, 0.0f, 0.0f, 0.0f, 20.0f});
    fill_i32(target, {0, 2});

    Tensor loss = ops::cross_entropy(logits, target);
    const float got = *static_cast<const float*>(loss.data());

    // exp(-20) 수준이므로 매우 작아야 함
    EXPECT_LT(got, 1e-6f);
    EXPECT_GE(got, 0.0f);
}

TEST(CrossEntropyForward, MatchesManualComputationSmallCase) {
    const size_t N = 2;
    const size_t C = 2;

    Tensor logits = Tensor::zeros(Shape({N, C}), DType::f32, nullptr, false);
    Tensor target = Tensor::zeros(Shape({N, 1}), DType::i32, nullptr, false);

    // sample0 logits=[1, 2], target=1
    // sample1 logits=[-1, 0.5], target=0
    fill_f32(logits, {1.0f, 2.0f, -1.0f, 0.5f});
    fill_i32(target, {1, 0});

    // manual: CE_i = -log( exp(x_t) / sum_j exp(x_j) )
    auto ce_one = [](float a, float b, int t) {
        const float m = std::max(a, b);
        const float ea = std::exp(a - m);
        const float eb = std::exp(b - m);
        const float denom = ea + eb;
        const float xt = (t == 0) ? (a - m) : (b - m);  // shifted target logit
        return -(xt - std::log(denom));                 // -log_softmax
    };

    const float ce0 = ce_one(1.0f, 2.0f, /*t=*/1);
    const float ce1 = ce_one(-1.0f, 0.5f, /*t=*/0);
    const float expected = (ce0 + ce1) / static_cast<float>(N);

    Tensor loss = ops::cross_entropy(logits, target);
    const float got = *static_cast<const float*>(loss.data());

    EXPECT_NEAR(got, expected, 1e-5f);
}
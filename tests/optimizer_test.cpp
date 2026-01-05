#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "minidl/detail/iter.h"
#include "minidl/nn.h"
#include "minidl/ops.h"
#include "minidl/optim.h"
#include "minidl/tensor.h"

using namespace minidl;

static float scalar_f32(const Tensor& t) {
    if (t.dtype() != DType::f32) throw std::runtime_error("scalar_f32 expects f32");
    if (t.numel() != 1) throw std::runtime_error("scalar_f32 expects numel()==1");
    return static_cast<const float*>(t.data())[0];
}

static float l1_abs_sum_f32_anystride(const Tensor& t) {
    if (t.dtype() != DType::f32) throw std::runtime_error("l1_abs_sum expects f32");

    const float* data = static_cast<const float*>(t.data());
    const auto& shape = t.shape().dims();
    const auto strides = t.strides();

    detail::NdCounter counter(shape);
    float s = 0.0f;
    for (; !counter.done(); counter.next()) {
        const auto& idx = counter.idx;
        const auto off = detail::offset_elems(idx, strides);
        s += std::fabs(data[off]);
    }
    return s;
}

static std::vector<float> snapshot_f32_contiguous(const Tensor& t) {
    if (t.dtype() != DType::f32) throw std::runtime_error("snapshot expects f32");
    if (!t.is_contiguous()) throw std::runtime_error("snapshot expects contiguous");
    std::vector<float> buf(t.numel());
    std::memcpy(buf.data(), t.data(), sizeof(float) * t.numel());
    return buf;
}

static bool changed_from_snapshot_f32_contiguous(const Tensor& t, const std::vector<float>& before) {
    if (t.dtype() != DType::f32) throw std::runtime_error("changed check expects f32");
    if (!t.is_contiguous()) throw std::runtime_error("changed check expects contiguous");
    if (t.numel() != before.size()) throw std::runtime_error("size mismatch");

    const float* now = static_cast<const float*>(t.data());
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (now[i] != before[i]) return true;
    }
    return false;
}

static bool all_zero_f32_contiguous(const Tensor& t) {
    if (t.dtype() != DType::f32) throw std::runtime_error("all_zero expects f32");
    if (!t.is_contiguous()) throw std::runtime_error("all_zero expects contiguous");
    const float* p = static_cast<const float*>(t.data());
    for (std::size_t i = 0; i < t.numel(); ++i) {
        if (p[i] != 0.0f) return false;
    }
    return true;
}

TEST(SGDOptimizer, OneLayerMatmulCrossEntropyUpdatesParams) {
    // ----- 1) toy classification -----
    constexpr std::size_t B = 4;    // batch
    constexpr std::size_t IN = 5;   // in_features
    constexpr std::size_t OUT = 3;  // num_classes

    Tensor x = Tensor::zeros({B, IN}, DType::f32, nullptr, /*requires_grad=*/false);
    ASSERT_TRUE(x.is_contiguous());
    {
        float* xd = static_cast<float*>(x.data());
        for (std::size_t i = 0; i < x.numel(); ++i) {
            // deterministic input
            xd[i] = static_cast<float>((int(i % 7) - 3)) * 0.1f;
        }
    }

    // y: class index
    Tensor y = Tensor::zeros({B, 1}, DType::i32, nullptr, /*requires_grad=*/false);
    {
        int32_t* yd = static_cast<int32_t*>(y.data());
        yd[0] = 0;
        yd[1] = 2;
        yd[2] = 1;
        yd[3] = 2;
    }

    // ----- 2) 1-layer module -----
    nn::Linear fc(IN, OUT, /*use_bias=*/true);

    SGD optim(/*lr=*/0.1f, fc.parameters());

    auto w_before = snapshot_f32_contiguous(fc.weight());
    std::vector<float> b_before;
    if (fc.use_bias()) b_before = snapshot_f32_contiguous(fc.bias());

    // ----- 3) forward -> cross entropy -----
    Tensor logits = fc.forward(x);
    // logits: [B, OUT]
    Tensor loss = ops::cross_entropy(logits, y);

    // ----- 4) backward -----
    loss.backward();

    ASSERT_TRUE(fc.weight().grad() != nullptr);
    EXPECT_EQ(fc.weight().grad()->dtype(), DType::f32);

    if (fc.use_bias()) {
        ASSERT_TRUE(fc.bias().grad() != nullptr);
        EXPECT_EQ(fc.bias().grad()->dtype(), DType::f32);
    }

    optim.step();

    EXPECT_TRUE(changed_from_snapshot_f32_contiguous(fc.weight(), w_before))
        << "weight should be updated after SGD.step()";

    if (fc.use_bias()) {
        EXPECT_TRUE(changed_from_snapshot_f32_contiguous(fc.bias(), b_before))
            << "bias should be updated after SGD.step()";
    }

    optim.zero_grad();

    if (fc.weight().grad()) {
        EXPECT_TRUE(all_zero_f32_contiguous(*fc.weight().grad())) << "weight.grad should be zero after zero_grad()";
    }
    if (fc.use_bias() && fc.bias().grad()) {
        EXPECT_TRUE(all_zero_f32_contiguous(*fc.bias().grad())) << "bias.grad should be zero after zero_grad()";
    }
}

TEST(SGDOptimizer, OneStepShouldNotIncreaseCrossEntropyLoss) {
    constexpr std::size_t B = 4;
    constexpr std::size_t IN = 5;
    constexpr std::size_t OUT = 3;

    Tensor x = Tensor::zeros({B, IN}, DType::f32, nullptr, /*requires_grad=*/false);
    {
        float* xd = static_cast<float*>(x.data());
        for (std::size_t i = 0; i < x.numel(); ++i) {
            xd[i] = static_cast<float>((int(i % 7) - 3)) * 0.1f;
        }
    }

    Tensor y = Tensor::zeros({B, 1}, DType::i32, nullptr, /*requires_grad=*/false);
    {
        int32_t* yd = static_cast<int32_t*>(y.data());
        yd[0] = 0;
        yd[1] = 2;
        yd[2] = 1;
        yd[3] = 2;
    }

    nn::Linear fc(IN, OUT, /*use_bias=*/true);
    SGD optim(/*lr=*/0.1f, fc.parameters());

    Tensor logits1 = fc.forward(x);
    Tensor loss1 = ops::cross_entropy(logits1, y);
    float l1 = scalar_f32(loss1);

    // backward + update
    optim.zero_grad();
    loss1.backward();
    optim.step();

    Tensor logits2 = fc.forward(x);
    Tensor loss2 = ops::cross_entropy(logits2, y);
    float l2 = scalar_f32(loss2);

    EXPECT_LE(l2, l1 + 1e-5f) << "CrossEntropy loss should not increase after one SGD step.";
}

TEST(GradAccumulation, BackwardTwiceAccumulatesGradsWithoutZeroGrad) {
    constexpr std::size_t B = 4;
    constexpr std::size_t IN = 5;
    constexpr std::size_t OUT = 3;

    Tensor x = Tensor::zeros({B, IN}, DType::f32, nullptr, /*requires_grad=*/false);
    {
        float* xd = static_cast<float*>(x.data());
        for (std::size_t i = 0; i < x.numel(); ++i) {
            xd[i] = static_cast<float>((int(i % 7) - 3)) * 0.1f;
        }
    }

    Tensor y = Tensor::zeros({B, 1}, DType::i32, nullptr, /*requires_grad=*/false);
    {
        int32_t* yd = static_cast<int32_t*>(y.data());
        yd[0] = 0;
        yd[1] = 2;
        yd[2] = 1;
        yd[3] = 2;
    }

    nn::Linear fc(IN, OUT, /*use_bias=*/true);

    Tensor logits1 = fc.forward(x);
    Tensor loss1 = ops::cross_entropy(logits1, y);
    loss1.backward();

    ASSERT_TRUE(fc.weight().grad() != nullptr);
    float g1 = l1_abs_sum_f32_anystride(*fc.weight().grad());
    EXPECT_GT(g1, 0.0f) << "First backward should produce non-zero grad.";

    Tensor logits2 = fc.forward(x);
    Tensor loss2 = ops::cross_entropy(logits2, y);
    loss2.backward();

    ASSERT_TRUE(fc.weight().grad() != nullptr);
    float g2 = l1_abs_sum_f32_anystride(*fc.weight().grad());

    EXPECT_GT(g2, g1 + 1e-6f) << "Grad should accumulate when calling backward twice without zero_grad().";

    // bias도 보고 싶으면 추가(선택)
    if (fc.use_bias()) {
        ASSERT_TRUE(fc.bias().grad() != nullptr);
        float b1 = l1_abs_sum_f32_anystride(*fc.bias().grad());
        EXPECT_GT(b1, 0.0f);
    }
}

#include <gtest/gtest.h>
#include <minidl/ops.h>
#include <minidl/shape.h>
#include <minidl/tensor.h>

using namespace minidl;

static inline void expect_all_eq_f32(const Tensor& t, float v) {
    const auto* p = static_cast<const float*>(t.data());
    for (std::size_t i = 0; i < t.numel(); ++i) EXPECT_FLOAT_EQ(p[i], v);
}
static inline void expect_all_eq_i32(const Tensor& t, std::int32_t v) {
    const auto* p = static_cast<const std::int32_t*>(t.data());
    for (std::size_t i = 0; i < t.numel(); ++i) EXPECT_EQ(p[i], v);
}

static inline Tensor ones_like_shape(const std::vector<std::size_t>& dims, DType dt) {
    return Tensor::ones(Shape(dims), dt);
}

using OpFn = Tensor (*)(const Tensor&, const Tensor&);
static Tensor op_add(const Tensor& a, const Tensor& b) { return ops::add(a, b); }
static Tensor op_mul(const Tensor& a, const Tensor& b) { return ops::mul(a, b); }

struct Scenario {
    const char* name;
    std::function<std::pair<Tensor, Tensor>(DType)> make;
    std::vector<std::size_t> expected_shape;
    // (add → 2, mul → 1)
    float expected_scalar_f32_add = 2.0f;
    float expected_scalar_f32_mul = 1.0f;
    int32_t expected_scalar_i32_add = 2;
    int32_t expected_scalar_i32_mul = 1;
};

static const Scenario SCENARIOS[] = {
    {"ContiguousSameShape",
     [](DType dt) {
         Tensor a = ones_like_shape({2, 3}, dt);
         Tensor b = ones_like_shape({2, 3}, dt);
         return std::make_pair(a, b);
     },
     {2, 3}},
    {"NonContiguousTranspose",
     [](DType dt) {
         Tensor x = ones_like_shape({2, 3}, dt);
         Tensor a = x.transpose({1, 0});          // 3x2 non-contiguous
         Tensor b = ones_like_shape({3, 2}, dt);  // 3x2
         return std::make_pair(a, b);
     },
     {3, 2}},
    {"BroadcastScalarLeft",
     [](DType dt) {
         Tensor a = Tensor::ones(Shape({}), dt);  // scalar
         Tensor b = ones_like_shape({2, 3}, dt);
         return std::make_pair(a, b);
     },
     {2, 3}},
    {"BroadcastMiddleDim",
     [](DType dt) {
         Tensor a = ones_like_shape({2, 1, 3}, dt);
         Tensor b = ones_like_shape({1, 4, 1}, dt);
         return std::make_pair(a, b);
     },
     {2, 4, 3}},
    {"BroadcastRowCol",
     [](DType dt) {
         Tensor a = ones_like_shape({1, 3}, dt);
         Tensor b = ones_like_shape({2, 1}, dt);
         return std::make_pair(a, b);
     },
     {2, 3}},
    {"ZeroSize",
     [](DType dt) {
         Tensor a = ones_like_shape({0, 3}, dt);
         Tensor b = ones_like_shape({0, 3}, dt);
         return std::make_pair(a, b);
     },
     {0, 3}},
};

using Param = std::tuple<OpFn, const char*, DType, int>;

class PointwiseBinaryTest : public ::testing::TestWithParam<Param> {};

TEST_P(PointwiseBinaryTest, Works) {
    OpFn op;
    const char* opname;
    DType dt;
    int idx;
    std::tie(op, opname, dt, idx) = GetParam();

    const Scenario& sc = SCENARIOS[idx];
    auto ab = sc.make(dt);
    const Tensor& a = ab.first;
    const Tensor& b = ab.second;

    auto a_before = a.contiguous();

    Tensor c = op(a, b);

    EXPECT_EQ(c.shape().dims(), sc.expected_shape);
    EXPECT_TRUE(c.is_contiguous());

    if (dt == DType::f32) {
        float ev = (opname == std::string("add")) ? sc.expected_scalar_f32_add : sc.expected_scalar_f32_mul;
        expect_all_eq_f32(c, ev);

        const float* pa = static_cast<const float*>(a.data());
        const float* pb = static_cast<const float*>(a_before.data());
        for (std::size_t i = 0; i < std::min<std::size_t>(a.numel(), 6); ++i) {
            EXPECT_FLOAT_EQ(pa[i], pb[i]);
        }
    } else if (dt == DType::i32) {
        int32_t ev = (opname == std::string("add")) ? sc.expected_scalar_i32_add : sc.expected_scalar_i32_mul;
        expect_all_eq_i32(c, ev);
    } else {
        FAIL() << "Unsupported dtype in param test";
    }
}

INSTANTIATE_TEST_SUITE_P(AddAndMul_All, PointwiseBinaryTest,
                         ::testing::Values(
                             // add, f32
                             Param{&op_add, "add", DType::f32, 0}, Param{&op_add, "add", DType::f32, 1},
                             Param{&op_add, "add", DType::f32, 2}, Param{&op_add, "add", DType::f32, 3},
                             Param{&op_add, "add", DType::f32, 4}, Param{&op_add, "add", DType::f32, 5},
                             // add, i32
                             Param{&op_add, "add", DType::i32, 0}, Param{&op_add, "add", DType::i32, 1},
                             Param{&op_add, "add", DType::i32, 2}, Param{&op_add, "add", DType::i32, 3},
                             Param{&op_add, "add", DType::i32, 4}, Param{&op_add, "add", DType::i32, 5},
                             // mul, f32
                             Param{&op_mul, "mul", DType::f32, 0}, Param{&op_mul, "mul", DType::f32, 1},
                             Param{&op_mul, "mul", DType::f32, 2}, Param{&op_mul, "mul", DType::f32, 3},
                             Param{&op_mul, "mul", DType::f32, 4}, Param{&op_mul, "mul", DType::f32, 5},
                             // mul, i32
                             Param{&op_mul, "mul", DType::i32, 0}, Param{&op_mul, "mul", DType::i32, 1},
                             Param{&op_mul, "mul", DType::i32, 2}, Param{&op_mul, "mul", DType::i32, 3},
                             Param{&op_mul, "mul", DType::i32, 4}, Param{&op_mul, "mul", DType::i32, 5}));

TEST(PointwiseBinaryNegative, BroadcastIncompatible) {
    auto a = Tensor::ones(Shape({2, 3}), DType::f32);
    auto b = Tensor::ones(Shape({4, 1}), DType::f32);
    EXPECT_THROW((void)ops::add(a, b), std::runtime_error);
    EXPECT_THROW((void)ops::mul(a, b), std::runtime_error);
}

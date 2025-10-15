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
static Tensor op_sub(const Tensor& a, const Tensor& b) { return ops::sub(a, b); }
static Tensor op_div(const Tensor& a, const Tensor& b) { return ops::div(a, b); }

struct Scenario {
    const char* name;
    std::function<std::pair<Tensor, Tensor>(DType)> make;
    std::vector<std::size_t> expected_shape;

    float expected_scalar_f32_add = 2.0f;
    float expected_scalar_f32_mul = 1.0f;
    float expected_scalar_f32_sub = 0.0f;
    float expected_scalar_f32_div = 1.0f;
    int32_t expected_scalar_i32_add = 2;
    int32_t expected_scalar_i32_mul = 1;
    int32_t expected_scalar_i32_sub = 0;
    int32_t expected_scalar_i32_div = 1;
};

static inline float get_f32_expected_value(const std::string& opname, const Scenario& sc) {
    if (opname == std::string("add")) {
        return sc.expected_scalar_f32_add;
    } else if (opname == std::string("mul")) {
        return sc.expected_scalar_f32_mul;
    } else if (opname == std::string("sub")) {
        return sc.expected_scalar_f32_sub;
    } else if (opname == std::string("div")) {
        return sc.expected_scalar_f32_div;
    } else {
        throw std::runtime_error("Not supported opname.");
    }
}

static inline float get_i32_expected_value(const std::string& opname, const Scenario& sc) {
    if (opname == std::string("add")) {
        return sc.expected_scalar_i32_add;
    } else if (opname == std::string("mul")) {
        return sc.expected_scalar_i32_mul;
    } else if (opname == std::string("sub")) {
        return sc.expected_scalar_i32_sub;
    } else if (opname == std::string("div")) {
        return sc.expected_scalar_i32_div;
    } else {
        throw std::runtime_error("Not supported opname.");
    }
}

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
        float ev = get_f32_expected_value(opname, sc);
        expect_all_eq_f32(c, ev);

        const float* pa = static_cast<const float*>(a.data());
        const float* pb = static_cast<const float*>(a_before.data());
        for (std::size_t i = 0; i < std::min<std::size_t>(a.numel(), 6); ++i) {
            EXPECT_FLOAT_EQ(pa[i], pb[i]);
        }
    } else if (dt == DType::i32) {
        int32_t ev = get_i32_expected_value(opname, sc);
        expect_all_eq_i32(c, ev);
    } else {
        FAIL() << "Unsupported dtype in param test";
    }
}

INSTANTIATE_TEST_SUITE_P(BinaryOps_All, PointwiseBinaryTest,
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
                             Param{&op_mul, "mul", DType::i32, 4}, Param{&op_mul, "mul", DType::i32, 5},
                             // sub, f32
                             Param{&op_sub, "sub", DType::f32, 0}, Param{&op_sub, "sub", DType::f32, 1},
                             Param{&op_sub, "sub", DType::f32, 2}, Param{&op_sub, "sub", DType::f32, 3},
                             Param{&op_sub, "sub", DType::f32, 4}, Param{&op_sub, "sub", DType::f32, 5},
                             // sub, i32
                             Param{&op_sub, "sub", DType::i32, 0}, Param{&op_sub, "sub", DType::i32, 1},
                             Param{&op_sub, "sub", DType::i32, 2}, Param{&op_sub, "sub", DType::i32, 3},
                             Param{&op_sub, "sub", DType::i32, 4}, Param{&op_sub, "sub", DType::i32, 5},
                             // div, f32
                             Param{&op_div, "div", DType::f32, 0}, Param{&op_div, "div", DType::f32, 1},
                             Param{&op_div, "div", DType::f32, 2}, Param{&op_div, "div", DType::f32, 3},
                             Param{&op_div, "div", DType::f32, 4}, Param{&op_div, "div", DType::f32, 5},
                             // div, i32
                             Param{&op_div, "div", DType::i32, 0}, Param{&op_div, "div", DType::i32, 1},
                             Param{&op_div, "div", DType::i32, 2}, Param{&op_div, "div", DType::i32, 3},
                             Param{&op_div, "div", DType::i32, 4}, Param{&op_div, "div", DType::i32, 5}));

TEST(PointwiseBinaryNegative, BroadcastIncompatible) {
    auto a = Tensor::ones(Shape({2, 3}), DType::f32);
    auto b = Tensor::ones(Shape({4, 1}), DType::f32);
    EXPECT_THROW((void)ops::add(a, b), std::runtime_error);
    EXPECT_THROW((void)ops::mul(a, b), std::runtime_error);
    EXPECT_THROW((void)ops::sub(a, b), std::runtime_error);
    EXPECT_THROW((void)ops::div(a, b), std::runtime_error);
}

TEST(DivZeroDivision, ZeroDivisionF32) {
    auto a = Tensor::ones(Shape({2, 3}), DType::f32);
    auto b = Tensor::zeros(Shape({2, 3}), DType::f32);

    auto c = ops::div(a, b);
    const float* p = static_cast<const float*>(c.data());
    std::vector<float> v(p, p + c.numel());
    for (float x : v) EXPECT_TRUE(std::isnan(x));
}

TEST(DivZeroDivison, ZeroDivisionI32) {
    auto a = Tensor::ones(Shape({2, 3}), DType::i32);
    auto b = Tensor::zeros(Shape({2, 3}), DType::i32);

    EXPECT_THROW((void)ops::div(a, b), std::runtime_error);
}

using TypePromotionParam = std::tuple<OpFn, const char*, DType, DType, DType>;

class BinaryTypePromotionTest : public ::testing::TestWithParam<TypePromotionParam> {};

TEST_P(BinaryTypePromotionTest, Works) {
    OpFn op;
    DType a_type, b_type, expected;
    const char* opname;
    std::tie(op, opname, a_type, b_type, expected) = GetParam();

    Tensor a = Tensor::ones(Shape({2, 3}), a_type);
    Tensor b = Tensor::ones(Shape({2, 3}), b_type);

    Tensor c = op(a, b);

    EXPECT_EQ(c.dtype(), expected);

    if (expected == DType::f32) {
        auto* z = static_cast<float*>(c.data());
        for (std::size_t i = 0; i < c.numel(); i++) {
            if (std::string(opname) == std::string("add")) {
                EXPECT_FLOAT_EQ(z[i], 2.0f);
            } else if (std::string(opname) == std::string("mul")) {
                EXPECT_FLOAT_EQ(z[i], 1.0f);
            } else if (std::string(opname) == std::string("sub")) {
                EXPECT_FLOAT_EQ(z[i], 0.0f);
            } else if (std::string(opname) == std::string("div")) {
                EXPECT_FLOAT_EQ(z[i], 1.0f);
            }
        }
    } else if (expected == DType::i32) {
        auto* z = static_cast<int32_t*>(c.data());
        for (std::size_t i = 0; i < c.numel(); i++) {
            if (std::string(opname) == std::string("add")) {
                EXPECT_EQ(z[i], 2);
            } else if (std::string(opname) == std::string("mul")) {
                EXPECT_EQ(z[i], 1);
            } else if (std::string(opname) == std::string("sub")) {
                EXPECT_EQ(z[i], 0);
            } else if (std::string(opname) == std::string("div")) {
                EXPECT_EQ(z[i], 1);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(BinaryPromotion_All, BinaryTypePromotionTest,
                         ::testing::Values(TypePromotionParam{&op_add, "add", DType::f32, DType::i32, DType::f32},
                                           TypePromotionParam{&op_add, "add", DType::i32, DType::f32, DType::f32},
                                           TypePromotionParam{&op_mul, "mul", DType::f32, DType::i32, DType::f32},
                                           TypePromotionParam{&op_mul, "mul", DType::i32, DType::f32, DType::f32},
                                           TypePromotionParam{&op_sub, "sub", DType::f32, DType::i32, DType::f32},
                                           TypePromotionParam{&op_sub, "sub", DType::i32, DType::f32, DType::f32},
                                           TypePromotionParam{&op_div, "div", DType::f32, DType::i32, DType::f32},
                                           TypePromotionParam{&op_div, "div", DType::i32, DType::f32, DType::f32}));
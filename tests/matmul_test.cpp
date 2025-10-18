#include <gtest/gtest.h>
#include <minidl/ops.h>
#include <minidl/tensor.h>

#include <ostream>

namespace minidl {
inline void PrintTo(const Shape& s, std::ostream* os) {
    *os << "Shape(";
    const auto& d = s.dims();
    for (size_t i = 0; i < d.size(); ++i) {
        if (i) *os << ",";
        *os << d[i];
    }
    *os << ")";
}
}  // namespace minidl

using namespace minidl;

using Native2DParam = std::tuple<Shape, Shape, Shape>;
class Native2DMatmulShapeTest : public ::testing::TestWithParam<Native2DParam> {};

TEST_P(Native2DMatmulShapeTest, Works) {
    Shape a_shape, b_shape, expected;
    std::tie(a_shape, b_shape, expected) = GetParam();

    auto a = Tensor::zeros(a_shape, DType::f32);
    auto b = Tensor::zeros(b_shape, DType::f32);

    auto c = ops::matmul(a, b);
    EXPECT_EQ(c.shape().dims(), expected.dims());
}

std::string ShapeParamName(const ::testing::TestParamInfo<Native2DMatmulShapeTest::ParamType>& info) {
    const auto& [a, b, out] = info.param;
    auto vec_to_str = [](const Shape& s) {
        std::string res = "";
        for (auto d : s.dims()) res += std::to_string(d) + "x";
        if (!res.empty()) res.pop_back();  // remove last 'x'
        return res;
    };
    return "A" + vec_to_str(a) + "_B" + vec_to_str(b) + "_C" + vec_to_str(out);
}

INSTANTIATE_TEST_SUITE_P(Native2DMatmulShapeTestAll, Native2DMatmulShapeTest,
                         ::testing::Values(Native2DParam(Shape({2, 3}), Shape({3, 2}), Shape({2, 2})),
                                           Native2DParam(Shape({3, 2}), Shape({2, 3}), Shape({3, 3})),
                                           Native2DParam(Shape({2, 2}), Shape({2, 2}), Shape({2, 2})),
                                           Native2DParam(Shape({1, 3}), Shape({3, 2}), Shape({1, 2})),
                                           Native2DParam(Shape({3, 3}), Shape({3, 1}), Shape({3, 1}))),
                         ShapeParamName);

TEST(Native2DMatmul, F32Matmul) {
    auto a = Tensor::zeros(Shape({2, 3}), DType::f32);
    float* x = static_cast<float*>(a.data());
    x[0] = 2.1f;
    x[1] = -0.2f;
    x[2] = -0.2f;
    x[3] = 0.6f;
    x[4] = 1.2f;
    x[5] = -1.1f;

    auto b = Tensor::zeros(Shape({3, 2}), DType::f32);
    float* y = static_cast<float*>(b.data());
    y[0] = -1.4f;
    y[1] = 0.2;
    y[2] = 0.0f;
    y[3] = 0.2f;
    y[4] = 1.6f;
    y[5] = 1.8f;

    auto c = ops::matmul(a, b);
    const float* z = static_cast<const float*>(c.data());
    EXPECT_NEAR(z[0], -3.26f, 1e-6f);
    EXPECT_NEAR(z[1], 0.02f, 1e-6f);
    EXPECT_NEAR(z[2], -2.6f, 1e-6f);
    EXPECT_NEAR(z[3], -1.62f, 1e-6f);
}

TEST(Native2DMatmul, I32Matmul) {
    auto a = Tensor::zeros(Shape({2, 3}), DType::i32);
    int32_t* x = static_cast<int32_t*>(a.data());
    x[0] = 9;
    x[1] = 6;
    x[2] = 2;
    x[3] = -2;
    x[4] = 6;
    x[5] = 9;

    auto b = Tensor::zeros(Shape({3, 2}), DType::i32);
    int32_t* y = static_cast<int32_t*>(b.data());
    y[0] = 3;
    y[1] = 9;
    y[2] = 3;
    y[3] = 1;
    y[4] = -3;
    y[5] = -8;

    auto c = ops::matmul(a, b);
    const int32_t* z = static_cast<const int32_t*>(c.data());
    EXPECT_EQ(z[0], 39);
    EXPECT_EQ(z[1], 71);
    EXPECT_EQ(z[2], -15);
    EXPECT_EQ(z[3], -84);
}

TEST(Native2DMatmul, NonContig) {
    auto a = Tensor::zeros(Shape({2, 3}), DType::f32);
    float* x = static_cast<float*>(a.data());
    x[0] = 2.1f;
    x[1] = -0.2f;
    x[2] = -0.2f;
    x[3] = 0.6f;
    x[4] = 1.2f;
    x[5] = -1.1f;

    auto b = Tensor::zeros(Shape({2, 3}), DType::f32);
    float* y = static_cast<float*>(b.data());
    y[0] = -1.3f;
    y[1] = 1.5f;
    y[2] = 0.6f;
    y[3] = 0.7f;
    y[4] = -0.9f;
    y[5] = 0.6f;
    auto b_transposed = b.transpose({1, 0});

    auto c = ops::matmul(a, b_transposed);
    const float* z = static_cast<const float*>(c.data());
    EXPECT_NEAR(z[0], -3.15f, 1e-6f);
    EXPECT_NEAR(z[1], 1.53f, 1e-6f);
    EXPECT_NEAR(z[2], 0.36f, 1e-6f);
    EXPECT_NEAR(z[3], -1.32f, 1e-6f);
}
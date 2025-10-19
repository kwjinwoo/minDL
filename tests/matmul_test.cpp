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
                                           Native2DParam(Shape({3, 3}), Shape({3, 1}), Shape({3, 1})),
                                           Native2DParam(Shape({1, 2, 3}), Shape({1, 3, 2}), Shape({1, 2, 2})),
                                           Native2DParam(Shape({10, 2, 3}), Shape({10, 3, 2}), Shape({10, 2, 2})),
                                           Native2DParam(Shape({1, 3, 2, 3}), Shape({1, 3, 3, 2}), Shape({1, 3, 2, 2})),
                                           Native2DParam(Shape({2, 3, 3, 3}), Shape({2, 1, 3, 1}),
                                                         Shape({2, 3, 3, 1}))),
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

// (b,i,j) → 1D index. out is contiguous row-major with shape (B, M, N).
static inline std::size_t idx3(std::size_t b, std::size_t i, std::size_t j, std::size_t B, std::size_t M,
                               std::size_t N) {
    (void)B;  // not needed
    return (b * M + i) * N + j;
}

// ----------------------------- F32: (B,M,K) x (B,K,N) -----------------------------
TEST(BatchedNativeMatmul, F32_BxMxK_times_BxKxN) {
    const std::size_t batch = 2, M = 2, K = 3, N = 2;

    Tensor A = Tensor::zeros(Shape({batch, M, K}), DType::f32);
    float* a = static_cast<float*>(A.data());
    // batch 0
    a[idx3(0, 0, 0, batch, M, K)] = 2.1f;
    a[idx3(0, 0, 1, batch, M, K)] = -0.2f;
    a[idx3(0, 0, 2, batch, M, K)] = -0.2f;
    a[idx3(0, 1, 0, batch, M, K)] = 0.6f;
    a[idx3(0, 1, 1, batch, M, K)] = 1.2f;
    a[idx3(0, 1, 2, batch, M, K)] = -1.1f;
    // batch 1
    a[idx3(1, 0, 0, batch, M, K)] = 1.0f;
    a[idx3(1, 0, 1, batch, M, K)] = 0.0f;
    a[idx3(1, 0, 2, batch, M, K)] = -1.0f;
    a[idx3(1, 1, 0, batch, M, K)] = 2.0f;
    a[idx3(1, 1, 1, batch, M, K)] = -3.0f;
    a[idx3(1, 1, 2, batch, M, K)] = 4.0f;

    Tensor B_t = Tensor::zeros(Shape({batch, K, N}), DType::f32);
    float* b = static_cast<float*>(B_t.data());
    // batch 0
    b[idx3(0, 0, 0, batch, K, N)] = -1.4f;
    b[idx3(0, 0, 1, batch, K, N)] = 0.2f;
    b[idx3(0, 1, 0, batch, K, N)] = 0.0f;
    b[idx3(0, 1, 1, batch, K, N)] = 0.2f;
    b[idx3(0, 2, 0, batch, K, N)] = 1.6f;
    b[idx3(0, 2, 1, batch, K, N)] = 1.8f;
    // batch 1
    b[idx3(1, 0, 0, batch, K, N)] = 0.5f;
    b[idx3(1, 0, 1, batch, K, N)] = 1.0f;
    b[idx3(1, 1, 0, batch, K, N)] = -2.0f;
    b[idx3(1, 1, 1, batch, K, N)] = 0.0f;
    b[idx3(1, 2, 0, batch, K, N)] = 3.0f;
    b[idx3(1, 2, 1, batch, K, N)] = -1.0f;

    Tensor C = ops::matmul(A, B_t);
    const float* z = static_cast<const float*>(C.data());

    // batch 0 = 기존 2D 결과
    EXPECT_NEAR(z[idx3(0, 0, 0, batch, M, N)], -3.26f, 1e-6f);
    EXPECT_NEAR(z[idx3(0, 0, 1, batch, M, N)], 0.02f, 1e-6f);
    EXPECT_NEAR(z[idx3(0, 1, 0, batch, M, N)], -2.60f, 1e-6f);
    EXPECT_NEAR(z[idx3(0, 1, 1, batch, M, N)], -1.62f, 1e-6f);

    // batch 1
    EXPECT_NEAR(z[idx3(1, 0, 0, batch, M, N)], -2.5f, 1e-6f);
    EXPECT_NEAR(z[idx3(1, 0, 1, batch, M, N)], 2.0f, 1e-6f);
    EXPECT_NEAR(z[idx3(1, 1, 0, batch, M, N)], 19.0f, 1e-6f);
    EXPECT_NEAR(z[idx3(1, 1, 1, batch, M, N)], -2.0f, 1e-6f);
}

// ------------------------ F32: (B,M,K) x (K,N)  (BROADCAST B) ------------------------
TEST(BatchedNativeMatmul, F32_BxMxK_times_KxN_BroadcastB) {
    const std::size_t batch = 2, M = 2, K = 3, N = 2;

    Tensor A = Tensor::zeros(Shape({batch, M, K}), DType::f32);
    float* a = static_cast<float*>(A.data());
    // batch 0
    a[idx3(0, 0, 0, batch, M, K)] = 2.1f;
    a[idx3(0, 0, 1, batch, M, K)] = -0.2f;
    a[idx3(0, 0, 2, batch, M, K)] = -0.2f;
    a[idx3(0, 1, 0, batch, M, K)] = 0.6f;
    a[idx3(0, 1, 1, batch, M, K)] = 1.2f;
    a[idx3(0, 1, 2, batch, M, K)] = -1.1f;
    // batch 1
    a[idx3(1, 0, 0, batch, M, K)] = 1.0f;
    a[idx3(1, 0, 1, batch, M, K)] = 0.0f;
    a[idx3(1, 0, 2, batch, M, K)] = -1.0f;
    a[idx3(1, 1, 0, batch, M, K)] = 2.0f;
    a[idx3(1, 1, 1, batch, M, K)] = -3.0f;
    a[idx3(1, 1, 2, batch, M, K)] = 4.0f;

    Tensor B_t = Tensor::zeros(Shape({K, N}), DType::f32);
    float* b = static_cast<float*>(B_t.data());
    b[0 * N + 0] = -1.4f;
    b[0 * N + 1] = 0.2f;
    b[1 * N + 0] = 0.0f;
    b[1 * N + 1] = 0.2f;
    b[2 * N + 0] = 1.6f;
    b[2 * N + 1] = 1.8f;

    Tensor C = ops::matmul(A, B_t);
    const float* z = static_cast<const float*>(C.data());

    // batch 0
    EXPECT_NEAR(z[idx3(0, 0, 0, batch, M, N)], -3.26f, 1e-6f);
    EXPECT_NEAR(z[idx3(0, 0, 1, batch, M, N)], 0.02f, 1e-6f);
    EXPECT_NEAR(z[idx3(0, 1, 0, batch, M, N)], -2.60f, 1e-6f);
    EXPECT_NEAR(z[idx3(0, 1, 1, batch, M, N)], -1.62f, 1e-6f);

    // batch 1
    EXPECT_NEAR(z[idx3(1, 0, 0, batch, M, N)], -3.0f, 1e-6f);
    EXPECT_NEAR(z[idx3(1, 0, 1, batch, M, N)], -1.6f, 1e-6f);
    EXPECT_NEAR(z[idx3(1, 1, 0, batch, M, N)], 3.6f, 1e-6f);
    EXPECT_NEAR(z[idx3(1, 1, 1, batch, M, N)], 7.0f, 1e-6f);
}

// ----------------------------- I32: (B,M,K) x (B,K,N) -----------------------------
TEST(BatchedNativeMatmul, I32_BxMxK_times_BxKxN) {
    const std::size_t batch = 2, M = 2, K = 3, N = 2;

    Tensor A = Tensor::zeros(Shape({batch, M, K}), DType::i32);
    int32_t* a = static_cast<int32_t*>(A.data());
    // batch 0
    a[idx3(0, 0, 0, batch, M, K)] = 9;
    a[idx3(0, 0, 1, batch, M, K)] = 6;
    a[idx3(0, 0, 2, batch, M, K)] = 2;
    a[idx3(0, 1, 0, batch, M, K)] = -2;
    a[idx3(0, 1, 1, batch, M, K)] = 6;
    a[idx3(0, 1, 2, batch, M, K)] = 9;
    // batch 1
    a[idx3(1, 0, 0, batch, M, K)] = 1;
    a[idx3(1, 0, 1, batch, M, K)] = 2;
    a[idx3(1, 0, 2, batch, M, K)] = 3;
    a[idx3(1, 1, 0, batch, M, K)] = 4;
    a[idx3(1, 1, 1, batch, M, K)] = 5;
    a[idx3(1, 1, 2, batch, M, K)] = 6;

    Tensor B_t = Tensor::zeros(Shape({batch, K, N}), DType::i32);
    int32_t* b = static_cast<int32_t*>(B_t.data());
    // batch 0
    b[idx3(0, 0, 0, batch, K, N)] = 3;
    b[idx3(0, 0, 1, batch, K, N)] = 9;
    b[idx3(0, 1, 0, batch, K, N)] = 3;
    b[idx3(0, 1, 1, batch, K, N)] = 1;
    b[idx3(0, 2, 0, batch, K, N)] = -3;
    b[idx3(0, 2, 1, batch, K, N)] = -8;
    // batch 1
    b[idx3(1, 0, 0, batch, K, N)] = 7;
    b[idx3(1, 0, 1, batch, K, N)] = 8;
    b[idx3(1, 1, 0, batch, K, N)] = -1;
    b[idx3(1, 1, 1, batch, K, N)] = 0;
    b[idx3(1, 2, 0, batch, K, N)] = 2;
    b[idx3(1, 2, 1, batch, K, N)] = -3;

    Tensor C = ops::matmul(A, B_t);
    const int32_t* z = static_cast<const int32_t*>(C.data());

    // batch 0
    EXPECT_EQ(z[idx3(0, 0, 0, batch, M, N)], 39);
    EXPECT_EQ(z[idx3(0, 0, 1, batch, M, N)], 71);
    EXPECT_EQ(z[idx3(0, 1, 0, batch, M, N)], -15);
    EXPECT_EQ(z[idx3(0, 1, 1, batch, M, N)], -84);

    // batch 1
    EXPECT_EQ(z[idx3(1, 0, 0, batch, M, N)], 11);
    EXPECT_EQ(z[idx3(1, 0, 1, batch, M, N)], -1);
    EXPECT_EQ(z[idx3(1, 1, 0, batch, M, N)], 35);
    EXPECT_EQ(z[idx3(1, 1, 1, batch, M, N)], 14);
}

// ------------------------ I32: (B,M,K) x (K,N)  (BROADCAST B) ------------------------
TEST(BatchedNativeMatmul, I32_BxMxK_times_KxN_BroadcastB) {
    const std::size_t batch = 2, M = 2, K = 3, N = 2;

    Tensor A = Tensor::zeros(Shape({batch, M, K}), DType::i32);
    int32_t* a = static_cast<int32_t*>(A.data());
    // batch 0
    a[idx3(0, 0, 0, batch, M, K)] = 9;
    a[idx3(0, 0, 1, batch, M, K)] = 6;
    a[idx3(0, 0, 2, batch, M, K)] = 2;
    a[idx3(0, 1, 0, batch, M, K)] = -2;
    a[idx3(0, 1, 1, batch, M, K)] = 6;
    a[idx3(0, 1, 2, batch, M, K)] = 9;
    // batch 1
    a[idx3(1, 0, 0, batch, M, K)] = 1;
    a[idx3(1, 0, 1, batch, M, K)] = 2;
    a[idx3(1, 0, 2, batch, M, K)] = 3;
    a[idx3(1, 1, 0, batch, M, K)] = 4;
    a[idx3(1, 1, 1, batch, M, K)] = 5;
    a[idx3(1, 1, 2, batch, M, K)] = 6;

    Tensor B_t = Tensor::zeros(Shape({K, N}), DType::i32);
    int32_t* b = static_cast<int32_t*>(B_t.data());
    b[0 * N + 0] = 3;
    b[0 * N + 1] = 9;
    b[1 * N + 0] = 3;
    b[1 * N + 1] = 1;
    b[2 * N + 0] = -3;
    b[2 * N + 1] = -8;

    Tensor C = ops::matmul(A, B_t);
    const int32_t* z = static_cast<const int32_t*>(C.data());

    // batch 0
    EXPECT_EQ(z[idx3(0, 0, 0, batch, M, N)], 39);
    EXPECT_EQ(z[idx3(0, 0, 1, batch, M, N)], 71);
    EXPECT_EQ(z[idx3(0, 1, 0, batch, M, N)], -15);
    EXPECT_EQ(z[idx3(0, 1, 1, batch, M, N)], -84);

    // batch 1
    EXPECT_EQ(z[idx3(1, 0, 0, batch, M, N)], 0);
    EXPECT_EQ(z[idx3(1, 0, 1, batch, M, N)], -13);
    EXPECT_EQ(z[idx3(1, 1, 0, batch, M, N)], 9);
    EXPECT_EQ(z[idx3(1, 1, 1, batch, M, N)], -7);
}

TEST(BatchedNativeMatmul, F32_NonContiguous_BxMxK_times_BxKxN) {
    const std::size_t B = 2, M = 2, K = 3, N = 2;

    // 1) Build contiguous sources with shapes (B,K,M) and (B,N,K)
    //    Then transpose last two dims to get non-contiguous (B,M,K) and (B,K,N)
    Tensor A_src = Tensor::zeros(Shape({B, K, M}), DType::f32);
    float* a = static_cast<float*>(A_src.data());
    // batch 0 → values chosen to match your earlier F32 2D test (after transpose)
    // After transpose({0,2,1}), A[0] becomes:
    // [[ 2.1, -0.2, -0.2 ],
    //  [ 0.6,  1.2, -1.1 ]]
    a[idx3(0, 0, 0, B, K, M)] = 2.1f;
    a[idx3(0, 0, 1, B, K, M)] = 0.6f;
    a[idx3(0, 1, 0, B, K, M)] = -0.2f;
    a[idx3(0, 1, 1, B, K, M)] = 1.2f;
    a[idx3(0, 2, 0, B, K, M)] = -0.2f;
    a[idx3(0, 2, 1, B, K, M)] = -1.1f;
    // batch 1 → arbitrary different values
    // After transpose, A[1] =
    // [[ 1.0,  0.0, -1.0 ],
    //  [ 2.0, -3.0,  4.0 ]]
    a[idx3(1, 0, 0, B, K, M)] = 1.0f;
    a[idx3(1, 0, 1, B, K, M)] = 2.0f;
    a[idx3(1, 1, 0, B, K, M)] = 0.0f;
    a[idx3(1, 1, 1, B, K, M)] = -3.0f;
    a[idx3(1, 2, 0, B, K, M)] = -1.0f;
    a[idx3(1, 2, 1, B, K, M)] = 4.0f;

    Tensor B_src = Tensor::zeros(Shape({B, N, K}), DType::f32);
    float* b = static_cast<float*>(B_src.data());
    // batch 0 → after transpose({0,2,1}), B becomes the (3x2) from your F32 test
    // B[0]^T rows: [-1.4, 0.0, 1.6] and [0.2, 0.2, 1.8]
    b[idx3(0, 0, 0, B, N, K)] = -1.4f;
    b[idx3(0, 0, 1, B, N, K)] = 0.0f;
    b[idx3(0, 0, 2, B, N, K)] = 1.6f;
    b[idx3(0, 1, 0, B, N, K)] = 0.2f;
    b[idx3(0, 1, 1, B, N, K)] = 0.2f;
    b[idx3(0, 1, 2, B, N, K)] = 1.8f;
    // batch 1 → different values
    // B1^T rows: [0.5, -2.0, 3.0] and [1.0, 0.0, -1.0]
    b[idx3(1, 0, 0, B, N, K)] = 0.5f;
    b[idx3(1, 0, 1, B, N, K)] = -2.0f;
    b[idx3(1, 0, 2, B, N, K)] = 3.0f;
    b[idx3(1, 1, 0, B, N, K)] = 1.0f;
    b[idx3(1, 1, 1, B, N, K)] = 0.0f;
    b[idx3(1, 1, 2, B, N, K)] = -1.0f;

    // 2) Make non-contiguous views with desired shapes
    Tensor A = A_src.transpose({0, 2, 1});   // (B,K,M) -> (B,M,K) non-contig
    Tensor Bv = B_src.transpose({0, 2, 1});  // (B,N,K) -> (B,K,N) non-contig
    ASSERT_FALSE(A.is_contiguous());
    ASSERT_FALSE(Bv.is_contiguous());

    // 3) Run matmul on non-contiguous inputs
    Tensor C = ops::matmul(A, Bv);  // (B,M,N)
    ASSERT_EQ(C.shape().rank(), 3);
    ASSERT_EQ(C.shape()[0], B);
    ASSERT_EQ(C.shape()[1], M);
    ASSERT_EQ(C.shape()[2], N);
    const float* z = static_cast<const float*>(C.data());

    // 4) Build contiguous copies and compute a manual golden reference
    Tensor A_c = A.contiguous();   // (B,M,K), row-major
    Tensor B_c = Bv.contiguous();  // (B,K,N), row-major
    const float* Ap = static_cast<const float*>(A_c.data());
    const float* Bp = static_cast<const float*>(B_c.data());

    std::vector<float> ref(B * M * N, 0.f);
    for (std::size_t bth = 0; bth < B; ++bth) {
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                float acc = 0.f;
                for (std::size_t k = 0; k < K; ++k) {
                    const float aij = Ap[idx3(bth, i, k, B, M, K)];
                    const float bkj = Bp[idx3(bth, k, j, B, K, N)];
                    acc += aij * bkj;
                }
                ref[idx3(bth, i, j, B, M, N)] = acc;
            }
        }
    }

    // 5) Compare
    for (std::size_t bth = 0; bth < B; ++bth) {
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                const auto got = z[idx3(bth, i, j, B, M, N)];
                const auto exp = ref[idx3(bth, i, j, B, M, N)];
                EXPECT_NEAR(got, exp, 1e-6f) << "mismatch at (batch=" << bth << ", i=" << i << ", j=" << j << ")";
            }
        }
    }
}
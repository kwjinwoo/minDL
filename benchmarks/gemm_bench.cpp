#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "minidl/ops.h"
#include "minidl/tensor.h"

using minidl::DType;
using minidl::Shape;
using minidl::Tensor;

// -------- CLI --------
static inline std::size_t arg_i(const char* name, int argc, char** argv, std::size_t def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], name) == 0) return static_cast<std::size_t>(std::stoll(argv[i + 1]));
    return def;
}
static inline bool arg_flag(const char* name, int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return true;
    return false;
}
static inline const char* arg_str(const char* name, int argc, char** argv, const char* def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return def;
}

// -------- data init --------
static void fill_random_f32(Tensor& t, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* p = static_cast<float*>(t.data());
    const std::size_t n = t.numel();
    for (std::size_t i = 0; i < n; ++i) p[i] = dist(rng);
}

// naive ref (contiguous, row-major)
static void ref_batched_gemm_f32(const float* A, const float* B, float* C, std::size_t Bn, std::size_t M, std::size_t K,
                                 std::size_t N) {
    const std::size_t a_bs = M * K, b_bs = K * N, c_bs = M * N;
    for (std::size_t b = 0; b < Bn; ++b) {
        const float* Ab = A + b * a_bs;
        const float* Bb = B + b * b_bs;
        float* Cb = C + b * c_bs;
        for (std::size_t i = 0; i < M; ++i) {
            const float* Ai = Ab + i * K;
            float* Ci = Cb + i * N;
            for (std::size_t j = 0; j < N; ++j) {
                float acc = 0.f;
                for (std::size_t k = 0; k < K; ++k) acc += Ai[k] * Bb[k * N + j];
                Ci[j] = acc;
            }
        }
    }
}

static double gflops(std::size_t Bn, std::size_t M, std::size_t K, std::size_t N, double sec) {
    long double ops = static_cast<long double>(Bn) * 2.0L * M * K * N;
    return static_cast<double>(ops / 1e9L / sec);
}

int main(int argc, char** argv) {
    std::size_t Bn = arg_i("--B", argc, argv, 16);
    std::size_t M = arg_i("--M", argc, argv, 256);
    std::size_t K = arg_i("--K", argc, argv, 256);
    std::size_t N = arg_i("--N", argc, argv, 256);
    std::size_t warmup = arg_i("--warmup", argc, argv, 1);
    std::size_t iters = arg_i("--iters", argc, argv, 5);
    std::size_t seed = arg_i("--seed", argc, argv, 42);
    bool check = arg_flag("--check", argc, argv);
    bool verbose = arg_flag("--verbose", argc, argv);

    const char* backend = arg_str("--backend", argc, argv, "auto");  // auto|simd|native

#if defined(_WIN32)
    _putenv_s("MINIDL_MATMUL_BACKEND", backend);
#else
    setenv("MINIDL_MATMUL_BACKEND", backend, 1);
#endif

    int threads = static_cast<int>(arg_i("--threads", argc, argv,
#ifdef _OPENMP
                                         omp_get_max_threads()
#else
                                         1
#endif
                                             ));
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    std::cout << "[miniDL GEMM bench]\n";
    std::cout << "B=" << Bn << " M=" << M << " K=" << K << " N=" << N << "  warmup=" << warmup << " iters=" << iters
              << "  dtype=f32  threads=" << threads << "  backend=" << backend << "\n";

    // Shapes: A[B,M,K], B[B,K,N]
    Tensor A = Tensor::zeros(Shape({Bn, M, K}), DType::f32);
    Tensor B = Tensor::zeros(Shape({Bn, K, N}), DType::f32);
    fill_random_f32(A, static_cast<uint32_t>(seed));
    fill_random_f32(B, static_cast<uint32_t>(seed + 1));

    // warmup
    for (std::size_t w = 0; w < warmup; ++w) {
        Tensor Cw = minidl::ops::matmul(A, B);
        (void)Cw;
    }

    // timed
    double best = 1e100, sum = 0.0;
    for (std::size_t it = 0; it < iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        Tensor C = minidl::ops::matmul(A, B);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        best = std::min(best, secs);
        sum += secs;
        if (verbose) {
            std::cout << "iter " << it << ": " << std::fixed << std::setprecision(6) << secs << " s, "
                      << std::setprecision(2) << gflops(Bn, M, K, N, secs) << " GFLOP/s\n";
        }
    }
    const double avg = sum / std::max<std::size_t>(1, iters);
    std::cout << "\nResult:\n";
    std::cout << "  best: " << std::fixed << std::setprecision(6) << best << " s  (" << std::setprecision(2)
              << gflops(Bn, M, K, N, best) << " GFLOP/s)\n";
    std::cout << "  avg : " << std::fixed << std::setprecision(6) << avg << " s  (" << std::setprecision(2)
              << gflops(Bn, M, K, N, avg) << " GFLOP/s)\n";

    if (check) {
        Tensor C = minidl::ops::matmul(A, B);
        std::vector<float> Cref(C.numel(), 0.0f);
        ref_batched_gemm_f32(static_cast<const float*>(A.data()), static_cast<const float*>(B.data()), Cref.data(), Bn,
                             M, K, N);
        const float* Cp = static_cast<const float*>(C.data());
        double max_abs = 0.0, max_rel = 0.0, rmse = 0.0;
        for (std::size_t i = 0; i < C.numel(); ++i) {
            double diff = std::abs(static_cast<double>(Cp[i]) - static_cast<double>(Cref[i]));
            double denom = std::max(1e-12, std::abs(static_cast<double>(Cref[i])));
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, diff / denom);
            rmse += diff * diff;
        }
        rmse = std::sqrt(rmse / static_cast<double>(C.numel()));
        std::cout << "Check vs naive: max_abs=" << std::setprecision(6) << max_abs << "  max_rel=" << max_rel
                  << "  rmse=" << rmse << "\n";
    }
    return 0;
}

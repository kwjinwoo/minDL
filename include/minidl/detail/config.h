#pragma once
#include <atomic>

namespace minidl::detail {

enum class MatmulBackend { Auto, ForceSIMD, ForceNative };

MatmulBackend matmul_backend_from_env();
extern std::atomic<MatmulBackend> g_matmul_backend;

}  // namespace minidl::detail

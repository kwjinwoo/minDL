#include "minidl/detail/config.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace minidl::detail {

MatmulBackend matmul_backend_from_env() {
    const char* s = std::getenv("MINIDL_MATMUL_BACKEND");
    if (!s) return MatmulBackend::Auto;

    std::string v(s);
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v == "simd") return MatmulBackend::ForceSIMD;
    if (v == "native") return MatmulBackend::ForceNative;
    return MatmulBackend::Auto;
}

std::atomic<MatmulBackend> g_matmul_backend{matmul_backend_from_env()};

}  // namespace minidl::detail

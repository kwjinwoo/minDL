#include "minidl/detail/layout.h"

namespace minidl::detail {

std::vector<std::size_t> default_strides(const std::vector<std::size_t>& shape) {
    // stride in element
    const std::int64_t r = static_cast<const std::int64_t>(shape.size());
    std::vector<std::size_t> strides;

    // empty strides
    if (r == 0) return strides;

    strides.resize(static_cast<size_t>(r));
    strides[static_cast<size_t>(r - 1)] = 1;

    for (std::int64_t i = r - 2; i >= 0; i--) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

bool is_contiguous(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& strides) {
    const std::int64_t r = static_cast<const std::int64_t>(shape.size());
    std::size_t expected = 1;
    for (std::int64_t d = r - 1; d >= 0; d--) {
        const std::size_t dim = shape[static_cast<std::size_t>(d)];
        const std::size_t s = strides[static_cast<std::size_t>(d)];

        if (dim == 1) continue;

        if (s != expected) return false;

        expected *= dim;
    }
    return true;
}

}  // namespace minidl::detail

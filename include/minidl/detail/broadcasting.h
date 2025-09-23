#pragma once
#include <cstddef>
#include <vector>

namespace minidl::detail {

inline std::vector<std::size_t> compute_broadcast_shape_stub(const std::vector<std::size_t>& a,
                                                             const std::vector<std::size_t>& b) {
    // TODO
    return (a.size() >= b.size()) ? a : b;
}

}  // namespace minidl::detail

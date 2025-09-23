#pragma once
#include <cstdint>
#include <vector>

namespace minidl::detail {

inline std::vector<std::int64_t> default_strides_stub(const std::vector<std::size_t>& /*shape*/,
                                                      std::size_t /*itemsize*/) {
    // TODO
    return {};
}

}  // namespace minidl::detail

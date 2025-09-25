#pragma once
#include <cstdint>
#include <vector>

namespace minidl::detail {

std::vector<std::size_t> default_strides(const std::vector<std::size_t>& /*shape*/);
bool is_contiguous(const std::vector<std::size_t>& /*shape*/, const std::vector<std::size_t>& /*strides*/);
}  // namespace minidl::detail

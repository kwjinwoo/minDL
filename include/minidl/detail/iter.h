#pragma once
#include <cstddef>
#include <vector>

#include "minidl/dtype.h"

namespace minidl::detail {

struct NdCounter {
    std::vector<std::size_t> shape;
    std::vector<std::size_t> idx;
    bool finished = false;

    explicit NdCounter(std::vector<std::size_t> s) : shape(std::move(s)), idx(shape.size(), 0) {
        for (auto d : shape) {
            if (d == 0) {
                finished = true;
                break;
            }
        }
    }
    bool done() const { return finished; }
    void next();
};

inline std::size_t offset_elems(const std::vector<std::size_t>& idx, const std::vector<std::size_t>& stride) {
    if (stride.empty()) return 0;

    std::size_t offset = 0;
    for (std::size_t i = 0; i < idx.size(); i++) {
        offset += idx[i] * stride[i];
    }
    return offset;
}

struct ElementReader {
    const void* base_;
    DType dtype_;

    explicit ElementReader(void* base, DType dtype) : base_(base), dtype_(dtype) {}

    template <typename T>
    inline T read_as(std::size_t idx) const {
        switch (dtype_) {
            case DType::f32:
                return static_cast<T>(static_cast<const float*>(base_)[idx]);
            case DType::i32:
                return static_cast<T>(static_cast<const int32_t*>(base_)[idx]);
            default:
                throw std::runtime_error("Element Reader: Unsupported Type.");
        }
    }
};

// Linear Indexing
inline std::vector<std::size_t> compute_radix(const std::vector<std::size_t>& dims) {
    const auto n = dims.size();
    std::vector<std::size_t> radix(n, 1);
    for (std::size_t i = n; i > 1; i--) {
        radix[i - 2] = radix[i - 1] * dims[i - 1];
    }
    return radix;
}

inline std::size_t linear_to_offset(const std::size_t linear_idx, const std::vector<std::size_t>& dims,
                                    const std::vector<std::size_t>& radix, const std::vector<std::size_t>& strides) {
    std::size_t off = 0;
    const std::size_t r = dims.size();
    for (std::size_t d = 0; d < r; d++) {
        off += ((linear_idx / radix[d]) % dims[d]) * strides[d];
    }
    return off;
}

}  // namespace minidl::detail

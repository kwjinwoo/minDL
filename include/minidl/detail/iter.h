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

}  // namespace minidl::detail

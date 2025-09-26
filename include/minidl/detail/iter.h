#pragma once
#include <cstddef>
#include <vector>

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

}  // namespace minidl::detail

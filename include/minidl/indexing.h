#pragma once
#include <vector>

namespace minidl {

struct NdCounter {
    const std::vector<std::size_t>& shape;
    std::vector<std::size_t> idx;
    bool finished = false;

    explicit NdCounter(const std::vector<std::size_t>& s) : shape(s), idx(s.size(), 0) {
        for (auto d : shape) {
            if (d == 0) {
                finished = true;
                break;
            }
        }
    }
    bool done() const { return finished; }

    void next() {
        if (finished) return;
        if (idx.empty()) {
            finished = true;
            return;
        }

        std::int64_t d = static_cast<std::int64_t>(idx.size() - 1);
        while (d >= 0) {
            idx[static_cast<std::size_t>(d)] += 1;
            if (idx[static_cast<std::size_t>(d)] < shape[static_cast<std::size_t>(d)]) {
                return;
            }
            idx[static_cast<std::size_t>(d)] = 0;
            d--;
        }
        finished = true;
    }
};

inline std::size_t offset_elems(const std::vector<std::size_t>& idx, const std::vector<std::size_t>& stride) {
    std::size_t offset = 0;
    for (std::size_t i = 0; i < idx.size(); i++) {
        offset += idx[i] * stride[i];
    }
    return offset;
}

}  // namespace minidl
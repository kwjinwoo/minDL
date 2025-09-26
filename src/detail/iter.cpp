#include "minidl/detail/iter.h"

namespace minidl::detail {

void NdCounter::next() {
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

}  // namespace minidl::detail

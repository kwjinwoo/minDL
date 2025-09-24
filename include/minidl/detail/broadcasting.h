#pragma once
#include <cstddef>
#include <vector>

namespace minidl::detail {

inline std::vector<std::size_t> compute_broadcast_shape(const std::vector<std::size_t>& a,
                                                        const std::vector<std::size_t>& b) {
    const std::int64_t ra = static_cast<std::int64_t>(a.size());
    const std::int64_t rb = static_cast<std::int64_t>(b.size());
    const std::int64_t r = std::max(ra, rb);

    std::vector<std::size_t> out(r, 1);

    for (std::int64_t i = 0; i < r; i++) {
        const std::int64_t ai = ra - 1 - i >= 0 ? a[ra - 1 - i] : 1;
        const std::int64_t bi = rb - 1 - i >= 0 ? b[rb - 1 - i] : 1;

        if (ai == bi || ai == 1 || bi == 1) {
            out[r - 1 - i] = static_cast<std::size_t>(std::max(ai, bi));
        } else {
            throw std::runtime_error("broadcast: incompatible shapes.");
        }
    }
    return out;
}

inline std::vector<std::size_t> expand_strides_for_broadcast(const std::vector<std::size_t>& in_shapes,
                                                             const std::vector<std::size_t>& in_strides,
                                                             const std::vector<std::size_t>& out_shapes) {
    const std::int64_t rin = static_cast<std::int64_t>(in_shapes.size());
    const std::int64_t rout = static_cast<std::int64_t>(out_shapes.size());

    std::vector<std::size_t> out_strides(rout, 0);

    for (std::int64_t i = 0; i < rout; i++) {
        const std::int64_t out_shape = out_shapes[rout - 1 - i];
        const std::int64_t in_shape = rin - 1 - i >= 0 ? in_shapes[rin - 1 - i] : 1;
        const std::size_t in_stride = rin - 1 - i >= 0 ? in_strides[rin - 1 - i] : 0;

        if (out_shape == in_shape)
            out_strides[rout - 1 - i] = in_stride;
        else if (in_shape == 1)
            out_strides[rout - 1 - i] = 0;
        else
            throw std::runtime_error("expand strides: incompatible shapes.");
    }
    return out_strides;
}

}  // namespace minidl::detail

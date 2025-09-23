#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace minidl::detail {

struct NdCounterStub {
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> idx;
    explicit NdCounterStub(std::vector<std::int64_t> s) : shape(std::move(s)), idx(shape.size(), 0) {}
    bool done() const { return true; }  // TODO
    void next() {}                      // TODO
};

inline std::int64_t offset_bytes_stub(const std::vector<std::int64_t>& /*idx*/,
                                      const std::vector<std::int64_t>& /*stride*/, std::size_t /*itemsize*/) {
    // TODO
    return 0;
}

}  // namespace minidl::detail

#pragma once
#include <cstdint>
#include <type_traits>

namespace minidl::detail {

template <typename F>
inline void dispatch_f32_i32_stub(int /*fake_dtype*/, F&& f) {
    // TODO
    using T = float;
    f.template operator()<T>();
}

}  // namespace minidl::detail

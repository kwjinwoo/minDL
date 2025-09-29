#pragma once
#include "minidl/dtype.h"

namespace minidl::detail {

template <typename F32Fn, typename I32Fn>
auto dispatch(DType dt, F32Fn&& f32_fn, I32Fn&& i32_fn) {
    switch (dt) {
        case DType::f32:
            return f32_fn();
        case DType::i32:
            return i32_fn();
        default:
            throw std::runtime_error("unsupported dtype");
    }
}

}  // namespace minidl::detail

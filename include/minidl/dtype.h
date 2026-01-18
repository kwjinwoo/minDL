#pragma once
#include <cstddef>
namespace minidl {
enum DType {
    f32,
    i32,
};

constexpr std::size_t size_of(const DType& dtype) {
    switch (dtype) {
        case DType::f32:
            return 4;
        case DType::i32:
            return 4;
    }
    return 0;
}

template <typename T>
constexpr DType dtype_of() {
    if constexpr (std::is_same_v<T, float>) {
        return DType::f32;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return DType::i32;
    } else {
        throw std::runtime_error("dtype_of: Unsupported C++ type.");
    }
}

}  // namespace minidl

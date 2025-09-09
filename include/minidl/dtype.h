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
}  // namespace minidl

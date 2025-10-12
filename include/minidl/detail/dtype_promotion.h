#include "algorithm"
#include "minidl/dtype.h"

namespace minidl::detail {

constexpr bool is_floating(DType d) {
    switch (d) {
        case DType::f32:
            return true;
        default:
            return false;
    }
}

constexpr int get_bitwidth(DType d) {
    switch (d) {
        case DType::f32:
        case DType::i32:
            return 32;
            // i16, i8, f64, i64 ...
        default:
            return -1;
    }
}

inline DType float_from_bitwidth(int bitwidth) {
    switch (bitwidth) {
        case 32:
            return DType::f32;
            // f16, f64 ...
        default:
            throw std::runtime_error("float_from_bitwdith: Invalid Bitwidth.");
    }
}

inline DType int_from_bitwidth(int bitwidth) {
    switch (bitwidth) {
        case 32:
            return DType::i32;
            // i8, i16 ...
        default:
            throw std::runtime_error("int_from_bitwdith: Invalid Bitwidth.");
    }
}

inline DType promote_dtype(DType a, DType b) {
    if (a == b) return a;

    const int bit_width = std::max(get_bitwidth(a), get_bitwidth(b));
    if (is_floating(a) || is_floating(b)) {
        return float_from_bitwidth(bit_width);
    } else {
        return int_from_bitwidth(bit_width);
    }
}

}  // namespace minidl::detail

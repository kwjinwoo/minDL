#pragma once
#include <initializer_list>
#include <numeric>
#include <vector>

namespace minidl {
class Shape {
   public:
    Shape() = default;
    explicit Shape(const std::vector<std::size_t>& dims) : dims_(dims) {}
    Shape(std::initializer_list<std::size_t> dims) : dims_(dims) {}

    // copy & move
    Shape(const Shape& other) = default;
    Shape& operator=(const Shape& other) = default;

    Shape(Shape&& other) noexcept = default;
    Shape& operator=(Shape&& other) noexcept = default;

    // getter
    std::size_t rank() const noexcept { return dims_.size(); }
    const std::vector<std::size_t>& dims() const noexcept { return dims_; }
    std::size_t operator[](std::size_t i) const noexcept { return dims_[i]; }

    // utils
    std::size_t numel() const noexcept {
        if (dims_.empty()) return 1;

        bool has_zero = false;
        for (auto d : dims_) {
            if (d == 0) {
                has_zero = true;
                break;
            }
        }
        if (has_zero) return 0;
        return std::accumulate(dims_.begin(), dims_.end(), std::size_t{1},
                               [](std::size_t a, std::size_t b) { return a * b; });
    }

   private:
    void validate() const {
        for (auto d : dims_) {
            if (d < 0) throw std::invalid_argument("Shape: negative dimension not allowed");
        }
    }
    std::vector<std::size_t> dims_;
};
}  // namespace minidl

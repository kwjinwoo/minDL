#pragma once
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace minidl {
class Shape {
   public:
    Shape() = default;
    explicit Shape(const std::vector<int64_t>& dims) : dims_(dims) {};
    Shape(std::initializer_list<int64_t> dims) : dims_(dims) {};

    // copy & move
    Shape(const Shape& other) = default;
    Shape& operator=(const Shape& other) = default;

    Shape(Shape&& other) noexcept = default;
    Shape& operator=(Shape&& other) noexcept = default;

    // getter
    std::size_t rank() const noexcept { return dims_.size(); }
    const std::vector<int64_t>& dims() const noexcept { return dims_; }
    int64_t operator[](int64_t i) const noexcept { return dims_[i]; }

   private:
    void validate() const {
        for (auto d : dims_) {
            if (d < 0) throw std::invalid_argument("Shape: negative dimension not allowed");
        }
    }
    std::vector<int64_t> dims_;
};
}  // namespace minidl

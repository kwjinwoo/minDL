#pragma once
#include <minidl/dtype.h>
#include <minidl/shape.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace minidl {

// forward declaration.
class Allocator;

struct Storage {
    void* data = nullptr;
    std::size_t nbytes = 0;
    std::shared_ptr<Allocator> alloc;
};

class Tensor {
   public:
    // constructor and deleter
    Tensor() = delete;
    Tensor(const Shape& shape, DType dtype, std::shared_ptr<Storage> storage, std::vector<int64_t> strides);
    ~Tensor() = default;

    // copy and move
    Tensor(const Tensor& tensor) = default;
    Tensor& operator=(const Tensor& tensor) = default;
    Tensor(Tensor&& tensor) = default;
    Tensor& operator=(Tensor&& tensor) = default;

    // factory methods
    static Tensor randn(const Shape& s, DType d = DType::f32, std::shared_ptr<Allocator> alloc = nullptr);
    static Tensor zeros(const Shape& s, DType d = DType::f32, std::shared_ptr<Allocator> alloc = nullptr);
    static Tensor ones(const Shape& s, DType d = DType::f32, std::shared_ptr<Allocator> alloc = nullptr);

    // view & reshape
    Tensor view(const Shape& new_shape) const;
    Tensor reshape(const Shape& new_shape) const;
    Tensor t() const;

    // get methods
    const Shape& shape() const noexcept { return shape_; }
    DType dtype() const noexcept { return dtype_; }
    const std::shared_ptr<Storage>& storage() const noexcept { return storage_; }
    const std::vector<int64_t>& strides() const noexcept { return strides_; }

    std::size_t numel() const noexcept;
    std::size_t itemsize() const noexcept;
    std::size_t nbytes() const noexcept { return numel() * itemsize(); }
    bool is_contiguous() const noexcept;

    const void* data() const noexcept;
    void* data() noexcept;

    Tensor contiguous() const;

   private:
    static std::vector<int64_t> default_strides(const Shape& shape);

    Shape shape_;
    DType dtype_;
    std::shared_ptr<Storage> storage_;
    std::vector<int64_t> strides_;
};

}  // namespace minidl

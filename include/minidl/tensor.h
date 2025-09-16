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
    Storage() = default;
    explicit Storage(std::shared_ptr<Allocator> alloc) : alloc_(std::move(alloc)) {};

    Storage(const Storage& other) = default;
    Storage& operator=(const Storage& other) = default;

    Storage(Storage&& other) noexcept = default;
    Storage& operator=(Storage&& other) noexcept = default;

    void* data = nullptr;
    std::size_t nbytes = 0;
    std::shared_ptr<Allocator> alloc_;
};

class Tensor {
   public:
    // constructor and deleter
    Tensor() = delete;
    Tensor(const Shape& shape, DType dtype, std::shared_ptr<Storage> storage)
        : shape_(shape), dtype_(dtype), storage_(std::move(storage)) {}
    ~Tensor() = default;

    // copy and move
    Tensor(const Tensor& tensor) = default;
    Tensor& operator=(const Tensor& tensor) = default;
    Tensor(Tensor&& tensor) = default;
    Tensor& operator=(Tensor&& tensor) = default;

    // factory methods
    // static Tensor randn(const Shape& s, DType d = DType::f32, std::shared_ptr<Allocator> alloc = nullptr);
    static Tensor zeros(const Shape& shape, DType dtype = DType::f32, std::shared_ptr<Allocator> alloc = nullptr);
    static Tensor ones(const Shape& shape, DType dtype = DType::f32, std::shared_ptr<Allocator> alloc = nullptr);
    static Tensor arange(std::size_t size, DType dtype = DType::f32, std::shared_ptr<Allocator> alloc = nullptr);

    // view & reshape
    Tensor view(const Shape& new_shape) const;
    // Tensor reshape(const Shape& new_shape) const;
    // Tensor t() const;

    // get methods
    const Shape& shape() const noexcept { return shape_; }
    DType dtype() const noexcept { return dtype_; }
    const std::shared_ptr<Storage>& storage() const noexcept { return storage_; }
    const std::vector<int64_t>& strides() const noexcept { return strides_; }
    void* data() const noexcept { return storage_->data; }

    std::size_t numel() const noexcept { return shape_.numel(); }
    std::size_t itemsize() const noexcept { return size_of(dtype_); }
    std::size_t nbytes() const noexcept { return numel() * itemsize(); }
    std::size_t rank() const noexcept { return shape_.rank(); }
    bool is_contiguous() const noexcept;

    Tensor contiguous() const;

   private:
    static std::vector<int64_t> default_strides(const Shape& shape);
    static void fill_ones_(void* data, int64_t numel, DType dtype);

    Shape shape_;
    DType dtype_;
    std::shared_ptr<Storage> storage_;
    std::vector<int64_t> strides_;
};

}  // namespace minidl

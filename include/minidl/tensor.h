#pragma once
#include <minidl/detail/layout.h>
#include <minidl/dtype.h>
#include <minidl/shape.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace minidl {

// forward declaration.
class Allocator;
struct GradFn;

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

struct TensorImpl {
    // value
    Shape shape;
    DType dtype;
    std::shared_ptr<Storage> storage;
    std::vector<std::size_t> strides;

    // grad
    bool requires_grad = false;
    std::shared_ptr<TensorImpl> grad;
    std::shared_ptr<GradFn> grad_fn;
};

class Tensor {
   public:
    // constructor and deleter
    Tensor() = delete;
    Tensor(const Shape& shape, DType dtype, std::shared_ptr<Storage> storage, bool requires_grad = false);
    ~Tensor();

    // copy and move
    Tensor(const Tensor& tensor) = default;
    Tensor& operator=(const Tensor& tensor) = default;
    Tensor(Tensor&& tensor) = default;
    Tensor& operator=(Tensor&& tensor) = default;

    // factory methods
    // static Tensor randn(const Shape& s, DType d = DType::f32, std::shared_ptr<Allocator> alloc = nullptr);
    static Tensor zeros(const Shape& shape, DType dtype = DType::f32, std::shared_ptr<Allocator> alloc = nullptr,
                        bool requires_grad = false);
    static Tensor ones(const Shape& shape, DType dtype = DType::f32, std::shared_ptr<Allocator> alloc = nullptr,
                       bool requires_grad = false);
    static Tensor arange(std::size_t size, DType dtype = DType::f32, std::shared_ptr<Allocator> alloc = nullptr,
                         bool requires_grad = false);
    static Tensor ones_like(const Tensor& t, std::shared_ptr<Allocator> alloc = nullptr, bool requires_grad = false);
    static Tensor from_scalar(float s, std::shared_ptr<Allocator> alloc = nullptr, bool requires_grad = false);

    // view & reshape
    Tensor view(const Shape& new_shape) const;
    Tensor reshape(const Shape& new_shape) const;
    Tensor transpose(const std::initializer_list<std::size_t> axes_ilist) const;

    // get methods
    const Shape& shape() const noexcept { return impl_->shape; }
    DType dtype() const noexcept { return impl_->dtype; }
    const std::shared_ptr<Storage>& storage() const noexcept { return impl_->storage; }
    const std::vector<std::size_t>& strides() const noexcept { return impl_->strides; }
    void* data() const noexcept { return impl_->storage->data; }
    bool requires_grad() const noexcept { return impl_->requires_grad; }

    std::shared_ptr<TensorImpl>& grad() { return impl_->grad; }
    const std::shared_ptr<TensorImpl>& grad() const { return impl_->grad; }
    std::shared_ptr<GradFn>& grad_fn() { return impl_->grad_fn; }
    const std::shared_ptr<GradFn>& grad_fn() const { return impl_->grad_fn; }

    // utils
    std::size_t numel() const noexcept { return impl_->shape.numel(); }
    std::size_t itemsize() const noexcept { return size_of(impl_->dtype); }
    std::size_t nbytes() const noexcept { return numel() * itemsize(); }
    std::size_t rank() const noexcept { return impl_->shape.rank(); }
    inline bool is_contiguous() const noexcept {
        // 0 element is conventionally contiguous
        if (numel() == 0) return true;
        return detail::is_contiguous(impl_->shape.dims(), strides());
    }
    Tensor contiguous() const;

    // to
    Tensor to(DType);

    // backward
    void backward();

   private:
    static inline std::vector<std::size_t> default_strides(const Shape& shape) {
        const auto dims = shape.dims();
        return detail::default_strides(dims);
    }
    static void fill_ones_(void* data, std::size_t numel, DType dtype);

    std::shared_ptr<TensorImpl> impl_;
};

}  // namespace minidl

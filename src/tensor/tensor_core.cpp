#include "minidl/allocators/default.h"
#include "minidl/autograd/grad_fn.h"
#include "minidl/detail/dispatch.h"
#include "minidl/detail/iter.h"
#include "minidl/tensor.h"

namespace minidl {

// constructor and deleter
Tensor::Tensor() = default;
Tensor::Tensor(const Shape& shape, DType dtype, std::shared_ptr<Storage> storage, bool requires_grad)
    : impl_(std::make_shared<TensorImpl>()) {
    impl_->shape = shape;
    impl_->dtype = dtype;
    impl_->storage = std::move(storage);
    impl_->strides = default_strides(shape);
    impl_->requires_grad = requires_grad;
}
Tensor::~Tensor() = default;

// to helpers
template <class FFloat, class FInt32>
auto dispatch_in(DType din, FFloat f_float, FInt32 f_int32) {
    return detail::dispatch(din, f_float, f_int32);
}

template <class FFloat, class FInt32>
auto dispatch_out(DType dout, FFloat f_float, FInt32 f_int32) {
    return detail::dispatch(dout, f_float, f_int32);
}

// to
Tensor Tensor::to(DType d) {
    if (d == dtype()) return *this;

    auto alloc = storage()->alloc_;
    auto storage = std::make_shared<Storage>(alloc);

    Tensor out(shape(), d, storage, requires_grad());
    out.storage()->nbytes = out.numel() * out.itemsize();

    if (out.nbytes() == 0) {
        out.storage()->data = nullptr;
        return out;
    }
    out.storage()->data = out.storage()->alloc_->allocate(out.nbytes());

    if (!out.data()) throw std::bad_alloc();

    auto converter = [&](auto* __restrict z, const auto* __restrict x) {
        using Tout = std::remove_pointer_t<decltype(z)>;

        if (is_contiguous()) {
            const std::size_t n = out.numel();
            for (std::size_t i = 0; i < n; ++i) {
                z[i] = static_cast<Tout>(x[i]);
            }
        } else {
            detail::NdCounter it(shape().dims());
            std::size_t zi = 0;
            while (!it.done()) {
                const auto xo = detail::offset_elems(it.idx, strides());
                z[zi++] = static_cast<Tout>(x[xo]);
                it.next();
            }
        }
    };

    dispatch_out(
        d,
        [&] {
            auto* z = static_cast<float*>(out.data());
            dispatch_in(
                dtype(), [&] { converter(z, static_cast<const float*>(data())); },
                [&] { converter(z, static_cast<const int32_t*>(data())); });
        },
        [&] {
            auto* z = static_cast<int32_t*>(out.data());
            dispatch_in(
                dtype(), [&] { converter(z, static_cast<const float*>(data())); },
                [&] { converter(z, static_cast<const int32_t*>(data())); });
        });
    return out;
}

// backward
void Tensor::backward() {
    if (!requires_grad()) return;
    Tensor out_grad = Tensor::ones_like(*this);
    impl_->backward(out_grad);
}

void TensorImpl::backward(const Tensor& out_grad) {
    if (!requires_grad) return;
    accumulate_grad(grad, out_grad);
    if (grad_fn) grad_fn->backward(out_grad);
}

}  // namespace minidl

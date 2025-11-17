#include "minidl/allocators/default.h"
#include "minidl/detail/dispatch.h"
#include "minidl/detail/iter.h"
#include "minidl/tensor.h"

namespace minidl {

// constructor and deleter
Tensor::Tensor(const Shape& shape, DType dtype, std::shared_ptr<Storage> storage, bool requires_grad)
    : shape_(shape), dtype_(dtype), storage_(std::move(storage)), requires_grad_(requires_grad) {}
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
    if (d == dtype_) return *this;

    auto alloc = storage_->alloc_;
    auto storage = std::make_shared<Storage>(alloc);

    Tensor out(shape_, d, storage);
    out.strides_ = out.default_strides(shape_);
    out.storage_->nbytes = out.numel() * out.itemsize();

    if (out.nbytes() == 0) {
        out.storage_->data = nullptr;
        return out;
    }
    out.storage_->data = out.storage_->alloc_->allocate(out.nbytes());

    if (!out.data()) throw std::bad_alloc();

    auto converter = [&](auto* __restrict z, const auto* __restrict x) {
        using Tout = std::remove_pointer_t<decltype(z)>;

        if (is_contiguous()) {
            const std::size_t n = out.numel();
            for (std::size_t i = 0; i < n; ++i) {
                z[i] = static_cast<Tout>(x[i]);
            }
        } else {
            detail::NdCounter it(shape_.dims());
            std::size_t zi = 0;
            while (!it.done()) {
                const auto xo = detail::offset_elems(it.idx, strides_);
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
                dtype_, [&] { converter(z, static_cast<const float*>(data())); },
                [&] { converter(z, static_cast<const int32_t*>(data())); });
        },
        [&] {
            auto* z = static_cast<int32_t*>(out.data());
            dispatch_in(
                dtype_, [&] { converter(z, static_cast<const float*>(data())); },
                [&] { converter(z, static_cast<const int32_t*>(data())); });
        });
    return out;
}

}  // namespace minidl

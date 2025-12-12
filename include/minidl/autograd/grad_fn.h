#include <memory>
#include <string>
#include <vector>

#include "minidl/ops.h"
#include "minidl/tensor.h"

namespace minidl {

inline void accumulate_grad(std::shared_ptr<Tensor>& dst, const Tensor& src) {
    if (dst) {
        Tensor tmp = ops::add(*dst, src);
        tmp.detach_();
        *dst = std::move(tmp);
    } else {
        Tensor tmp = src;
        tmp.detach_();
        dst = std::make_shared<Tensor>(std::move(tmp));
    }
}

struct GradFn : public std::enable_shared_from_this<GradFn> {
    std::string name_;

    explicit GradFn(std::string name = "GradFn") : name_(std::move(name)) {}
    virtual ~GradFn() = default;

    virtual void backward(const Tensor& out_grad) = 0;
};

struct AddBackward : public GradFn {
    Tensor a_;
    Tensor b_;

    explicit AddBackward(const Tensor& a, const Tensor& b) : GradFn("AddBackward"), a_(a), b_(b) {}

    static Tensor reduce_grad_to_shape(const Tensor& grad, const Shape& in_shape) {
        const Shape& gshape = grad.shape();
        if (gshape.dims() == in_shape.dims()) return grad;

        const std::size_t out_ndim = gshape.rank();
        const std::size_t in_ndim = in_shape.rank();

        if (in_ndim > out_ndim) throw std::runtime_error("reduce_grad_to_shape: input rank > grad rank");

        std::vector<std::size_t> axes;
        const std::size_t lead = out_ndim - in_ndim;

        for (std::size_t i = 0; i < lead; i++) {
            axes.push_back(i);
        }

        const std::vector<std::size_t> out_dims = gshape.dims();
        const std::vector<std::size_t> in_dims = in_shape.dims();

        for (std::size_t i = 0; i < in_ndim; i++) {
            std::size_t out_dim = out_dims[lead + i];
            std::size_t in_dim = in_dims[i];

            if (in_dim == 1 && out_dim > 1) {
                axes.push_back(lead + i);
            }
        }

        if (axes.empty()) return grad;

        Tensor reduced = ops::sum(grad, axes, false);
        if (reduced.shape().dims() != in_dims) {
            reduced = reduced.reshape(in_shape);
        }
        return reduced;
    }

    void backward(const Tensor& out_grad) override {
        if (a_.requires_grad()) {
            Tensor ga = reduce_grad_to_shape(out_grad, a_.shape());
            ga.detach_();
            a_.impl()->backward(ga);
        }
        if (b_.requires_grad()) {
            Tensor gb = reduce_grad_to_shape(out_grad, b_.shape());
            gb.detach_();
            b_.impl()->backward(gb);
        }
    }
};

struct MulBackward : public GradFn {
    Tensor a_;
    Tensor b_;

    explicit MulBackward(const Tensor& a, const Tensor& b) : GradFn("MulBackward"), a_(a), b_(b) {}

    void backward(const Tensor& out_grad) override {
        if (a_.requires_grad()) {
            auto ga = ops::mul(out_grad, b_);
            ga.detach_();
            a_.impl()->backward(ga);
        }

        if (b_.requires_grad()) {
            auto gb = ops::mul(out_grad, a_);
            gb.detach_();
            b_.impl()->backward(gb);
        }
    }
};

struct MatMulBackward : public GradFn {
    Tensor a_;
    Tensor b_;

    explicit MatMulBackward(const Tensor& a, const Tensor& b) : GradFn("MatMulBackward"), a_(a), b_(b) {}

    void backward(const Tensor& out_grad) override {
        if (a_.requires_grad()) {
            auto b_t = b_.transpose({1, 0});
            auto ga = ops::matmul(out_grad, b_t);
            ga.detach_();
            a_.impl()->backward(ga);
        }

        if (b_.requires_grad()) {
            auto a_t = a_.transpose({1, 0});
            auto gb = ops::matmul(a_t, out_grad);
            gb.detach_();
            b_.impl()->backward(gb);
        }
    }
};

struct ReluBackward : public GradFn {
    Tensor x_;

    explicit ReluBackward(const Tensor& x) : GradFn("ReluBackward"), x_(x) {}

    void backward(const Tensor& out_grad) override {
        if (x_.requires_grad()) {
            auto gx = Tensor::zeros(out_grad.shape());

            auto gx_data = static_cast<float*>(gx.data());
            auto x_data = static_cast<const float*>(x_.data());
            auto out_grad_data = static_cast<const float*>(out_grad.data());

            const std::size_t n = gx.numel();
            for (std::size_t i = 0; i < n; i++) {
                if (x_data[i] > 0) {
                    gx_data[i] = out_grad_data[i];
                } else {
                    gx_data[i] = 0.0f;
                }
            }
            gx.detach_();
            x_.impl()->backward(gx);
        }
    }
};

struct SigmoidBackward : public GradFn {
    Tensor x_;

    explicit SigmoidBackward(const Tensor& x) : GradFn("SigmoidBackward"), x_(x) {}

    void backward(const Tensor& out_grad) override {
        if (x_.requires_grad()) {
            auto y_val = ops::sigmoid(x_);
            y_val.detach_();

            auto gx = ops::mul(out_grad, ops::mul(y_val, ops::sub(Tensor::ones_like(y_val), y_val)));
            gx.detach_();
            x_.impl()->backward(gx);
        }
    }
};

struct TransposeBackward : public GradFn {
    Tensor x_;
    std::vector<std::size_t> axes_;

    explicit TransposeBackward(const Tensor& x, const std::vector<std::size_t>& axes)
        : GradFn("TransposeBackward"), x_(x), axes_(axes) {}

    void backward(const Tensor& out_grad) override {
        if (!x_.requires_grad()) return;

        std::vector<std::size_t> inv_axes(axes_.size());
        for (std::size_t i = 0; i < axes_.size(); i++) {
            inv_axes[axes_[i]] = i;
        }

        Tensor gx = out_grad.transpose(inv_axes);
        gx.detach_();
        x_.impl()->backward(gx);
    }
};

struct ViewBackward : public GradFn {
    Tensor x_;
    Shape origin_shape_;

    explicit ViewBackward(const Tensor& x, const Shape& s) : GradFn("ViewBackward"), x_(x), origin_shape_(s) {}

    void backward(const Tensor& out_grad) override {
        if (x_.requires_grad()) {
            Tensor gx = out_grad.view(origin_shape_);
            gx.detach_();
            x_.impl()->backward(gx);
        }
    }
};

struct ReshapeBackward : public GradFn {
    Tensor x_;
    Shape origin_shape_;

    explicit ReshapeBackward(const Tensor& x, const Shape& s) : GradFn("ReshapeBackward"), x_(x), origin_shape_(s) {}

    void backward(const Tensor& out_grad) override {
        if (x_.requires_grad()) {
            Tensor gx = out_grad.reshape(origin_shape_);
            gx.detach_();
            x_.impl()->backward(gx);
        }
    }
};

struct SumBackward : public GradFn {
    Tensor x_;
    Shape in_shape_;
    std::vector<std::size_t> axes_;
    bool keepdims_;

    explicit SumBackward(const Tensor& x, const Shape& shape, const std::vector<std::size_t>& axes, bool keepdims)
        : GradFn("SumBackward"), x_(x), in_shape_(shape), axes_(axes), keepdims_(keepdims) {}

    void backward(const Tensor& out_grad) override {
        if (!x_.requires_grad()) return;

        const auto r = in_shape_.rank();
        std::vector<bool> reduction(r, axes_.empty());
        if (!axes_.empty()) {
            for (auto axis : axes_) {
                if (axis >= r) throw std::runtime_error("SumBackward: axis of range.");
                reduction[axis] = true;
            }
        }

        Tensor gx;

        if (keepdims_) {
            if (out_grad.shape().dims() == in_shape_.dims()) {
                gx = out_grad;
            } else {
                gx = ops::mul(Tensor::ones(in_shape_, x_.dtype()), out_grad);
            }
        } else {
            const auto& out_dims = out_grad.shape().dims();
            std::vector<std::size_t> reshaped_dims;
            reshaped_dims.reserve(r);

            std::size_t out_i = 0;
            for (std::size_t i = 0; i < r; i++) {
                if (reduction[i]) {
                    reshaped_dims.push_back(1);
                } else {
                    reshaped_dims.push_back(out_dims[out_i++]);
                }
            }

            Tensor reshaped_out_grad = out_grad.view(Shape(reshaped_dims));
            gx = ops::mul(Tensor::ones(in_shape_, x_.dtype()), reshaped_out_grad);
        }

        gx.detach_();
        x_.impl()->backward(gx);
    }
};

}  // namespace minidl

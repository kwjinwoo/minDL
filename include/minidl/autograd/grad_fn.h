#include <iostream>
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

    void backward(const Tensor& out_grad) override {
        if (a_.requires_grad()) a_.impl()->backward(out_grad);
        if (b_.requires_grad()) b_.impl()->backward(out_grad);
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

    explicit SumBackward(const Tensor& x) : GradFn("SumBackward"), x_(x) {}

    void backward(const Tensor& out_grad) override {
        if (x_.requires_grad()) {
            Tensor gx = ops::mul(Tensor::ones(x_.shape(), x_.dtype()), out_grad);
            gx.detach_();
            x_.impl()->backward(gx);
        }
    }
};

}  // namespace minidl

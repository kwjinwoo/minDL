#include <memory>
#include <string>
#include <vector>

#include "minidl/ops.h"
#include "minidl/tensor.h"

namespace minidl {

inline void accumulate_grad(std::shared_ptr<Tensor>& dst, const Tensor& src) {
    if (dst) {
        *dst = ops::add(*dst, src);
    } else {
        dst = std::make_shared<Tensor>(src);
    }
}

struct GradFn : public std::enable_shared_from_this<GradFn> {
    std::string name_;

    explicit GradFn(std::string name = "GradFn") : name_(std::move(name)) {}
    virtual ~GradFn() = default;

    virtual void backward(const Tensor& out_grad) = 0;
};

struct AddBackward : public GradFn {
    std::weak_ptr<TensorImpl> a_impl_;
    std::weak_ptr<TensorImpl> b_impl_;
    SavedValue a_val_;
    SavedValue b_val_;

    explicit AddBackward(const Tensor& a, const Tensor& b)
        : GradFn("AddBackward"),
          a_impl_(a.impl()),
          b_impl_(b.impl()),
          a_val_(SavedValue::from(a)),
          b_val_(SavedValue::from(b)) {}

    void backward(const Tensor& out_grad) override {
        auto a_ = a_impl_.lock();
        auto b_ = b_impl_.lock();
        if (a_ && a_->requires_grad) a_->backward(out_grad);
        if (b_ && b_->requires_grad) b_->backward(out_grad);
    }
};

struct MulBackward : public GradFn {
    std::weak_ptr<TensorImpl> a_impl_;
    std::weak_ptr<TensorImpl> b_impl_;
    SavedValue a_val_;
    SavedValue b_val_;

    explicit MulBackward(const Tensor& a, const Tensor& b)
        : GradFn("MulBackward"),
          a_impl_(a.impl()),
          b_impl_(b.impl()),
          a_val_(SavedValue::from(a)),
          b_val_(SavedValue::from(b)) {}

    void backward(const Tensor& out_grad) override {
        auto a_ = a_impl_.lock();
        auto b_ = b_impl_.lock();

        if (a_ && a_->requires_grad) {
            auto b_tensor = b_val_.to_tensor();
            auto ga = ops::mul(out_grad, b_tensor);
            a_->backward(ga);
        }

        if (b_ && b_->requires_grad) {
            auto a_tensor = a_val_.to_tensor();
            auto gb = ops::mul(out_grad, a_tensor);
            b_->backward(gb);
        }
    }
};

struct MatMulBackward : public GradFn {
    std::weak_ptr<TensorImpl> a_impl_;
    std::weak_ptr<TensorImpl> b_impl_;
    SavedValue a_val_;
    SavedValue b_val_;

    explicit MatMulBackward(const Tensor& a, const Tensor& b)
        : GradFn("MatMulBackward"),
          a_impl_(a.impl()),
          b_impl_(b.impl()),
          a_val_(SavedValue::from(a)),
          b_val_(SavedValue::from(b)) {}

    void backward(const Tensor& out_grad) override {
        auto a_ = a_impl_.lock();
        auto b_ = b_impl_.lock();

        if (a_ && a_->requires_grad) {
            auto b_val = b_val_.to_tensor();
            auto b_t = b_val.transpose({1, 0});
            auto ga = ops::matmul(out_grad, b_t);
            a_->backward(ga);
        }

        if (b_ && b_->requires_grad) {
            auto a_val = a_val_.to_tensor();
            auto a_t = a_val.transpose({1, 0});
            auto gb = ops::matmul(a_t, out_grad);
            b_->backward(gb);
        }
    }
};

}  // namespace minidl

#include <memory>
#include <string>
#include <vector>

#include "minidl/ops.h"
#include "minidl/tensor.h"

namespace minidl {

struct GradFn : public std::enable_shared_from_this<GradFn> {
    std::string name_;

    explicit GradFn(std::string name = "GradFn") : name_(std::move(name)) {}
    virtual ~GradFn() = default;

    virtual void backward(const Tensor& out_grad) = 0;
};

struct AddBackward : public GradFn {
    Tensor* a_;
    Tensor* b_;

    explicit AddBackward(Tensor* a, Tensor* b) : GradFn("AddBackward"), a_(a), b_(b) {}
    void backward(const Tensor& out_grad) override {
        if (a_->requires_grad()) a_->grad() = std::make_shared<Tensor>(out_grad);
        if (b_->requires_grad()) b_->grad() = std::make_shared<Tensor>(out_grad);
    }
};

struct MulBackward : public GradFn {
    Tensor* a_;
    Tensor* b_;

    explicit MulBackward(Tensor* a, Tensor* b) : GradFn("MulBackward"), a_(a), b_(b) {}
    void backward(const Tensor& out_grad) override {
        if (a_->requires_grad()) a_->grad() = std::make_shared<Tensor>(ops::mul(out_grad, *b_));
        if (b_->requires_grad()) b_->grad() = std::make_shared<Tensor>(ops::mul(out_grad, *a_));
    }
};

}  // namespace minidl

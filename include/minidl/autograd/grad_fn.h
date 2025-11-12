#include <memory>
#include <string>
#include <vector>

#include "minidl/tensor.h"

namespace minidl {

using GradFnInputs = std::vector<std::weak_ptr<Tensor>>;

struct GradFn : public std::enable_shared_from_this<GradFn> {
    GradFnInputs inputs_;
    std::string name_;

    explicit GradFn(std::string name = "GradFn") : name_(std::move(name)) {}
    virtual ~GradFn() = default;

    virtual void backward(const Tensor& out_grad) = 0;
};

struct AddBackward : public GradFn {
    AddBackward() : GradFn("AddBackward") {}
    void backward(const Tensor& out_grad) override {
        auto a = inputs_[0].lock();
        auto b = inputs_[1].lock();

        if (a && a->reqires_grad()) a->grad() = std::make_shared<Tensor>(out_grad);
        if (b && b->reqires_grad()) b->grad() = std::make_shared<Tensor>(out_grad);
    }
};

}  // namespace minidl

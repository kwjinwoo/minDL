#include <vector>

#include "minidl/detail/iter.h"
#include "minidl/tensor.h"

namespace minidl {

class Optimizer {
   public:
    explicit Optimizer(std::vector<Tensor*> params) : params_(params) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual void zero_grad() const {
        for (auto* p : params_) {
            if (!p) continue;
            auto g = p->grad();
            if (g) {
                *g = Tensor::zeros_like(*p);
            }
        }
    }

   protected:
    const std::vector<Tensor*>& params() { return params_; }

   private:
    std::vector<Tensor*> params_;
};

class SGD : public Optimizer {
   public:
    explicit SGD(float lr, std::vector<Tensor*> params) : Optimizer(std::move(params)), lr_(lr) {}
    void step() override {
        for (auto* p : params()) {
            if (!p) continue;

            auto g = p->grad();
            if (!g) continue;

            if (p->dtype() != DType::f32 || g->dtype() != DType::f32) {
                throw std::runtime_error("SGD::step currently supports only f32 parameters/grads.");
            }

            if (p->shape().dims() != g->shape().dims()) {
                throw std::runtime_error("SGD::step requires grad shape == param shape.");
            }

            const std::vector<std::size_t>& shape = p->shape().dims();
            float* p_data = static_cast<float*>(p->data());
            const std::vector<std::size_t>& p_strides = p->strides();

            const float* g_data = static_cast<const float*>(g->data());
            const std::vector<std::size_t>& g_strides = g->strides();

            const std::size_t n = p->numel();
            const float lr = lr_;

            detail::NdCounter counter(p->shape().dims());
            for (; !counter.done(); counter.next()) {
                const std::vector<std::size_t>& idx = counter.idx;

                const auto p_off = detail::offset_elems(idx, p_strides);
                const auto g_off = detail::offset_elems(idx, g_strides);
                p_data[p_off] -= lr * g_data[g_off];
            }
        }
    }

   private:
    float lr_;
};

}  // namespace minidl

#include <string>
#include <vector>

#include "minidl/tensor.h"

namespace minidl::nn {

class Module {
   public:
    explicit Module(std::string name = "") : name_(std::move(name)) {}
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& x) const = 0;

    Tensor operator()(const Tensor& x) { return forward(x); }

    const std::string& name() const { return name_; }

    void register_parameter(const std::string& name, Tensor* param) { parameters_.push_back({name, param}); }
    void register_module(const std::string& name, Module* module) { children_.push_back({name, module}); }

   protected:
    std::string name_;
    std::vector<std::pair<std::string, Tensor*>> parameters_;
    std::vector<std::pair<std::string, Module*>> children_;
};

class Linear : public Module {
    Tensor weight_;
    Tensor bias_;
    bool use_bias_ = true;

   public:
    explicit Linear(std::size_t in_features, std::size_t out_features, bool use_bias = true);

    Tensor forward(const Tensor& x) const override;
    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }
};

class Sequential : public Module {
   public:
    Sequential(std::string name = "Sequantial") : Module(std::move(name)) {}

    template <typename M, typename... Args>
    M& add(std::string name, Args&&... args) {
        auto m = std::make_unique<M>(std::forward<Args>(args)...);
        M& ref = *m;

        register_module(name, m.get());
        modules_.push_back(std::move(m));
        return ref;
    }

    Tensor forward(const Tensor& x) const override {
        Tensor out = x;
        for (auto& m : modules_) {
            out = m->forward(out);
        }
        return out;
    }

   private:
    std::vector<std::unique_ptr<Module>> modules_;
};

}  // namespace minidl::nn

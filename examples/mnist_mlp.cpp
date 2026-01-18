#include <minidl/data.h>
#include <minidl/dtype.h>
#include <minidl/nn.h>
#include <minidl/ops.h>
#include <minidl/optim.h>
#include <minidl/shape.h>
#include <minidl/tensor.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace minidl;

static uint32_t read_be_u32(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f) throw std::runtime_error("IDX: failed to read u32");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

class MNISTDataset final : public Dataset {
   public:
    MNISTDataset(std::string root, bool train) : root_(std::move(root)), train_(train) { load_(); }

    std::size_t size() const override { return n_; }
    std::size_t arity() const override { return 2; }

    void get(std::size_t idx, std::vector<Tensor>& out) const override {
        if (idx >= n_) throw std::out_of_range("MNISTDataset::get idx out of range");

        out.clear();
        out.reserve(2);

        // x: [1,28,28] f32
        Tensor x = Tensor::empty(Shape({1, 28, 28}), DType::f32);
        float* xd = static_cast<float*>(x.data());
        const std::size_t off = idx * (28 * 28);
        for (std::size_t p = 0; p < 28 * 28; ++p) {
            xd[p] = float(images_[off + p]) / 255.0f;
        }

        // y: [1] i32
        Tensor y = Tensor::empty(Shape({1}), DType::i32);
        auto* yd = static_cast<int32_t*>(y.data());
        yd[0] = static_cast<int32_t>(labels_[idx]);

        out.push_back(std::move(x));
        out.push_back(std::move(y));
    }

   private:
    void load_() {
        const std::string img = root_ + "/" + (train_ ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte");
        const std::string lab = root_ + "/" + (train_ ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte");

        load_images_(img);
        load_labels_(lab);

        if (labels_.size() != n_) throw std::runtime_error("MNIST: labels count mismatch");
    }

    void load_images_(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Failed to open MNIST images: " + path);

        const uint32_t magic = read_be_u32(f);
        const uint32_t count = read_be_u32(f);
        const uint32_t rows = read_be_u32(f);
        const uint32_t cols = read_be_u32(f);

        if (magic != 2051) throw std::runtime_error("MNIST images: bad magic (expected 2051)");
        if (rows != 28 || cols != 28) throw std::runtime_error("MNIST images: expected 28x28");

        n_ = count;
        images_.resize(std::size_t(n_) * 28 * 28);

        f.read(reinterpret_cast<char*>(images_.data()), std::streamsize(images_.size()));
        if (!f) throw std::runtime_error("MNIST images: failed to read payload");
    }

    void load_labels_(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Failed to open MNIST labels: " + path);

        const uint32_t magic = read_be_u32(f);
        const uint32_t count = read_be_u32(f);

        if (magic != 2049) throw std::runtime_error("MNIST labels: bad magic (expected 2049)");
        if (n_ != 0 && count != n_) throw std::runtime_error("MNIST labels: count mismatch");

        if (n_ == 0) n_ = count;

        labels_.resize(n_);
        f.read(reinterpret_cast<char*>(labels_.data()), std::streamsize(labels_.size()));
        if (!f) throw std::runtime_error("MNIST labels: failed to read payload");
    }

   private:
    std::string root_;
    bool train_;

    uint32_t n_{0};
    std::vector<uint8_t> images_;  // flattened [n, 28, 28]
    std::vector<uint8_t> labels_;  // [n]
};

struct MnistMLP : public nn::Module {
    nn::Linear fc1{784, 256, true};
    nn::Linear fc2{256, 10, true};

    MnistMLP() : nn::Module("MnistMLP") {
        register_module("fc1", &fc1);
        register_module("fc2", &fc2);
    }

    Tensor forward(const Tensor& x) const override {
        // x: [B,1,28,28]
        Tensor h = x.view({x.shape().dims()[0], 784});
        h = ops::relu(fc1.forward(h));
        return fc2.forward(h);  // logits [B,10]
    }
};

static float accuracy_top1(const Tensor& logits, const Tensor& y_index) {
    // logits: [B, C] float32
    // y_index: [B, 1] int32 (or [B])

    const auto& dims = logits.shape().dims();
    if (dims.size() != 2) throw std::runtime_error("accuracy_top1 expects logits rank=2");
    const std::size_t B = dims[0];
    const std::size_t C = dims[1];

    const float* ld = static_cast<const float*>(logits.data());

    // y_index could be [B] or [B,1]
    const auto& ydims = y_index.shape().dims();
    const int32_t* yd = static_cast<const int32_t*>(y_index.data());
    const bool y_is_col = (ydims.size() == 2 && ydims[0] == B && ydims[1] == 1);
    const bool y_is_vec = (ydims.size() == 1 && ydims[0] == B);
    if (!y_is_col && !y_is_vec) throw std::runtime_error("accuracy_top1 expects y shape [B] or [B,1]");

    std::size_t correct = 0;
    for (std::size_t i = 0; i < B; ++i) {
        // argmax over C
        std::size_t best = 0;
        float bestv = ld[i * C + 0];
        for (std::size_t j = 1; j < C; ++j) {
            const float v = ld[i * C + j];
            if (v > bestv) {
                bestv = v;
                best = j;
            }
        }

        const int32_t yi = y_is_col ? yd[i] : yd[i];
        if (static_cast<std::size_t>(yi) == best) ++correct;
    }

    return (B == 0) ? 0.0f : static_cast<float>(correct) / static_cast<float>(B);
}

static void train_one_epoch(MnistMLP& model, Optimizer& opt, DataLoader& loader, std::size_t epoch) {
    loader.reset_epoch();

    std::size_t step = 0;
    for (auto batch : loader) {
        // batch[0]=x, batch[1]=y
        Tensor logits = model.forward(batch[0]);
        Tensor loss = ops::cross_entropy(logits, batch[1]);

        opt.zero_grad();
        loss.backward();
        opt.step();

        float loss_v = loss.item<float>();

        if (step % 100 == 0) {
            float acc = accuracy_top1(logits, batch[1]);
            std::cout << "[epoch " << epoch << " step " << step << "] loss=" << loss_v << " acc=" << acc << "\n";
        }
        ++step;
    }
}

static void evaluate(MnistMLP& model, DataLoader& loader, const std::string& name) {
    loader.reset_epoch();

    float loss_sum = 0.0f;
    float acc_sum = 0.0f;
    std::size_t steps = 0;

    for (auto batch : loader) {
        Tensor logits = model.forward(batch[0]);
        Tensor loss = ops::cross_entropy(logits, batch[1]);

        float loss_v = loss.item<float>();
        float acc = accuracy_top1(logits, batch[1]);

        loss_sum += loss_v;
        acc_sum += acc;
        ++steps;
    }

    std::cout << "[" << name << "] loss=" << (steps ? loss_sum / steps : 0.0f)
              << " acc=" << (steps ? acc_sum / steps : 0.0f) << "\n";
}

int main(int argc, char** argv) {
    std::string root = (argc >= 2) ? argv[1] : "./data/mnist";

    const std::size_t batch_size = 64;
    const std::size_t epochs = 3;
    const float lr = 1e-2f;

    MNISTDataset train_ds(root, /*train=*/true);
    MNISTDataset test_ds(root, /*train=*/false);

    DataLoader train_loader(train_ds, batch_size, /*shuffle=*/true, /*drop_last=*/false);
    DataLoader test_loader(test_ds, batch_size, /*shuffle=*/false, /*drop_last=*/false);

    MnistMLP model;
    SGD opt(lr, model.parameters());

    for (std::size_t e = 0; e < epochs; ++e) {
        train_one_epoch(model, opt, train_loader, e);
        evaluate(model, test_loader, "test");
    }
    return 0;
}
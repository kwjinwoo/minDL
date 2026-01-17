#include <numeric>
#include <random>

#include "minidl/detail/dispatch.h"
#include "minidl/tensor.h"

namespace minidl {
class Dataset {
   public:
    Dataset() = default;
    virtual ~Dataset() = default;

    virtual std::size_t size() const = 0;
    virtual void get(std::size_t idx, std::vector<Tensor>& out) const = 0;
    virtual std::size_t arity() const = 0;
};

class DataLoader {
   public:
    using Batch = std::vector<Tensor>;

    class Iterator {
       public:
        using value_type = Batch;
        Iterator(const DataLoader* loader, std::size_t pos) : loader_(loader), pos_(pos) {}

        value_type operator*() const { return loader_->make_batch(pos_); }
        Iterator& operator++() {
            pos_ += loader_->batch_size_;
            return *this;
        }

        bool operator!=(const Iterator& other) const { return pos_ != other.pos_; }

       private:
        const DataLoader* loader_;
        std::size_t pos_;
    };

    explicit DataLoader(const Dataset& dataset, std::size_t batch_size, bool shuffle, bool drop_last = false,
                        std::uint32_t seed = 42)
        : dataset_(dataset),
          batch_size_(batch_size),
          shuffle_(shuffle),
          drop_last_(drop_last),
          indices_(dataset.size()),
          rng_(seed) {
        std::iota(indices_.begin(), indices_.end(), 0);
    }
    const Dataset& dataset() const { return dataset_; }
    std::size_t batch_size() const { return batch_size_; }
    bool shuffle() const { return shuffle_; }
    bool drop_last() const { return drop_last_; }

    // Call once per epoch
    void reset_epoch() {
        const std::size_t n = dataset_.size();
        if (indices_.size() != n) {
            indices_.resize(n);
            std::iota(indices_.begin(), indices_.end(), 0);
        }

        if (shuffle_) {
            std::shuffle(indices_.begin(), indices_.end(), rng_);
        }
    }

    Iterator begin() const { return Iterator(this, 0); }
    Iterator end() const { return Iterator(this, end_pos_()); }

    std::size_t num_batches() const {
        const std::size_t n = dataset_.size();
        if (drop_last_) return n / batch_size_;
        return (n + batch_size_ - 1) / batch_size_;
    }

   private:
    const Dataset& dataset_;
    std::size_t batch_size_;
    bool shuffle_;
    bool drop_last_;
    std::vector<std::size_t> indices_;
    std::mt19937 rng_;

    std::size_t end_pos_() const {
        const std::size_t n = dataset_.size();
        if (drop_last_) return (n / batch_size_) * batch_size_;
        return n;
    }
    Batch make_batch(std::size_t pos) const {
        const std::size_t arity = dataset_.arity();
        const std::size_t n = dataset_.size();

        if (pos >= n) return {};

        // effective_B: last batch can be smaller when drop_last == false
        const std::size_t effective_B = std::min(batch_size_, n - pos);

        // If drop_last is true, end_pos_() prevents calling make_batch on tail,
        // but keep this guard for safety.
        if (drop_last_ && effective_B < batch_size_) return {};

        Batch batches;
        batches.reserve(arity);

        std::vector<Tensor> samples;
        samples.reserve(arity);

        // 1) first sample (for shape/dtype inference) uses shuffled index
        const std::size_t idx0 = indices_[pos];
        dataset_.get(idx0, samples);

        // 2) allocate batch tensors: shape = [effective_B] + sample_shape
        //    and cache per-field sample_bytes for memcpy stride.
        std::vector<std::size_t> sample_bytes;
        sample_bytes.resize(arity);

        for (std::size_t k = 0; k < arity; ++k) {
            const std::size_t sample_rank = samples[k].rank();
            const auto& sample_dims = samples[k].shape().dims();

            std::vector<std::size_t> batch_dims(sample_rank + 1);
            batch_dims[0] = effective_B;
            for (std::size_t c = 0; c < sample_rank; ++c) {
                batch_dims[c + 1] = sample_dims[c];
            }

            batches.push_back(Tensor::empty(Shape(batch_dims), samples[k].dtype()));
            sample_bytes[k] = samples[k].nbytes();
        }

        // 3) fill
        for (std::size_t i = 0; i < effective_B; ++i) {
            const std::size_t real_idx = indices_[pos + i];

            samples.clear();
            dataset_.get(real_idx, samples);

            for (std::size_t k = 0; k < arity; ++k) {
                const std::size_t bytes = sample_bytes[k];

                auto* dst = static_cast<std::uint8_t*>(batches[k].data()) + i * bytes;
                const auto* src = static_cast<const std::uint8_t*>(samples[k].data());

                std::memcpy(dst, src, bytes);
            }
        }
        return batches;
    }
};

}  // namespace minidl

#include <gtest/gtest.h>
#include <minidl/data.h>
#include <minidl/tensor.h>

using namespace minidl;

class DummyDataset : public Dataset {
   public:
    explicit DummyDataset(std::size_t n) : n_(n) {}

    std::size_t size() const override { return n_; }
    std::size_t arity() const override { return 2; }

    void get(std::size_t idx, std::vector<Tensor>& out) const override {
        out.clear();

        // x: [2] float32
        Tensor x = Tensor::empty(Shape({2}), DType::f32);
        auto* xd = static_cast<float*>(x.data());
        xd[0] = static_cast<float>(idx);
        xd[1] = static_cast<float>(idx + 1);

        // y: scalar int32
        Tensor y = Tensor::empty(Shape({}), DType::i32);
        *static_cast<int32_t*>(y.data()) = static_cast<int32_t>(idx);

        out.push_back(x);
        out.push_back(y);
    }

   private:
    std::size_t n_;
};

TEST(DataLoaderTest, SequentialBatch) {
    DummyDataset ds(/*n=*/10);
    DataLoader loader(ds, /*batch_size=*/4, /*shuffle=*/false, /*drop_last=*/true);

    loader.reset_epoch();

    auto it = loader.begin();
    auto batch = *it;

    ASSERT_EQ(batch.size(), 2u);

    // x_batch: [4, 2]
    EXPECT_EQ(batch[0].shape().dims(), (std::vector<std::size_t>{4, 2}));
    // y_batch: [4]
    EXPECT_EQ(batch[1].shape().dims(), (std::vector<std::size_t>{4}));

    auto* x = static_cast<float*>(batch[0].data());
    auto* y = static_cast<int32_t*>(batch[1].data());

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(x[i * 2 + 0], float(i));
        EXPECT_FLOAT_EQ(x[i * 2 + 1], float(i + 1));
        EXPECT_EQ(y[i], i);
    }
}

TEST(DataLoaderTest, ShuffleDeterministic) {
    DummyDataset ds(8);

    DataLoader loader1(ds, 4, /*shuffle=*/true, /*drop_last=*/true, /*seed=*/123);
    DataLoader loader2(ds, 4, /*shuffle=*/true, /*drop_last=*/true, /*seed=*/123);

    loader1.reset_epoch();
    loader2.reset_epoch();

    auto b1 = *loader1.begin();
    auto b2 = *loader2.begin();

    auto* y1 = static_cast<int32_t*>(b1[1].data());
    auto* y2 = static_cast<int32_t*>(b2[1].data());

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(y1[i], y2[i]);
    }
}

TEST(DataLoaderTest, EffectiveBatchSizeLast) {
    DummyDataset ds(/*n=*/10);
    DataLoader loader(ds, /*batch_size=*/4, /*shuffle=*/false, /*drop_last=*/false);

    loader.reset_epoch();

    auto it = loader.begin();
    ++it;             // batch 0: [0,1,2,3]
    ++it;             // batch 1: [4,5,6,7]
    auto last = *it;  // batch 2: [8,9]

    // x_batch: [2, 2]
    EXPECT_EQ(last[0].shape().dims(), (std::vector<std::size_t>{2, 2}));
    EXPECT_EQ(last[1].shape().dims(), (std::vector<std::size_t>{2}));

    auto* y = static_cast<int32_t*>(last[1].data());
    EXPECT_EQ(y[0], 8);
    EXPECT_EQ(y[1], 9);
}

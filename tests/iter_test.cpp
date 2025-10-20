#include <gtest/gtest.h>
#include <minidl/detail/iter.h>
#include <minidl/tensor.h>

using namespace minidl;

TEST(NdCounterNext, TotalIterCountOneDim) {
    std::vector<std::size_t> shape({3});

    const std::size_t numel = 3;

    detail::NdCounter counter(shape);

    std::size_t counted = 0;
    for (; !counter.done(); counter.next()) counted++;

    EXPECT_EQ(numel, counted);
}

TEST(NdCounterNext, TotalIterCountTwodims) {
    std::vector<std::size_t> shape({3, 2});

    const std::size_t numel = 6;

    detail::NdCounter counter(shape);

    std::size_t counted = 0;
    for (; !counter.done(); counter.next()) counted++;

    EXPECT_EQ(numel, counted);
}

TEST(NdCounterNext, TotalIterCountThreeDims) {
    std::vector<std::size_t> shape({3, 2, 4});

    const std::size_t numel = 24;

    detail::NdCounter counter(shape);

    std::size_t counted = 0;
    for (; !counter.done(); counter.next()) counted++;

    EXPECT_EQ(numel, counted);
}

TEST(NdCounterNext, TotalIterCountScalar) {
    std::vector<std::size_t> shape({});

    const std::size_t numel = 1;

    detail::NdCounter counter(shape);

    std::size_t counted = 0;
    for (; !counter.done(); counter.next()) counted++;

    EXPECT_EQ(numel, counted);
}

TEST(NdCounterNext, ScalarVisitsExactlyOnce) {
    std::vector<std::size_t> shape({});
    detail::NdCounter c(shape);

    ASSERT_FALSE(c.done());
    c.next();
    ASSERT_TRUE(c.done());
}

TEST(NdCounterNext, RowMajorOrder3x2) {
    std::vector<std::size_t> shape({3, 2});
    detail::NdCounter c(shape);

    std::vector<std::array<std::size_t, 2>> seen;
    for (; !c.done(); c.next()) {
        const auto& idx = c.idx;
        seen.push_back({idx[0], idx[1]});
    }

    std::vector<std::array<std::size_t, 2>> expected = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 1},
    };
    EXPECT_EQ(seen, expected);
}

TEST(NdCounterNext, ZeroSizedDimNoIteration) {
    std::vector<std::size_t> shape({3, 0, 2});
    detail::NdCounter c(shape);

    size_t counted = 0;
    for (; !c.done(); c.next()) counted++;
    EXPECT_EQ(counted, 0u);  // numel == 0
}

TEST(OffsetElems, OneDimOffset) {
    std::vector<size_t> idx({2});
    std::vector<std::size_t> strides({1});

    std::size_t expected = 2;
    auto out = detail::offset_elems(idx, strides);
    EXPECT_EQ(expected, out);
}

TEST(OffsetElems, TwoDimsOffset) {
    // (2, 3) shape
    std::vector<std::size_t> idx({1, 2});
    std::vector<std::size_t> strides({3, 1});

    std::size_t expected = 5;
    auto out = detail::offset_elems(idx, strides);
    EXPECT_EQ(expected, out);
}

TEST(OffsetElems, ThreeDimsOffset) {
    // (2, 3, 4) shape
    std::vector<std::size_t> idx({1, 2, 2});
    std::vector<std::size_t> strides({12, 4, 1});

    std::size_t expected = 22;
    auto out = detail::offset_elems(idx, strides);
    EXPECT_EQ(expected, out);
}

TEST(OffsetElems, ScalarOffset) {
    std::vector<std::size_t> idx;
    std::vector<std::size_t> strides;

    std::size_t expected = 0;
    auto out = detail::offset_elems(idx, strides);
    EXPECT_EQ(expected, out);
}

TEST(OffsetElems, ZeroStrideBroadcast) {
    std::vector<std::size_t> strides({0, 1});

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            std::size_t out = detail::offset_elems({i, j}, strides);
            EXPECT_EQ(out, j);
        }
    }
}

TEST(OffsetElems, NonContiguousGeneral) {
    std::vector<std::size_t> strides({1, 3});

    // (i,j) -> i*1 + j*3
    EXPECT_EQ(detail::offset_elems({0, 0}, strides), 0u);
    EXPECT_EQ(detail::offset_elems({2, 0}, strides), 2u);
    EXPECT_EQ(detail::offset_elems({0, 2}, strides), 6u);
    EXPECT_EQ(detail::offset_elems({2, 3}, strides), 11u);
}

TEST(ElemReader, F32DataReadasI32) {
    auto a = Tensor::zeros(Shape({3}), DType::f32);
    auto* x = static_cast<float*>(a.data());
    x[0] = 1.0f;
    x[1] = -2.8;
    x[2] = -29.4;

    detail::ElementReader ra(a.data(), a.dtype());

    EXPECT_EQ(ra.read_as<int32_t>(0), 1);
    EXPECT_EQ(ra.read_as<int32_t>(1), -2);
    EXPECT_EQ(ra.read_as<int32_t>(2), -29);
}

TEST(ElemReader, I32DataReadasF32) {
    auto a = Tensor::zeros(Shape({3}), DType::i32);
    auto* x = static_cast<int32_t*>(a.data());
    x[0] = 1;
    x[1] = -3;
    x[2] = 0;

    detail::ElementReader ra(a.data(), a.dtype());

    EXPECT_FLOAT_EQ(ra.read_as<float>(0), 1.0f);
    EXPECT_FLOAT_EQ(ra.read_as<float>(1), -3.0f);
    EXPECT_FLOAT_EQ(ra.read_as<float>(2), 0.0f);
}

TEST(ComputeRadix, EmptyShape) {
    std::vector<std::size_t> shape;
    auto radix = detail::compute_radix(shape);

    EXPECT_TRUE(radix.empty());
}

TEST(ComputeRadix, SingleDim) {
    std::vector<std::size_t> shape({6});
    auto radix = detail::compute_radix(shape);

    EXPECT_EQ(radix[0], 1);
}

TEST(ComputeRadix, RadixMultiDim) {
    std::vector<std::size_t> shape({2, 3, 5});
    auto radix = detail::compute_radix(shape);

    EXPECT_EQ(radix[0], 15);
    EXPECT_EQ(radix[1], 5);
    EXPECT_EQ(radix[2], 1);
}

static std::vector<std::size_t> ref_unravel(std::size_t blin, const std::vector<std::size_t>& shape,
                                            const std::vector<std::size_t>& radix) {
    const std::size_t r = shape.size();
    std::vector<std::size_t> idx(r, 0);
    for (std::size_t d = 0; d < r; ++d) {
        const std::size_t base = radix[d];  // base는 0 아님 (r>0이면 마지막은 1)
        idx[d] = (base ? (blin / base) : 0) % (shape[d] ? shape[d] : 1);
    }
    return idx;
}

// num_batches = product(shape)
static std::size_t num_batches(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

TEST(LinearToOffset, EmptyShapeIsZero) {
    std::vector<std::size_t> shape{};
    std::vector<std::size_t> strides{};
    auto rad = detail::compute_radix(shape);
    EXPECT_EQ(detail::linear_to_offset(0, shape, rad, strides), 0);
}

TEST(LinearToOffset, NoBroadcast_RowMajor2D) {
    // shape: [2,3], row-major strides: [3,1]
    std::vector<std::size_t> shape{2, 3};
    std::vector<std::size_t> strides{3, 1};
    auto rad = detail::compute_radix(shape);
    const auto nb = num_batches(shape);

    for (std::size_t blin = 0; blin < nb; ++blin) {
        auto idx = ref_unravel(blin, shape, rad);
        auto expected = detail::offset_elems(idx, strides);
        auto got = detail::linear_to_offset(blin, shape, rad, strides);
        EXPECT_EQ(got, expected) << "blin=" << blin;
    }
}

TEST(LinearToOffset, WithBroadcastZeroStride) {
    std::vector<std::size_t> shape{2, 1, 4};
    std::vector<std::size_t> strides{4, 0, 1};
    auto rad = detail::compute_radix(shape);
    const auto nb = num_batches(shape);

    for (std::size_t blin = 0; blin < nb; ++blin) {
        auto idx = ref_unravel(blin, shape, rad);
        auto expected = detail::offset_elems(idx, strides);
        auto got = detail::linear_to_offset(blin, shape, rad, strides);
        EXPECT_EQ(got, expected) << "blin=" << blin;
    }
}

TEST(LinearToOffset, RandomSmall3DLike) {
    std::vector<std::size_t> shape{3, 2, 2};
    std::vector<std::size_t> strides{4, 2, 1};
    auto rad = detail::compute_radix(shape);
    const auto nb = num_batches(shape);

    for (std::size_t blin = 0; blin < nb; ++blin) {
        auto idx = ref_unravel(blin, shape, rad);
        auto expected = detail::offset_elems(idx, strides);
        auto got = detail::linear_to_offset(blin, shape, rad, strides);
        EXPECT_EQ(got, expected) << "blin=" << blin;
    }
}